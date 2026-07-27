#!/usr/bin/env python3
"""Calibrate a hard L2 bound for FD-Adv Inception pool3 features.

The image path intentionally matches ``main_fd.py``:

1. ImageFolder from ``DATA_ROOT/train`` (or DATA_ROOT itself),
2. ADM center crop to ``--img_size`` and ``ToTensor`` to [0, 1],
3. the repository's pretrained Inception loader, which resizes to 299 and
   maps [0, 1] to [-1, 1].

The script measures one L2 norm per image and recommends

    B = margin * quantile(norms, q)

It supports both a single GPU and ``torchrun``.  Distributed shards are made
explicitly, so exactly ``--num_images`` unique images are processed without
DistributedSampler padding.

Example:

    torchrun --nproc_per_node=8 scripts/calibrate_inception_feature_bound.py \
        --data_path /path/to/imagenet \
        --num_images 50000 \
        --quantile 0.999 \
        --margin 1.05
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.distributed as dist
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from frechet_distance.repr_models import load_repr_model  # noqa: E402
from utils.data_util import center_crop_arr  # noqa: E402
from utils.distributed_util import (  # noqa: E402
    enable_distributed,
    get_global_rank,
    get_local_rank,
    get_world_size,
)


logger = logging.getLogger("FD_loss")
DEFAULT_REPORT_QUANTILES = (0.0, 0.5, 0.9, 0.95, 0.99, 0.995, 0.999, 1.0)


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate B for a hard L2 cap on pretrained Inception pool3 features."
    )
    parser.add_argument(
        "--data_path",
        default=os.environ.get("DATA_ROOT"),
        help="ImageNet root containing train/, or the train/ directory itself. "
             "Defaults to the DATA_ROOT environment variable.",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=50_000,
        help="Number of unique real images to measure (default: 50000).",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=256,
        help="ADM center-crop size before Inception preprocessing (default: 256).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Per-GPU inference batch size (default: 128).",
    )
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument(
        "--sampling",
        choices=("stratified", "random"),
        default="stratified",
        help="Reproducible subset sampling strategy (default: stratified).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.999,
        help="Norm quantile used to calibrate the radius (default: 0.999).",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=1.05,
        help="Multiplicative margin applied to the selected quantile (default: 1.05).",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=Path("data/fid_stats/inception_pool3_feature_bound.json"),
        help="Path for the calibration report.",
    )
    parser.add_argument(
        "--output_norms",
        type=Path,
        default=None,
        help="Optional .npy path for all measured norms.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing output files.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.data_path:
        raise ValueError("Set --data_path or the DATA_ROOT environment variable")
    if args.num_images < 1:
        raise ValueError("--num_images must be >= 1")
    if args.img_size < 1:
        raise ValueError("--img_size must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.num_workers < 0:
        raise ValueError("--num_workers must be >= 0")
    if not 0.0 < args.quantile < 1.0:
        raise ValueError("--quantile must be strictly between 0 and 1")
    if not math.isfinite(args.margin) or args.margin < 1.0:
        raise ValueError("--margin must be finite and >= 1")
    if args.output_norms is not None and args.output_norms.suffix != ".npy":
        raise ValueError("--output_norms must end in .npy")


def resolve_imagefolder_root(data_path: str | Path) -> Path:
    root = Path(data_path).expanduser()
    train_dir = root / "train"
    if train_dir.is_dir():
        root = train_dir
    if not root.is_dir():
        raise FileNotFoundError(f"ImageFolder directory does not exist: {root}")
    return root


def _sample_random_indices(
    dataset_size: int,
    num_images: int,
    rng: np.random.Generator,
) -> np.ndarray:
    return rng.choice(dataset_size, size=num_images, replace=False).astype(np.int64)


def _sample_stratified_indices(
    targets: Sequence[int],
    num_images: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample exactly ``num_images`` while preserving class proportions."""
    labels = np.asarray(targets, dtype=np.int64)
    classes, counts = np.unique(labels, return_counts=True)
    expected = num_images * counts.astype(np.float64) / labels.size
    quotas = np.floor(expected).astype(np.int64)
    remainder = int(num_images - quotas.sum())

    if remainder:
        fractions = expected - quotas
        tie_breakers = rng.random(classes.size)
        order = sorted(
            range(classes.size),
            key=lambda index: (fractions[index], tie_breakers[index]),
            reverse=True,
        )
        quotas[np.asarray(order[:remainder], dtype=np.int64)] += 1

    # One stable sort avoids scanning the full target array once per class.
    label_order = np.argsort(labels, kind="stable")
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    selected = []
    for start, count, quota in zip(starts, counts, quotas):
        class_indices = label_order[start:start + count]
        if quota:
            selected.append(rng.choice(class_indices, size=int(quota), replace=False))

    result = np.concatenate(selected).astype(np.int64, copy=False)
    rng.shuffle(result)
    if result.size != num_images or np.unique(result).size != num_images:
        raise RuntimeError("Internal error: stratified sampling did not produce unique indices")
    return result


def select_dataset_indices(
    targets: Sequence[int],
    num_images: int,
    sampling: str,
    seed: int,
) -> np.ndarray:
    dataset_size = len(targets)
    if dataset_size == 0:
        raise ValueError("Cannot sample an empty dataset")
    num_images = min(int(num_images), dataset_size)
    rng = np.random.default_rng(seed)
    if sampling == "random":
        return _sample_random_indices(dataset_size, num_images, rng)
    if sampling == "stratified":
        return _sample_stratified_indices(targets, num_images, rng)
    raise ValueError(f"Unknown sampling strategy: {sampling}")


def build_dataset(imagefolder_root: Path, img_size: int) -> datasets.ImageFolder:
    # Keep this identical to main_fd.build_real_image_batch_fn.
    transform = transforms.Compose([
        transforms.Lambda(lambda image: center_crop_arr(image, img_size)),
        transforms.ToTensor(),
    ])
    return datasets.ImageFolder(str(imagefolder_root), transform=transform)


def build_local_loader(
    dataset: datasets.ImageFolder,
    selected_indices: np.ndarray,
    rank: int,
    world_size: int,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, np.ndarray]:
    # Explicit striding gives unique, non-padded shards even when N % world_size != 0.
    local_indices = selected_indices[rank::world_size]
    local_dataset = Subset(dataset, local_indices.tolist())
    loader = DataLoader(
        local_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    return loader, local_indices


@torch.inference_mode()
def extract_pool3_norms(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    rank: int,
) -> torch.Tensor:
    local_norms = []
    progress = tqdm(
        loader,
        desc="extracting Inception pool3 norms",
        disable=rank != 0,
        dynamic_ncols=True,
    )
    for images, _ in progress:
        images = images.to(device, non_blocking=True)
        pool3, _ = model(images)
        if pool3.ndim != 2 or pool3.shape[1] != 2048:
            raise RuntimeError(f"Expected pool3 shape [N, 2048], got {tuple(pool3.shape)}")
        # Match the FP32 training feature map, then accumulate each 2048-D norm
        # in FP64 so the tail ordering is not affected by reduction precision.
        norms = torch.linalg.vector_norm(pool3.float().double(), ord=2, dim=1)
        if not torch.isfinite(norms).all():
            raise FloatingPointError("Encountered a non-finite Inception pool3 norm")
        local_norms.append(norms.cpu())

    if not local_norms:
        return torch.empty(0, dtype=torch.float64)
    return torch.cat(local_norms)


def gather_variable_1d(
    local_values: torch.Tensor,
    device: torch.device,
    rank: int,
    world_size: int,
) -> torch.Tensor | None:
    """Gather variable-length 1-D tensors without padding the dataset itself."""
    if world_size == 1:
        return local_values.cpu()

    local_values = local_values.to(device=device, non_blocking=True)
    local_count = torch.tensor([local_values.numel()], dtype=torch.long, device=device)
    counts = [torch.zeros_like(local_count) for _ in range(world_size)]
    dist.all_gather(counts, local_count)
    counts_int = [int(count.item()) for count in counts]
    max_count = max(counts_int)

    padded = torch.zeros(max_count, dtype=local_values.dtype, device=device)
    padded[:local_values.numel()] = local_values
    gathered = [torch.empty_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded)

    if rank != 0:
        return None
    return torch.cat([
        values[:count].cpu()
        for values, count in zip(gathered, counts_int)
    ])


def _linear_quantile(values: np.ndarray, q: float) -> float:
    try:
        return float(np.quantile(values, q, method="linear"))
    except TypeError:  # NumPy < 1.22
        return float(np.quantile(values, q, interpolation="linear"))


def _quantile_key(q: float) -> str:
    return f"{q:.6f}".rstrip("0").rstrip(".")


def summarize_norms(
    norms: np.ndarray,
    calibration_quantile: float,
    margin: float,
) -> dict:
    values = np.asarray(norms, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("norms must be a non-empty 1-D array")
    if not np.isfinite(values).all():
        raise ValueError("norms contain NaN or infinity")
    if (values < 0).any():
        raise ValueError("norms must be non-negative")

    levels = sorted(set(DEFAULT_REPORT_QUANTILES + (float(calibration_quantile),)))
    quantiles = {
        _quantile_key(level): _linear_quantile(values, level)
        for level in levels
    }
    calibrated_value = _linear_quantile(values, calibration_quantile)
    recommended_bound = float(margin * calibrated_value)

    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std_population": float(values.std(ddof=0)),
        "rms": float(np.sqrt(np.mean(np.square(values)))),
        "quantiles": quantiles,
        "calibration_quantile": float(calibration_quantile),
        "calibration_quantile_value": calibrated_value,
        "margin": float(margin),
        "recommended_B": recommended_bound,
        "fraction_above_recommended_B": float(np.mean(values > recommended_bound)),
        "quantile_method": "linear",
    }


def setup_distributed() -> tuple[int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run pretrained Inception calibration")
    enable_distributed()
    rank = get_global_rank()
    world_size = get_world_size()
    device = torch.device("cuda", get_local_rank())
    return rank, world_size, device


def _ensure_output_is_writable(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}; pass --overwrite to replace it")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = get_args_parser().parse_args()
    validate_args(args)
    # Do this before process-group initialization so an existing artifact makes
    # every torchrun worker fail immediately instead of leaving peers at a barrier.
    _ensure_output_is_writable(args.output_json, args.overwrite)
    if args.output_norms is not None:
        _ensure_output_is_writable(args.output_norms, args.overwrite)
    rank, world_size, device = setup_distributed()
    if rank != 0:
        logger.setLevel(logging.WARNING)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    imagefolder_root = resolve_imagefolder_root(args.data_path)
    dataset = build_dataset(imagefolder_root, args.img_size)
    selected_indices = select_dataset_indices(
        dataset.targets,
        num_images=args.num_images,
        sampling=args.sampling,
        seed=args.seed,
    )
    effective_num_images = int(selected_indices.size)
    loader, local_indices = build_local_loader(
        dataset,
        selected_indices,
        rank=rank,
        world_size=world_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if rank == 0:
        if effective_num_images < args.num_images:
            logger.warning(
                "Requested %d images, but the dataset only contains %d; using all images",
                args.num_images,
                effective_num_images,
            )
        logger.info(
            "Calibrating Inception pool3 bound: dataset=%s, selected=%d/%d, "
            "sampling=%s, img_size=%d, gpus=%d",
            imagefolder_root,
            effective_num_images,
            len(dataset),
            args.sampling,
            args.img_size,
            world_size,
        )

    # Every rank loads the same frozen torch-fidelity Inception used by main_fd.py.
    model, feat_dim, has_logits, target_size = load_repr_model(
        "inception",
        device=device,
    )
    if feat_dim != 2048 or not has_logits:
        raise RuntimeError(
            f"Unexpected Inception metadata: feat_dim={feat_dim}, has_logits={has_logits}"
        )
    model.eval().requires_grad_(False)

    dist.barrier()
    start_time = time.perf_counter()
    local_norms = extract_pool3_norms(model, loader, device, rank)
    if local_norms.numel() != local_indices.size:
        raise RuntimeError(
            f"Rank {rank} processed {local_norms.numel()} images, expected {local_indices.size}"
        )
    all_norms = gather_variable_1d(local_norms, device, rank, world_size)
    dist.barrier()
    elapsed = time.perf_counter() - start_time

    if rank == 0:
        assert all_norms is not None
        if all_norms.numel() != effective_num_images:
            raise RuntimeError(
                f"Gathered {all_norms.numel()} norms, expected {effective_num_images}"
            )
        norms_np = all_norms.numpy().astype(np.float64, copy=False)
        norm_summary = summarize_norms(norms_np, args.quantile, args.margin)

        selected_targets = np.asarray(dataset.targets, dtype=np.int64)[selected_indices]
        _, selected_class_counts = np.unique(selected_targets, return_counts=True)
        report = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "Hard L2 radius calibration for FD-Adv Inception pool3 features",
            "feature_map": {
                "model": "inception",
                "feature_layer": "pool3",
                "feature_dim": feat_dim,
                "pretrained": True,
                "model_internal_target_size": target_size,
            },
            "preprocessing": {
                "dataset_transform": [
                    f"ADM center_crop_arr(image_size={args.img_size})",
                    "torchvision.transforms.ToTensor() -> [0, 1]",
                ],
                "inception_transform": [
                    "repository TF-style bilinear resize to 299x299",
                    "x * 2 - 1",
                ],
                "random_horizontal_flip": False,
            },
            "dataset": {
                "imagefolder_root": str(imagefolder_root.resolve()),
                "dataset_size": len(dataset),
                "selected_images": effective_num_images,
                "sampling": args.sampling,
                "seed": args.seed,
                "selected_indices_sha256": hashlib.sha256(
                    selected_indices.astype("<i8", copy=False).tobytes()
                ).hexdigest(),
                "selected_classes": int(selected_class_counts.size),
                "selected_class_count_min": int(selected_class_counts.min()),
                "selected_class_count_max": int(selected_class_counts.max()),
            },
            "distributed": {
                "world_size": world_size,
                "batch_size_per_gpu": args.batch_size,
            },
            "runtime": {
                "elapsed_seconds": elapsed,
                "images_per_second": effective_num_images / max(elapsed, 1e-12),
            },
            "norm_statistics": norm_summary,
            "recommended_B": norm_summary["recommended_B"],
        }

        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        if args.output_norms is not None:
            args.output_norms.parent.mkdir(parents=True, exist_ok=True)
            np.save(args.output_norms, norms_np.astype(np.float32))

        logger.info("Processed %d unique images in %.1fs (%.1f img/s)",
                    effective_num_images, elapsed, effective_num_images / max(elapsed, 1e-12))
        logger.info("Q(%s) = %.8g", args.quantile, norm_summary["calibration_quantile_value"])
        logger.info("Recommended B = %.8g (margin=%s)",
                    norm_summary["recommended_B"], args.margin)
        logger.info("Baseline fraction above B = %.6f", norm_summary["fraction_above_recommended_B"])
        logger.info("Saved report: %s", args.output_json)
        if args.output_norms is not None:
            logger.info("Saved norms: %s", args.output_norms)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
