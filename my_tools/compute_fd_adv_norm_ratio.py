#!/usr/bin/env python3
"""Measure adversarial-vs-original FD representation norm ratio on ImageNet.

By default this script:

1. resolves the newest checkpoint under the provided checkpoint directory,
2. loads the adversarial repr model from ``fd_adv_states``,
3. loads the matching pretrained original repr model,
4. evaluates both on the first 10,000 ImageNet images in deterministic order,
5. reports the ratio of their feature L2 norms.

The main output is ``adv_rms / ref_rms``. For convenience, the script also
prints the squared ratio used by the norm-offset regularizer.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from frechet_distance.repr_models import load_repr_model, model_short_name  # noqa: E402
from utils.data_util import center_crop_arr  # noqa: E402
from utils.distributed_util import (  # noqa: E402
    enable_distributed,
    get_global_rank,
    get_world_size,
)


LOGGER = logging.getLogger("FD_loss")

DEFAULT_CKPT_DIR = Path(
    os.environ.get("CKPT_DIR")
    or os.environ.get("CKPT_PATH")
    or (
        "/mmu-vcg/gaomingju/workspace/foundation/FD-Loss-Ours/work_dirs/"
        "Jit-adv-ablation/JiT_B-fd-sim-advinc-w0.1-from-base-ADVLR-1e-6/checkpoints"
    )
)
DEFAULT_DATA_ROOT = Path(
    os.environ.get("DATA_ROOT") or "/mmu-vcg/zhangxu34/datasets/ImageNet-1K"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute adversarial/original FD repr norm ratio on ImageNet."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CKPT_DIR,
        help="checkpoint file or directory containing step_*.pth files",
    )
    parser.add_argument(
        "--repr_model",
        type=str,
        default="inception",
        help="repr backbone name to load for both original and adversarial models",
    )
    parser.add_argument(
        "--adv_state_name",
        type=str,
        default=None,
        help="optional fd_adv_states entry name; defaults to the unique matching repr name",
    )
    parser.add_argument(
        "--pool_type",
        choices=("cls", "avg"),
        default="cls",
        help="feature output to use when the repr model exposes both cls and avg tokens",
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="ImageNet root or split directory",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default="train",
        help="ImageNet split to read",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=10_000,
        help="number of ImageNet images to use",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=256,
        help="center-crop size before the repr model's own preprocessing",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="batch size per GPU",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="DataLoader workers per rank",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="optional path to save the summary as JSON",
    )
    return parser.parse_args()


def resolve_checkpoint_path(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    latest = path / "latest.pth"
    if latest.exists():
        return latest

    step_paths = sorted(
        path.glob("step_*.pth"),
        key=_extract_step_key,
    )
    if step_paths:
        return step_paths[-1]

    all_pths = sorted(path.glob("*.pth"), key=lambda p: p.stat().st_mtime)
    if all_pths:
        return all_pths[-1]
    raise FileNotFoundError(f"no .pth files found in checkpoint directory: {path}")


def _extract_step_key(path: Path) -> int:
    stem = path.stem
    try:
        return int(stem.rsplit("_", 1)[-1])
    except ValueError:
        return -1


def resolve_split_root(data_root: Path, split: str) -> Path:
    data_root = data_root.expanduser()
    if data_root.name == split and data_root.is_dir():
        return data_root
    split_root = data_root / split
    if split_root.is_dir():
        return split_root
    raise FileNotFoundError(
        f"could not find ImageNet split directory '{split}' under {data_root}"
    )


def load_checkpoint_state(checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        return checkpoint
    return {"model": checkpoint}


def select_adv_state(
    checkpoint: dict[str, Any],
    *,
    repr_model: str,
    adv_state_name: str | None,
) -> dict[str, Any]:
    states = checkpoint.get("fd_adv_states", [])
    if not states:
        if "model" in checkpoint:
            LOGGER.warning(
                "checkpoint has no fd_adv_states; falling back to checkpoint['model']"
            )
            return {
                "name": adv_state_name or model_short_name(repr_model),
                "model": checkpoint["model"],
                "feature_transform": "unknown",
            }
        raise KeyError("checkpoint does not contain fd_adv_states or a model state")

    preferred = adv_state_name or model_short_name(repr_model)
    matches = [state for state in states if state.get("name") == preferred]
    if len(matches) == 1:
        return matches[0]
    if len(states) == 1:
        return states[0]
    if len(matches) > 1:
        raise RuntimeError(
            f"checkpoint has multiple fd_adv_states entries named {preferred!r}; "
            "please pass --adv_state_name explicitly"
        )
    available = ", ".join(sorted(str(state.get("name")) for state in states))
    raise KeyError(
        f"could not find fd_adv_states entry named {preferred!r}; available: {available}"
    )


def build_loader(
    data_root: Path,
    split: str,
    img_size: int,
    num_images: int,
    batch_size: int,
    num_workers: int,
    rank: int,
    world_size: int,
):
    split_root = resolve_split_root(data_root, split)
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda img: center_crop_arr(img, img_size)),
            transforms.ToTensor(),
        ]
    )
    dataset = datasets.ImageFolder(str(split_root), transform=transform)
    total_images = len(dataset)
    limit = min(num_images, total_images)
    indices = list(range(limit))
    local_indices = indices[rank::world_size]
    subset = Subset(dataset, local_indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader, total_images, limit, len(local_indices)


def _select_features(primary: torch.Tensor, secondary: torch.Tensor | None, pool_type: str) -> torch.Tensor:
    if pool_type == "avg":
        if secondary is None:
            raise ValueError(
                "pool_type='avg' was requested, but this repr model does not expose an avg token"
            )
        return secondary
    return primary


@torch.inference_mode()
def _forward_features(model: torch.nn.Module, images: torch.Tensor, pool_type: str) -> torch.Tensor:
    use_amp = not getattr(model, "is_inception", False)
    with torch.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
        primary, secondary = model(images)
    return _select_features(primary, secondary, pool_type)


@torch.inference_mode()
def accumulate_norms(
    ref_model: torch.nn.Module,
    adv_model: torch.nn.Module,
    loader: DataLoader,
    pool_type: str,
    rank: int,
    world_size: int,
):
    device = torch.device("cuda")
    adv_sq_sum = torch.zeros((), dtype=torch.float64, device=device)
    ref_sq_sum = torch.zeros((), dtype=torch.float64, device=device)
    adv_l2_sum = torch.zeros((), dtype=torch.float64, device=device)
    ref_l2_sum = torch.zeros((), dtype=torch.float64, device=device)
    count = 0

    desc = f"[rank {rank}] norm stats" if world_size > 1 else "norm stats"
    pbar = tqdm(loader, desc=desc, position=rank, disable=False)
    for images, _ in pbar:
        images = images.to(device, non_blocking=True)
        ref_feats = _forward_features(ref_model, images, pool_type).float()
        adv_feats = _forward_features(adv_model, images, pool_type).float()

        ref_l2 = torch.linalg.vector_norm(ref_feats, ord=2, dim=-1)
        adv_l2 = torch.linalg.vector_norm(adv_feats, ord=2, dim=-1)

        ref_sq_sum.add_(ref_l2.square().sum().double())
        adv_sq_sum.add_(adv_l2.square().sum().double())
        ref_l2_sum.add_(ref_l2.sum().double())
        adv_l2_sum.add_(adv_l2.sum().double())
        count += int(images.shape[0])
        pbar.set_postfix({"images": count})

    if world_size > 1:
        torch.distributed.reduce(ref_sq_sum, dst=0, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.reduce(adv_sq_sum, dst=0, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.reduce(ref_l2_sum, dst=0, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.reduce(adv_l2_sum, dst=0, op=torch.distributed.ReduceOp.SUM)
        count_t = torch.tensor([count], dtype=torch.long, device=device)
        torch.distributed.reduce(count_t, dst=0, op=torch.distributed.ReduceOp.SUM)
        count = int(count_t.item())

    if rank != 0:
        return None
    if count < 1:
        raise RuntimeError("no images were processed")

    ref_second_moment = float(ref_sq_sum.item() / count)
    adv_second_moment = float(adv_sq_sum.item() / count)
    ref_rms = math.sqrt(ref_second_moment)
    adv_rms = math.sqrt(adv_second_moment)
    ref_mean_l2 = float(ref_l2_sum.item() / count)
    adv_mean_l2 = float(adv_l2_sum.item() / count)

    if ref_rms <= 0.0:
        raise RuntimeError("reference repr norm is zero")

    return {
        "num_images": count,
        "adv_second_moment": adv_second_moment,
        "ref_second_moment": ref_second_moment,
        "second_moment_ratio": adv_second_moment / ref_second_moment,
        "adv_rms": adv_rms,
        "ref_rms": ref_rms,
        "rms_ratio": adv_rms / ref_rms,
        "adv_mean_l2": adv_mean_l2,
        "ref_mean_l2": ref_mean_l2,
        "mean_l2_ratio": adv_mean_l2 / ref_mean_l2 if ref_mean_l2 > 0 else float("inf"),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    enable_distributed()
    rank = get_global_rank()
    world_size = get_world_size()

    if rank == 0:
        LOGGER.info(
            "checkpoint=%s repr_model=%s split=%s num_images=%d world_size=%d",
            args.checkpoint,
            args.repr_model,
            args.split,
            args.num_images,
            world_size,
        )

    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    checkpoint = load_checkpoint_state(checkpoint_path)
    adv_state = select_adv_state(
        checkpoint,
        repr_model=args.repr_model,
        adv_state_name=args.adv_state_name,
    )

    if rank == 0:
        LOGGER.info(
            "resolved checkpoint=%s adv_state=%s feature_transform=%s",
            checkpoint_path,
            adv_state.get("name"),
            adv_state.get("feature_transform", "unknown"),
        )

    ref_model, feat_dim, ref_has_logits, native_size = load_repr_model(
        args.repr_model,
        device="cuda",
    )
    adv_model, _, adv_has_logits, _ = load_repr_model(
        args.repr_model,
        device="cuda",
    )
    if args.pool_type == "avg" and (ref_has_logits or adv_has_logits):
        raise ValueError(
            "pool_type='avg' is not valid for this repr model because the "
            "second output is logits rather than avg-pooled features"
        )
    try:
        adv_model.load_state_dict(adv_state["model"], strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"failed to load adversarial repr state {adv_state.get('name')!r} "
            f"from {checkpoint_path} into repr_model={args.repr_model!r}"
        ) from exc
    adv_model.eval()

    if rank == 0:
        LOGGER.info(
            "repr feat_dim=%s native_size=%s pool_type=%s",
            feat_dim,
            native_size,
            args.pool_type,
        )

    loader, total_images, limit, local_count = build_loader(
        args.data_root,
        args.split,
        args.img_size,
        args.num_images,
        args.batch_size,
        args.num_workers,
        rank,
        world_size,
    )
    if rank == 0:
        LOGGER.info(
            "dataset=%s/%s total=%d used=%d local=%d",
            args.data_root,
            args.split,
            total_images,
            limit,
            local_count,
        )

    result = accumulate_norms(
        ref_model,
        adv_model,
        loader,
        args.pool_type,
        rank,
        world_size,
    )

    if world_size > 1:
        torch.distributed.barrier()

    if rank == 0 and result is not None:
        result.update(
            {
                "checkpoint_path": str(checkpoint_path),
                "repr_model": args.repr_model,
                "repr_feat_dim": feat_dim,
                "repr_native_size": native_size,
                "adv_state_name": adv_state.get("name"),
                "feature_transform": adv_state.get("feature_transform", "unknown"),
                "feature_norm_cap": float(adv_state.get("feature_norm_cap", 0.0)),
                "residual_rms_kappa": float(adv_state.get("residual_rms_kappa", 0.0)),
                "residual_rms_tau": float(adv_state.get("residual_rms_tau", 0.0)),
                "norm_offset_weight": float(adv_state.get("norm_offset_weight", 0.0)),
                "norm_offset_split": bool(adv_state.get("norm_offset_split", False)),
                "pool_type": args.pool_type,
                "img_size": args.img_size,
                "split": args.split,
                "total_images_available": total_images,
                "images_used": limit,
                "images_per_rank": local_count,
            }
        )

        LOGGER.info(
            "adv_rms=%.6f ref_rms=%.6f rms_ratio=%.6f second_moment_ratio=%.6f",
            result["adv_rms"],
            result["ref_rms"],
            result["rms_ratio"],
            result["second_moment_ratio"],
        )
        LOGGER.info(
            "adv_mean_l2=%.6f ref_mean_l2=%.6f mean_l2_ratio=%.6f",
            result["adv_mean_l2"],
            result["ref_mean_l2"],
            result["mean_l2_ratio"],
        )

        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            LOGGER.info("wrote %s", args.output_json)

        print(json.dumps(result, sort_keys=True))

    if world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
