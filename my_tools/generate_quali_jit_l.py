#!/usr/bin/env python3
"""Generate matched JiT-L qualitative samples for a three-column paper figure.

The default run produces one sample for each ImageNet class (0..999) from:

1. the original JiT-L checkpoint (50 sampling steps),
2. JiT-L post-trained with FD-SIM (1 sampling step), and
3. JiT-L post-trained with adversarial FD-SIM (1 sampling step).

For a given class, all three models receive exactly the same initial noise.
The noise is derived only from ``noise_seed + class_id``, so pairing is stable
across checkpoints, batch sizes, GPU counts, and resumed runs.

The three aligned results are then concatenated horizontally, in the order
``original | FD Loss | Adv FD``, into 1000 paper-ready 768x256 PNGs under
``paper/quali/jit_l_threeway``. Per-model PNGs are retained as resumable
intermediate results.

Contiguous class subsets and multiple independent noises per class can be
requested with ``--class_start``, ``--class_count``, and
``--samples_per_class``. Pass ``--threeway_only`` to remove the per-model
intermediates after the composites have been validated.

Example (8 GPUs)::

    torchrun --standalone --nproc_per_node=8 paper/generate_quali_jit_l.py

The existing paper evaluations use the top-level ``model`` weights for all
three checkpoints, which is also the default here.  To reproduce the original
JiT repository's EMA sampling convention instead, pass
``--original_weight_key model_ema1``.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import models
from utils.data_util import to_uint8_numpy
from utils.distributed_util import (
    enable_distributed,
    get_global_rank,
    get_local_rank,
    get_world_size,
    is_main_process,
)
from utils.sampling_util import generate_images


LOGGER = logging.getLogger("jit_l_quali")

DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}

COMPOSITE_OUTPUT_NAME = "jit_l_threeway"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    output_name: str
    checkpoint: Path
    weight_key: str
    sampling_steps: int


def default_imagenet_train_dir() -> Path | None:
    candidates = []
    if os.environ.get("DATA_ROOT"):
        data_root = Path(os.environ["DATA_ROOT"]).expanduser()
        candidates.append(data_root if data_root.name == "train" else data_root / "train")
    candidates.append(Path("/mmu-vcg/zhangxu34/datasets/ImageNet-1K/train"))
    return next((path for path in candidates if path.is_dir()), None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate aligned JiT-L / FD Loss / Adv FD paper samples."
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=REPO_ROOT / "paper" / "quali",
    )
    parser.add_argument(
        "--original_checkpoint",
        type=Path,
        default=REPO_ROOT / "checkpoints/baseline/jit-l-16/checkpoint-last.pth",
    )
    parser.add_argument(
        "--fd_checkpoint",
        type=Path,
        default=REPO_ROOT / "checkpoints/post-trained/JiT-L_FD-SIM.pth",
    )
    parser.add_argument(
        "--adv_checkpoint",
        type=Path,
        default=(
            REPO_ROOT
            / "work_dirs/Jit-B-adv/JiT_L-fd-sim-advinc-w0.1-from-base"
            / "checkpoints/step_0124999.pth"
        ),
    )
    parser.add_argument(
        "--original_weight_key",
        choices=("model", "model_ema1", "model_ema2"),
        default="model",
        help=(
            "Use 'model' to match this repo's recorded baseline evaluation; "
            "use 'model_ema1' for the upstream JiT sampling convention."
        ),
    )
    parser.add_argument("--fd_weight_key", default="model")
    parser.add_argument("--adv_weight_key", default="model")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=("original", "fd_loss", "adv_fd"),
        default=("original", "fd_loss", "adv_fd"),
        help="Generate only a subset; useful for resuming or debugging.",
    )

    parser.add_argument(
        "--num_images",
        type=int,
        default=1000,
        help=(
            "Legacy one-sample-per-class selection count. Ignored when "
            "--class_count is supplied."
        ),
    )
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument(
        "--class_start",
        type=int,
        default=0,
        help="First class ID for contiguous class selection.",
    )
    parser.add_argument(
        "--class_count",
        type=int,
        default=None,
        help=(
            "Select this many consecutive classes starting at --class_start. "
            "When omitted, retain the legacy --num_images selection policy."
        ),
    )
    parser.add_argument(
        "--samples_per_class",
        type=int,
        default=1,
        help="Number of independently seeded samples generated for each class.",
    )
    parser.add_argument("--eval_bsz", type=int, default=16)
    parser.add_argument("--noise_seed", type=int, default=20260717)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--cfg", type=float, default=2.4)
    parser.add_argument("--interval_min", type=float, default=0.1)
    parser.add_argument("--interval_max", type=float, default=1.0)
    parser.add_argument(
        "--sampling_method", choices=("euler", "heun"), default="heun"
    )
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="bf16")
    parser.add_argument("--noise_scale", type=float, default=1.0)
    parser.add_argument("--original_steps", type=int, default=50)
    parser.add_argument("--fd_steps", type=int, default=1)
    parser.add_argument("--adv_steps", type=int, default=1)
    parser.add_argument(
        "--imagenet_train_dir",
        type=Path,
        default=default_imagenet_train_dir(),
        help=(
            "Optional ImageNet train directory. If supplied, sorted WNIDs are "
            "included in manifest.csv. It is not needed for generation."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate existing PNGs. By default completed samples are skipped.",
    )
    parser.add_argument(
        "--threeway_only",
        action="store_true",
        help=(
            "After all three-way composites are validated, remove the per-model "
            "PNG directories so only the composite PNGs are retained."
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate paths and print the resolved run configuration without CUDA.",
    )
    return parser.parse_args()


def build_specs(args: argparse.Namespace) -> list[ModelSpec]:
    return [
        ModelSpec(
            key="original",
            display_name="JiT-L (original)",
            output_name="jit_l_original",
            checkpoint=args.original_checkpoint.expanduser().resolve(),
            weight_key=args.original_weight_key,
            sampling_steps=args.original_steps,
        ),
        ModelSpec(
            key="fd_loss",
            display_name="JiT-L + FD Loss",
            output_name="jit_l_fd_loss",
            checkpoint=args.fd_checkpoint.expanduser().resolve(),
            weight_key=args.fd_weight_key,
            sampling_steps=args.fd_steps,
        ),
        ModelSpec(
            key="adv_fd",
            display_name="JiT-L + Adv FD",
            output_name="jit_l_adv_fd",
            checkpoint=args.adv_checkpoint.expanduser().resolve(),
            weight_key=args.adv_weight_key,
            sampling_steps=args.adv_steps,
        ),
    ]


def validate_args(args: argparse.Namespace, specs: list[ModelSpec]) -> None:
    if args.class_count is None:
        if not 1 <= args.num_images <= args.num_classes:
            raise ValueError(
                "num_images must be in [1, num_classes] so every sample has a "
                f"different class; got {args.num_images} and {args.num_classes}."
            )
        if args.class_start != 0 or args.samples_per_class != 1:
            raise ValueError(
                "--class_start and --samples_per_class require --class_count"
            )
    else:
        if args.class_start < 0:
            raise ValueError("class_start must be non-negative")
        if args.class_count <= 0:
            raise ValueError("class_count must be positive")
        if args.class_start + args.class_count > args.num_classes:
            raise ValueError(
                "selected class range exceeds num_classes: "
                f"[{args.class_start}, {args.class_start + args.class_count}) "
                f"versus {args.num_classes}"
            )
        if args.samples_per_class <= 0:
            raise ValueError("samples_per_class must be positive")
    if args.eval_bsz <= 0:
        raise ValueError("eval_bsz must be positive")
    if args.noise_seed < 0:
        raise ValueError("noise_seed must be non-negative")
    if not 0.0 <= args.interval_min <= args.interval_max <= 1.0:
        raise ValueError("expected 0 <= interval_min <= interval_max <= 1")
    for spec in specs:
        if spec.sampling_steps <= 0:
            raise ValueError(f"{spec.key} sampling steps must be positive")
        if not spec.checkpoint.is_file():
            raise FileNotFoundError(
                f"checkpoint for {spec.display_name} not found: {spec.checkpoint}"
            )


def build_samples(args: argparse.Namespace) -> list[tuple[int, int]]:
    """Return ``(sample_index, class_id)`` pairs in deterministic order."""
    if args.class_count is not None:
        class_ids = [
            class_id
            for class_id in range(
                args.class_start, args.class_start + args.class_count
            )
            for _ in range(args.samples_per_class)
        ]
    elif args.num_images == args.num_classes:
        class_ids = list(range(args.num_classes))
    else:
        class_ids = [
            (idx * args.num_classes) // args.num_images
            for idx in range(args.num_images)
        ]
    if args.class_count is None and len(class_ids) != len(set(class_ids)):
        raise RuntimeError("internal error: class selection contains duplicates")
    return list(enumerate(class_ids))


def sample_in_class(args: argparse.Namespace, sample_index: int) -> int:
    if args.class_count is None:
        return 0
    return sample_index % args.samples_per_class


def sample_noise_seed(
    args: argparse.Namespace,
    sample_index: int,
    class_id: int,
) -> int:
    if args.class_count is None:
        # Preserve the original one-sample-per-class behavior exactly.
        return args.noise_seed + class_id
    return (
        args.noise_seed
        + class_id * args.samples_per_class
        + sample_in_class(args, sample_index)
    )


def load_wnids(train_dir: Path | None, num_classes: int) -> list[str]:
    if train_dir is None:
        return [""] * num_classes
    train_dir = train_dir.expanduser().resolve()
    wnids = sorted(path.name for path in train_dir.iterdir() if path.is_dir())
    if len(wnids) != num_classes:
        raise ValueError(
            f"expected {num_classes} class directories in {train_dir}, "
            f"found {len(wnids)}"
        )
    return wnids


def image_name(sample_index: int, class_id: int) -> str:
    return f"{sample_index:06d}_class{class_id:04d}.png"


def make_fixed_noise(
    noise_seeds: list[int],
    img_size: int,
    noise_scale: float,
    device: torch.device,
) -> torch.Tensor:
    noises = []
    for noise_seed in noise_seeds:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(noise_seed)
        noises.append(
            torch.randn(
                3,
                img_size,
                img_size,
                generator=generator,
                dtype=torch.float32,
            )
        )
    return torch.stack(noises).to(device, non_blocking=True) * noise_scale


def atomic_json_dump(data: Any, path: Path) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def checkpoint_record(spec: ModelSpec) -> dict[str, Any]:
    stat = spec.checkpoint.stat()
    return {
        "key": spec.key,
        "display_name": spec.display_name,
        "output_dir": spec.output_name,
        "checkpoint": str(spec.checkpoint),
        "checkpoint_bytes": stat.st_size,
        "checkpoint_mtime_ns": stat.st_mtime_ns,
        "weight_key": spec.weight_key,
        "sampling_steps": spec.sampling_steps,
    }


def build_generation_identity(
    args: argparse.Namespace,
    specs: list[ModelSpec],
    samples: list[tuple[int, int]],
) -> dict[str, Any]:
    class_ids = [class_id for _, class_id in samples]
    class_id_digest = hashlib.sha256(
        ",".join(str(class_id) for class_id in class_ids).encode("utf-8")
    ).hexdigest()
    identity = {
        "schema_version": 1,
        "num_images_per_model": len(samples),
        "source_images_expected": len(samples) * len(specs),
        "paper_images_expected": len(samples),
        "total_expected_pngs": len(samples) * (len(specs) + 1),
        "class_policy": "unique class IDs; full run uses ImageNet classes 0..999",
        "class_ids_sha256": class_id_digest,
        "noise_policy": "torch_cpu_randn_v1: Generator seed = noise_seed + class_id",
        "noise_seed": args.noise_seed,
        "model": "JiT_L",
        "img_size": args.img_size,
        "num_classes": args.num_classes,
        "cfg": args.cfg,
        "cfg_interval": [args.interval_min, args.interval_max],
        "sampling_method": args.sampling_method,
        "dtype": args.dtype,
        "noise_scale": args.noise_scale,
        "rope_2d": True,
        "learned_pe": True,
        "legacy_time_convention": True,
        "models": [checkpoint_record(spec) for spec in specs],
        "paper_output": {
            "type": "horizontal_concat",
            "output_dir": COMPOSITE_OUTPUT_NAME,
            "order": [spec.key for spec in specs],
            "gap_pixels": 0,
            "size": [args.img_size * len(specs), args.img_size],
        },
    }
    if args.class_count is not None:
        noise_seeds = [
            sample_noise_seed(args, sample_index, class_id)
            for sample_index, class_id in samples
        ]
        identity.update(
            {
                "schema_version": 2,
                "class_policy": "contiguous class range with repeated samples",
                "class_start": args.class_start,
                "class_count": args.class_count,
                "samples_per_class": args.samples_per_class,
                "noise_policy": (
                    "torch_cpu_randn_v1: Generator seed = noise_seed + "
                    "class_id * samples_per_class + sample_in_class"
                ),
                "sample_noise_seeds_sha256": hashlib.sha256(
                    ",".join(str(seed) for seed in noise_seeds).encode("utf-8")
                ).hexdigest(),
            }
        )
    if args.threeway_only:
        identity["retention_policy"] = "retain_threeway_composites_only"
        identity["paper_output"]["retained_only"] = True
    return identity


def generation_fingerprint(identity: dict[str, Any]) -> str:
    payload = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def iter_output_pngs(output_root: Path, specs: list[ModelSpec]):
    output_dirs = [output_root / spec.output_name for spec in specs]
    output_dirs.append(output_root / COMPOSITE_OUTPUT_NAME)
    for output_dir in output_dirs:
        if output_dir.is_dir():
            yield from (
                path
                for path in output_dir.glob("*.png")
                if not path.name.startswith(".")
            )


def write_run_metadata(
    args: argparse.Namespace,
    all_specs: list[ModelSpec],
    samples: list[tuple[int, int]],
    wnids: list[str],
) -> str:
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    for spec in all_specs:
        output_dir = output_root / spec.output_name
        output_dir.mkdir(parents=True, exist_ok=True)
        # Crash leftovers are never final samples and must not affect resume.
        for path in output_dir.glob(".*.tmp.png"):
            path.unlink()
    composite_dir = output_root / COMPOSITE_OUTPUT_NAME
    composite_dir.mkdir(parents=True, exist_ok=True)
    for path in composite_dir.glob(".*.tmp.png"):
        path.unlink()

    identity = build_generation_identity(args, all_specs, samples)
    fingerprint = generation_fingerprint(identity)
    config_path = output_root / "run_config.json"
    existing_config = None
    if config_path.is_file():
        try:
            existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"cannot read existing run config: {config_path}") from error

    existing_pngs = list(iter_output_pngs(output_root, all_specs))
    compatible = bool(
        existing_config
        and existing_config.get("generation_fingerprint") == fingerprint
        and existing_config.get("generation_identity") == identity
    )
    if existing_pngs and not compatible:
        if not args.overwrite:
            raise RuntimeError(
                f"{output_root} already contains generated PNGs from a different "
                "configuration. Use a new --output_root, or pass --overwrite to "
                "replace all three model outputs."
            )
        if set(args.only) != {"original", "fd_loss", "adv_fd"}:
            raise RuntimeError(
                "A configuration-changing --overwrite must generate all three "
                "models; remove --only or use a new --output_root."
            )
        compatible = False

    if args.overwrite:
        overwrite_specs = all_specs if not compatible else [
            spec for spec in all_specs if spec.key in set(args.only)
        ]
        overwrite_dirs = [output_root / spec.output_name for spec in overwrite_specs]
        overwrite_dirs.append(composite_dir)
        for output_dir in overwrite_dirs:
            for path in output_dir.glob("*.png"):
                if path.name.startswith("."):
                    continue
                path.unlink()

    now = datetime.now(timezone.utc).isoformat()
    config = {
        "generation_fingerprint": fingerprint,
        "generation_identity": identity,
        "created_at_utc": (
            existing_config.get("created_at_utc", now)
            if compatible and existing_config
            else now
        ),
        "last_invoked_at_utc": now,
        "last_requested_models": list(args.only),
        "output_root": str(output_root),
        "class_map": {
            "source": (
                str(args.imagenet_train_dir.expanduser().resolve())
                if args.imagenet_train_dir is not None
                else None
            ),
            "wnids_sha256": hashlib.sha256(
                "\n".join(wnids).encode("utf-8")
            ).hexdigest(),
        },
        "weights_note": (
            "Defaults match this repo's online evaluations. For the upstream "
            "JiT convention use --original_weight_key model_ema1."
        ),
    }

    if args.threeway_only:
        all_output_names = {"jit_l_threeway": COMPOSITE_OUTPUT_NAME}
    else:
        all_output_names = {
            "jit_l_original": "jit_l_original",
            "jit_l_fd_loss": "jit_l_fd_loss",
            "jit_l_adv_fd": "jit_l_adv_fd",
            "jit_l_threeway": COMPOSITE_OUTPUT_NAME,
        }
    manifest_path = output_root / "manifest.csv"
    tmp_manifest = output_root / ".manifest.csv.tmp"
    with tmp_manifest.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "sample_index",
            "class_id",
        ]
        if args.class_count is not None:
            fieldnames.append("sample_in_class")
        fieldnames.extend(
            [
                "wnid",
                "noise_seed",
                "filename",
                *all_output_names,
            ]
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample_index, class_id in samples:
            filename = image_name(sample_index, class_id)
            row = {
                "sample_index": sample_index,
                "class_id": class_id,
                "wnid": wnids[class_id],
                "noise_seed": sample_noise_seed(args, sample_index, class_id),
                "filename": filename,
                **{
                    key: str(Path(directory) / filename)
                    for key, directory in all_output_names.items()
                },
            }
            if args.class_count is not None:
                row["sample_in_class"] = sample_in_class(args, sample_index)
            writer.writerow(row)
    os.replace(tmp_manifest, manifest_path)

    atomic_json_dump(config, output_root / "run_config.json")

    for spec in all_specs:
        atomic_json_dump(
            {
                "generation_fingerprint": fingerprint,
                "shared_generation_config": {
                    key: value
                    for key, value in identity.items()
                    if key != "models"
                },
                "model": checkpoint_record(spec),
            },
            output_root / spec.output_name / "generation_config.json",
        )
    atomic_json_dump(
        {
            "generation_fingerprint": fingerprint,
            **identity["paper_output"],
        },
        composite_dir / "generation_config.json",
    )
    return fingerprint


def build_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    model = models.JiTDenoiser_models["JiT_L"](
        img_size=args.img_size,
        num_classes=args.num_classes,
        label_drop_prob=0.1,
        attn_dropout=0.0,
        proj_dropout=0.0,
        P_mean=0.8,
        P_std=0.8,
        t_eps=5e-2,
        noise_scale=args.noise_scale,
        rope_2d=True,
        learned_pe=True,
        legacy_time_convention=True,
        grad_checkpointing=False,
    )
    return model.to(device).eval().requires_grad_(False)


def load_checkpoint_weights(model: torch.nn.Module, spec: ModelSpec) -> None:
    LOGGER.info(
        "loading %s: %s[%s]",
        spec.display_name,
        spec.checkpoint,
        spec.weight_key,
    )
    checkpoint = torch.load(
        spec.checkpoint,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise TypeError(f"expected a dict checkpoint, got {type(checkpoint)!r}")
    if spec.weight_key in checkpoint:
        state_dict = checkpoint[spec.weight_key]
    elif spec.weight_key == "model" and all(
        isinstance(value, torch.Tensor) for value in checkpoint.values()
    ):
        state_dict = checkpoint
    else:
        available = ", ".join(str(key) for key in checkpoint.keys())
        raise KeyError(
            f"weight key {spec.weight_key!r} is absent from {spec.checkpoint}; "
            f"available top-level keys: {available}"
        )
    model.load_state_dict(state_dict, strict=True)
    torch.cuda.synchronize()
    del state_dict
    del checkpoint
    gc.collect()


def save_png_atomic(image, path: Path, rank: int) -> None:
    tmp = path.with_name(f".{path.stem}.rank{rank}.tmp.png")
    Image.fromarray(image).save(tmp, format="PNG")
    os.replace(tmp, path)


def save_pil_atomic(image: Image.Image, path: Path, rank: int) -> None:
    tmp = path.with_name(f".{path.stem}.rank{rank}.tmp.png")
    image.save(tmp, format="PNG")
    os.replace(tmp, path)


@torch.inference_mode()
def generate_one_model(
    args: argparse.Namespace,
    model: torch.nn.Module,
    spec: ModelSpec,
    pending_samples: list[tuple[int, int]],
    device: torch.device,
) -> None:
    rank = get_global_rank()
    world_size = get_world_size()
    output_dir = args.output_root / spec.output_name
    # Split the global missing list, rather than filtering after a fixed rank
    # split. This keeps resume workloads balanced even if missing IDs cluster.
    local_samples = pending_samples[rank::world_size]

    if is_main_process():
        LOGGER.info(
            "%s: generating %d missing image(s), steps=%d, cfg=%.3g",
            spec.display_name,
            len(pending_samples),
            spec.sampling_steps,
            args.cfg,
        )

    args.num_sampling_steps = spec.sampling_steps
    completed = 0
    for start in range(0, len(local_samples), args.eval_bsz):
        batch = local_samples[start : start + args.eval_bsz]
        sample_indices = [sample_index for sample_index, _ in batch]
        class_ids = [class_id for _, class_id in batch]
        noise_seeds = [
            sample_noise_seed(args, sample_index, class_id)
            for sample_index, class_id in batch
        ]
        labels = torch.tensor(class_ids, dtype=torch.long, device=device)
        z_t = make_fixed_noise(
            noise_seeds=noise_seeds,
            img_size=args.img_size,
            noise_scale=args.noise_scale,
            device=device,
        )
        images = generate_images(
            args,
            model,
            labels=labels,
            cfg=args.cfg,
            tokenizer=None,
            z_t=z_t,
        )
        images_uint8 = to_uint8_numpy(images)
        for image, sample_index, class_id in zip(
            images_uint8, sample_indices, class_ids
        ):
            save_png_atomic(
                image,
                output_dir / image_name(sample_index, class_id),
                rank,
            )
        completed += len(batch)
        LOGGER.info(
            "rank %d %s: %d/%d local images",
            rank,
            spec.key,
            completed,
            len(local_samples),
        )


def get_pending_samples(
    args: argparse.Namespace,
    spec: ModelSpec,
    samples: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    if args.overwrite:
        return list(samples)
    output_dir = args.output_root / spec.output_name
    return [
        sample
        for sample in samples
        if not (output_dir / image_name(*sample)).is_file()
    ]


def validate_output_files(
    output_dir: Path,
    samples: list[tuple[int, int]],
) -> tuple[int, str | None]:
    expected = {image_name(*sample) for sample in samples}
    actual = {
        path.name
        for path in output_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".png"
        and not path.name.startswith(".")
    }
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if not missing and not extra:
        return len(actual), None
    details = []
    if missing:
        details.append(f"missing={len(missing)} (first: {missing[:3]})")
    if extra:
        details.append(f"extra={len(extra)} (first: {extra[:3]})")
    return len(actual), "; ".join(details)


def finalize_output_directory(
    output_dir: Path,
    display_name: str,
    output_key: str,
    samples: list[tuple[int, int]],
    device: torch.device,
    fingerprint: str,
) -> int:
    torch.distributed.barrier()
    actual, error = validate_output_files(output_dir, samples)
    error_flag = torch.tensor(
        1 if error is not None else 0, dtype=torch.int64, device=device
    )
    torch.distributed.all_reduce(error_flag, op=torch.distributed.ReduceOp.MAX)
    if error_flag.item():
        if error is None:
            error = "another rank observed an inconsistent output directory"
        raise RuntimeError(f"{display_name} output validation failed: {error}")

    if is_main_process():
        atomic_json_dump(
            {
                "generation_fingerprint": fingerprint,
                "output": output_key,
                "images": actual,
            },
            output_dir / "_SUCCESS",
        )
        LOGGER.info("%s complete: %d PNGs", display_name, actual)
    torch.distributed.barrier()
    return actual


def get_pending_composites(
    args: argparse.Namespace,
    all_specs: list[ModelSpec],
    samples: list[tuple[int, int]],
    dirty_samples: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    output_dir = args.output_root / COMPOSITE_OUTPUT_NAME
    pending = []
    for sample in samples:
        filename = image_name(*sample)
        composite_path = output_dir / filename
        if sample in dirty_samples or not composite_path.is_file():
            pending.append(sample)
            continue
        composite_mtime_ns = composite_path.stat().st_mtime_ns
        if any(
            (args.output_root / spec.output_name / filename).stat().st_mtime_ns
            > composite_mtime_ns
            for spec in all_specs
        ):
            pending.append(sample)
    return pending


def generate_composites(
    args: argparse.Namespace,
    all_specs: list[ModelSpec],
    pending_samples: list[tuple[int, int]],
) -> None:
    rank = get_global_rank()
    world_size = get_world_size()
    output_dir = args.output_root / COMPOSITE_OUTPUT_NAME
    local_samples = pending_samples[rank::world_size]
    if is_main_process():
        LOGGER.info(
            "building %d three-way composite image(s): %s",
            len(pending_samples),
            " | ".join(spec.display_name for spec in all_specs),
        )

    for completed, (sample_index, class_id) in enumerate(local_samples, start=1):
        filename = image_name(sample_index, class_id)
        panels = []
        for spec in all_specs:
            source_path = args.output_root / spec.output_name / filename
            with Image.open(source_path) as image:
                panel = image.convert("RGB").copy()
            expected_size = (args.img_size, args.img_size)
            if panel.size != expected_size:
                raise RuntimeError(
                    f"unexpected source size for {source_path}: "
                    f"{panel.size}, expected {expected_size}"
                )
            panels.append(panel)

        composite = Image.new(
            "RGB",
            (args.img_size * len(panels), args.img_size),
        )
        x_offset = 0
        for panel in panels:
            composite.paste(panel, (x_offset, 0))
            x_offset += panel.width
        save_pil_atomic(composite, output_dir / filename, rank)
        LOGGER.info(
            "rank %d threeway: %d/%d local images",
            rank,
            completed,
            len(local_samples),
        )


def remove_model_output_directories(
    output_root: Path,
    all_specs: list[ModelSpec],
) -> None:
    """Remove validated per-model intermediates for ``--threeway_only`` runs."""
    resolved_root = output_root.resolve()
    for spec in all_specs:
        output_dir = (resolved_root / spec.output_name).resolve()
        if output_dir.parent != resolved_root:
            raise RuntimeError(f"unsafe model output path: {output_dir}")
        if output_dir.is_dir():
            shutil.rmtree(output_dir)
            LOGGER.info("removed intermediate model output: %s", output_dir)


def resolved_config_for_print(
    args: argparse.Namespace,
    all_specs: list[ModelSpec],
    samples: list[tuple[int, int]],
) -> dict[str, Any]:
    config = {
        "output_root": str(args.output_root),
        "num_images_per_model": len(samples),
        "unique_classes": len({class_id for _, class_id in samples}),
        "noise_seed": args.noise_seed,
        "cfg": args.cfg,
        "interval": [args.interval_min, args.interval_max],
        "dtype": args.dtype,
        "requested_models": list(args.only),
        "retention": (
            "threeway_only" if args.threeway_only else "threeway_and_per_model"
        ),
        "paper_output": {
            "directory": str(args.output_root / COMPOSITE_OUTPUT_NAME),
            "images": len(samples),
            "size": [args.img_size * len(all_specs), args.img_size],
            "order": [spec.key for spec in all_specs],
        },
        "models": [
            {
                "name": spec.display_name,
                "checkpoint": str(spec.checkpoint),
                "weight_key": spec.weight_key,
                "sampling_steps": spec.sampling_steps,
                "output_dir": str(args.output_root / spec.output_name),
            }
            for spec in all_specs
        ],
    }
    if args.class_count is not None:
        config["class_selection"] = {
            "start": args.class_start,
            "count": args.class_count,
            "samples_per_class": args.samples_per_class,
            "end_exclusive": args.class_start + args.class_count,
        }
        config["noise_seed_range"] = [
            min(
                sample_noise_seed(args, sample_index, class_id)
                for sample_index, class_id in samples
            ),
            max(
                sample_noise_seed(args, sample_index, class_id)
                for sample_index, class_id in samples
            ),
        ]
    return config


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.expanduser().resolve()
    all_specs = build_specs(args)
    requested = set(args.only)
    specs = [spec for spec in all_specs if spec.key in requested]
    validate_args(args, all_specs)
    samples = build_samples(args)
    wnids = load_wnids(args.imagenet_train_dir, args.num_classes)
    identity = build_generation_identity(args, all_specs, samples)
    fingerprint = generation_fingerprint(identity)

    if args.dry_run:
        print(json.dumps(resolved_config_for_print(args, all_specs, samples), indent=2))
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Run this script in the fdloss environment "
            "on a GPU compute node."
        )

    enable_distributed()
    rank = get_global_rank()
    device = torch.device("cuda", get_local_rank())
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s %(levelname)s [rank {rank}] %(message)s",
    )
    torch.manual_seed(args.seed + rank)
    args.enable_amp = args.dtype != "fp32"
    args.amp_dtype = DTYPES[args.dtype]
    args.same_noise = False

    try:
        metadata_error = None
        if is_main_process():
            try:
                written_fingerprint = write_run_metadata(
                    args, all_specs, samples, wnids
                )
                if written_fingerprint != fingerprint:
                    raise RuntimeError("internal error: run fingerprint changed")
                (args.output_root / "_SUCCESS").unlink(missing_ok=True)
                (
                    args.output_root / COMPOSITE_OUTPUT_NAME / "_SUCCESS"
                ).unlink(missing_ok=True)
                for spec in specs:
                    (args.output_root / spec.output_name / "_SUCCESS").unlink(
                        missing_ok=True
                    )
                LOGGER.info(
                    "output=%s, models=%d, images/model=%d, unique classes=%d",
                    args.output_root,
                    len(specs),
                    len(samples),
                    len({class_id for _, class_id in samples}),
                )
            except Exception as error:
                metadata_error = f"{type(error).__name__}: {error}"
                LOGGER.error("metadata preparation failed: %s", metadata_error)

        metadata_failed = torch.tensor(
            1 if metadata_error is not None else 0,
            dtype=torch.int64,
            device=device,
        )
        torch.distributed.broadcast(metadata_failed, src=0)
        if metadata_failed.item():
            if metadata_error is None:
                metadata_error = "rank 0 failed while preparing run metadata"
            raise RuntimeError(metadata_error)
        torch.distributed.barrier()

        model = None
        counts = {}
        dirty_samples: set[tuple[int, int]] = set()
        for spec in specs:
            pending_samples = get_pending_samples(args, spec, samples)
            if not pending_samples:
                if is_main_process():
                    LOGGER.info("%s already has every requested PNG", spec.display_name)
            else:
                dirty_samples.update(pending_samples)
                if model is None:
                    model = build_model(args, device)
                    LOGGER.info(
                        "created JiT-L with %d parameters",
                        sum(parameter.numel() for parameter in model.parameters()),
                    )
                load_checkpoint_weights(model, spec)
                generate_one_model(args, model, spec, pending_samples, device)
                torch.cuda.empty_cache()
            counts[spec.key] = finalize_output_directory(
                args.output_root / spec.output_name,
                spec.display_name,
                spec.key,
                samples,
                device,
                fingerprint,
            )

        torch.distributed.barrier()
        source_errors = [
            validate_output_files(args.output_root / spec.output_name, samples)[1]
            for spec in all_specs
        ]
        sources_ready = torch.tensor(
            0 if any(error is not None for error in source_errors) else 1,
            dtype=torch.int64,
            device=device,
        )
        torch.distributed.all_reduce(
            sources_ready, op=torch.distributed.ReduceOp.MIN
        )
        if sources_ready.item():
            pending_composites = get_pending_composites(
                args, all_specs, samples, dirty_samples
            )
            if pending_composites:
                generate_composites(args, all_specs, pending_composites)
            elif is_main_process():
                LOGGER.info("all three-way composite PNGs already exist")
            counts[COMPOSITE_OUTPUT_NAME] = finalize_output_directory(
                args.output_root / COMPOSITE_OUTPUT_NAME,
                "JiT-L three-way composites",
                COMPOSITE_OUTPUT_NAME,
                samples,
                device,
                fingerprint,
            )
        elif is_main_process():
            LOGGER.info(
                "three-way concatenation deferred until all three source "
                "model directories are complete"
            )

        torch.distributed.barrier()
        if is_main_process():
            LOGGER.info("finished: %s", counts)
            all_counts = {}
            all_complete = True
            for spec in all_specs:
                count, error = validate_output_files(
                    args.output_root / spec.output_name, samples
                )
                all_counts[spec.key] = count
                all_complete = all_complete and error is None
            paper_count, paper_error = validate_output_files(
                args.output_root / COMPOSITE_OUTPUT_NAME, samples
            )
            all_complete = all_complete and paper_error is None
            if all_complete:
                if args.threeway_only:
                    remove_model_output_directories(args.output_root, all_specs)
                atomic_json_dump(
                    {
                        "generation_fingerprint": fingerprint,
                        "source_images_per_model": (
                            "generated_then_removed"
                            if args.threeway_only
                            else all_counts
                        ),
                        "paper_images": paper_count,
                        "paper_output": COMPOSITE_OUTPUT_NAME,
                        "retained_outputs": (
                            [COMPOSITE_OUTPUT_NAME]
                            if args.threeway_only
                            else [
                                *(spec.output_name for spec in all_specs),
                                COMPOSITE_OUTPUT_NAME,
                            ]
                        ),
                    },
                    args.output_root / "_SUCCESS",
                )
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
