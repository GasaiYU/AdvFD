#!/usr/bin/env python3
"""Measure when a direct-x sCM stops depending on its input.

This diagnostic evaluates a trained ``SCMJiTDenoiser`` at fixed physical
TrigFlow times.  At every time it independently changes the clean image,
noise, complete model input, and class label.  The resulting metrics separate
three different failure modes:

* an input-independent scalar/image;
* an input-independent 16x16 JiT decoder template;
* a model that remains input-dependent but is simply inaccurate.

Example:

    python meanflow/diagnose_scm_xpred_t_sweep.py \
        --checkpoint work_dirs/meanflow/EXPERIMENT/checkpoints/step_0019999.pth

The script uses ``model_ema`` by default and writes JSON, CSV, and two image
panels below ``work_dirs/diagnostics/scm_xpred_t_sweep``.
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import typing_extensions


# PyTorch 2.4 imports this metadata-only decorator, while the project image
# may contain an older typing_extensions package.
if not hasattr(typing_extensions, "deprecated"):
    def _deprecated(*args, **kwargs):
        del args, kwargs

        def decorator(obj):
            return obj

        return decorator

    typing_extensions.deprecated = _deprecated

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from meanflow.train_scm_jit_b import SCMJiTDenoiser  # noqa: E402
from utils.data_util import center_crop_arr  # noqa: E402


DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "work_dirs/meanflow/jit_b_scm_xpred_reversed_time_boundary_band"
    / "checkpoints/step_0019999.pth"
)
DEFAULT_DATA_PATH = Path(
    os.environ.get(
        "DATA_ROOT",
        "/mmu-vcg/zhangxu34/datasets/ImageNet-1K/",
    )
)
DEFAULT_TIMES = (
    "0,0.001,0.005,0.01,0.015,0.02,0.025,0.03,0.05,0.1,0.2,"
    "0.4,0.8,1.2,1.3,1.4,1.5,1.55,tmax"
)


def _checkpoint_args(checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
    value = checkpoint.get("args", {})
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    kwargs = {
        "map_location": "cpu",
        "weights_only": False,
    }
    try:
        return torch.load(path, mmap=True, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def _select_state_dict(
    checkpoint: Mapping[str, Any], key: str
) -> Mapping[str, torch.Tensor]:
    if key == "auto":
        for candidate in ("model_ema", "model", "training_model"):
            if candidate in checkpoint:
                key = candidate
                break
        else:
            raise KeyError(
                "no supported model state found; available checkpoint keys: "
                f"{list(checkpoint)[:20]}"
            )
    if key not in checkpoint:
        raise KeyError(
            f"checkpoint key {key!r} not found; available: "
            f"{list(checkpoint)[:20]}"
        )
    state_dict = checkpoint[key]
    if not isinstance(state_dict, Mapping):
        raise TypeError(
            f"checkpoint[{key!r}] must be a state dict, got "
            f"{type(state_dict).__name__}"
        )
    return state_dict


def _build_model(
    config: Mapping[str, Any], batch_size: int
) -> SCMJiTDenoiser:
    return SCMJiTDenoiser(
        img_size=int(config.get("img_size", 256)),
        model_size="base",
        num_classes=int(config.get("num_classes", 1000)),
        label_drop_prob=float(config.get("label_drop_prob", 0.1)),
        attn_dropout=float(config.get("attn_dropout", 0.45)),
        proj_dropout=float(config.get("proj_dropout", 0.45)),
        P_mean=float(config.get("P_mean", -1.0)),
        P_std=float(config.get("P_std", 1.6)),
        t_eps=float(config.get("t_eps", 0.05)),
        noise_scale=float(config.get("noise_scale", 1.0)),
        legacy_time_convention=False,
        rope_2d=bool(config.get("rope_2d", True)),
        learned_pe=bool(config.get("learned_pe", True)),
        grad_checkpointing=False,
        adaptive_double_norm=True,
        dropout_all_blocks=True,
        objective="scm",
        tangent_norm_c=float(config.get("tangent_norm_c", 0.1)),
        tangent_warmup_steps=int(config.get("tangent_warmup_steps", 10000)),
        adaptive_weighting=False,
        sigma_data=float(config.get("sigma_data", 0.5)),
        sigma_max=float(config.get("sigma_max", 80.0)),
        jvp_dtype=str(config.get("jvp_dtype", "amp")),
        x_adapt_steps=0,
        x_loss_sin_min=float(config.get("x_loss_sin_min", 1e-3)),
        boundary_loss_weight=float(config.get("boundary_loss_weight", 1.0)),
        boundary_band_max=float(config.get("boundary_band_max", 0.02)),
        deterministic_boundary=True,
        collapse_monitor_samples=batch_size,
        network_time_mode=str(config.get("network_time_mode", "legacy_reversed")),
    )


def _resolve_train_dir(path: Path) -> Path:
    train = path / "train"
    return train if train.is_dir() else path


def _load_real_batch(
    data_path: Path,
    image_size: int,
    batch_size: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms

    train_dir = _resolve_train_dir(data_path)
    if not train_dir.is_dir():
        raise FileNotFoundError(f"ImageNet directory not found: {train_dir}")
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda image: center_crop_arr(image, image_size)),
            transforms.ToTensor(),
        ]
    )
    dataset = datasets.ImageFolder(str(train_dir), transform=transform)
    if batch_size > len(dataset):
        raise ValueError(
            f"batch_size={batch_size} exceeds dataset size {len(dataset)}"
        )
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:batch_size]
    samples = [dataset[int(index)] for index in indices]
    images = torch.stack([sample[0] for sample in samples]).mul(2.0).sub(1.0)
    labels = torch.tensor([sample[1] for sample in samples], dtype=torch.long)
    return images, labels, [int(index) for index in indices]


def _parse_times(spec: str, sigma_data: float, sigma_max: float) -> List[float]:
    t_max = math.atan(sigma_max / sigma_data)
    values = []
    for raw in spec.split(","):
        token = raw.strip().lower()
        if not token:
            continue
        value = t_max if token in {"tmax", "max"} else float(token)
        if not 0.0 <= value <= t_max:
            raise ValueError(
                f"time {value} is outside [0, tmax={t_max:.9f}]"
            )
        values.append(value)
    if not values:
        raise ValueError("--times must contain at least one value")
    # Preserve user order while avoiding duplicate endpoint aliases.
    deduplicated = []
    for value in values:
        if not any(abs(value - previous) < 1e-12 for previous in deduplicated):
            deduplicated.append(value)
    return deduplicated


def _rms(value: torch.Tensor) -> torch.Tensor:
    return value.float().square().mean().sqrt()


def _batch_std_rms(value: torch.Tensor) -> torch.Tensor:
    return value.float().var(dim=0, unbiased=False).mean().sqrt()


def _spatial_std_rms(value: torch.Tensor) -> torch.Tensor:
    value = value.float()
    centered = value - value.mean(dim=(1, 2, 3), keepdim=True)
    return _rms(centered)


def _mean_off_diagonal_correlation(value: torch.Tensor) -> torch.Tensor:
    value = value.float()
    centered = value - value.mean(dim=(1, 2, 3), keepdim=True)
    flat = centered.flatten(1)
    flat = flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
    count = flat.size(0)
    if count < 2:
        return flat.new_zeros(())
    gram = flat @ flat.t()
    return (gram.sum() - gram.diagonal().sum()) / (count * (count - 1))


def _cross_sample_template_fraction(value: torch.Tensor) -> torch.Tensor:
    value = value.float()
    sample_centered = value - value.mean(dim=(1, 2, 3), keepdim=True)
    common_template = sample_centered.mean(dim=0, keepdim=True)
    residual = sample_centered - common_template
    raw_fraction = (
        1.0
        - residual.square().mean()
        / sample_centered.square().mean().clamp_min(1e-12)
    )
    # Even independent zero-mean samples assign 1 / batch_size of their
    # energy to the finite-sample mean.  Remove that bias so independent
    # outputs report approximately zero and identical templates report one.
    chance_fraction = 1.0 / value.size(0)
    return (raw_fraction - chance_fraction) / (1.0 - chance_fraction)


def _as_float(value: torch.Tensor) -> float:
    return float(value.detach().float().cpu())


def _response_gain(
    output_delta: torch.Tensor,
    input_delta: torch.Tensor,
) -> torch.Tensor:
    # At t=0, changing path noise changes neither x_t nor the exact output.
    # Batched GEMM roundoff can nevertheless leave a tiny output difference;
    # dividing it by a zero input difference would produce a meaningless
    # enormous ratio.
    if float(input_delta.detach().float().cpu()) < 1e-7:
        return output_delta.new_zeros(())
    return output_delta / input_delta


def _contrast_normalize(value: torch.Tensor) -> torch.Tensor:
    value = value.float()
    mean = value.mean(dim=(1, 2, 3), keepdim=True)
    std = value.std(dim=(1, 2, 3), unbiased=False, keepdim=True)
    return ((value - mean) / (6.0 * std.clamp_min(1e-6)) + 0.5).clamp(0, 1)


def _autocast_context(device: torch.device, dtype: str):
    if dtype == "fp32":
        return torch.autocast(device_type=device.type, enabled=False)
    amp_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[dtype]
    return torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=True)


@torch.inference_mode()
def _evaluate_time(
    model: SCMJiTDenoiser,
    x0: torch.Tensor,
    matched_labels: torch.Tensor,
    fixed_labels: torch.Tensor,
    alternate_labels: torch.Tensor,
    noise_a: torch.Tensor,
    noise_b: torch.Tensor,
    physical_t: float,
    amp_dtype: str,
) -> Tuple[Dict[str, float], torch.Tensor, torch.Tensor]:
    batch_size = x0.size(0)
    device = x0.device
    t = x0.new_full((batch_size, 1, 1, 1), physical_t)
    cos_t = torch.cos(t)
    sin_t = torch.sin(t)
    x0_other = torch.roll(x0, shifts=1, dims=0)

    x_t = cos_t * x0 + sin_t * noise_a
    x_t_independent = cos_t * x0_other + sin_t * noise_b
    x_t_other_noise = cos_t * x0 + sin_t * noise_b
    x_t_other_clean = cos_t * x0_other + sin_t * noise_a

    # One concatenated forward is materially faster than six small forwards,
    # especially for CPU fallback diagnostics.  Chunk order matches the
    # controlled interventions documented above.
    combined_input = torch.cat(
        [
            x_t,
            x_t_independent,
            x_t_other_noise,
            x_t_other_clean,
            x_t,
            x_t,
        ],
        dim=0,
    )
    combined_labels = torch.cat(
        [
            fixed_labels,
            fixed_labels,
            fixed_labels,
            fixed_labels,
            alternate_labels,
            matched_labels,
        ],
        dim=0,
    )
    combined_t = t.repeat(6, 1, 1, 1)
    with _autocast_context(device, amp_dtype):
        combined_output = model._scm_consistency_output(
            combined_input,
            combined_t,
            combined_labels,
            cfg=1.0,
        ).float()
    (
        fixed_output,
        independent_output,
        other_noise_output,
        other_clean_output,
        alternate_label_output,
        matched_output,
    ) = combined_output.chunk(6, dim=0)

    input_delta = _rms(x_t - x_t_independent)
    noise_input_delta = _rms(x_t - x_t_other_noise)
    clean_input_delta = _rms(x_t - x_t_other_clean)
    input_output_delta = _rms(fixed_output - independent_output)
    noise_output_delta = _rms(fixed_output - other_noise_output)
    clean_output_delta = _rms(fixed_output - other_clean_output)
    output_batch_std = _batch_std_rms(fixed_output)
    output_spatial_std = _spatial_std_rms(fixed_output)

    collapse = model._collapse_metrics(fixed_output)
    metrics = {
        "t": float(physical_t),
        "normalized_t": float(physical_t / (0.5 * math.pi)),
        "jit_t": float(model._network_time(t[:1]).float().item()),
        "matched_x0_mse": _as_float(
            (matched_output - x0.float()).square().mean()
        ),
        "output_mean": _as_float(fixed_output.mean()),
        "output_rms": _as_float(_rms(fixed_output)),
        "output_spatial_std_rms": _as_float(output_spatial_std),
        "output_batch_std_rms": _as_float(output_batch_std),
        "batch_std_over_spatial_std": _as_float(
            output_batch_std / output_spatial_std.clamp_min(1e-12)
        ),
        "cross_sample_corr": _as_float(
            _mean_off_diagonal_correlation(fixed_output)
        ),
        "cross_sample_template_frac": _as_float(
            _cross_sample_template_fraction(fixed_output)
        ),
        "input_delta_rms": _as_float(input_delta),
        "input_output_delta_rms": _as_float(input_output_delta),
        "input_response_gain": _as_float(
            _response_gain(input_output_delta, input_delta)
        ),
        "noise_input_delta_rms": _as_float(noise_input_delta),
        "noise_output_delta_rms": _as_float(noise_output_delta),
        "noise_response_gain": _as_float(
            _response_gain(noise_output_delta, noise_input_delta)
        ),
        "clean_input_delta_rms": _as_float(clean_input_delta),
        "clean_output_delta_rms": _as_float(clean_output_delta),
        "clean_response_gain": _as_float(
            _response_gain(clean_output_delta, clean_input_delta)
        ),
        "label_output_delta_rms": _as_float(
            _rms(fixed_output - alternate_label_output)
        ),
    }
    metrics.update({name: _as_float(value) for name, value in collapse.items()})
    return metrics, fixed_output.cpu(), matched_output.cpu()


def _write_csv(path: Path, rows: Sequence[Mapping[str, float]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_panels(
    output_dir: Path,
    fixed_outputs: Sequence[torch.Tensor],
    matched_outputs: Sequence[torch.Tensor],
    batch_size: int,
) -> None:
    import torchvision

    fixed = torch.cat(list(fixed_outputs), dim=0)
    matched = torch.cat(list(matched_outputs), dim=0)
    torchvision.utils.save_image(
        ((fixed + 1.0) / 2.0).clamp(0, 1),
        output_dir / "fixed_label_outputs_clamped.png",
        nrow=batch_size,
        pad_value=1,
    )
    torchvision.utils.save_image(
        _contrast_normalize(fixed),
        output_dir / "fixed_label_outputs_contrast.png",
        nrow=batch_size,
        pad_value=1,
    )
    torchvision.utils.save_image(
        ((matched + 1.0) / 2.0).clamp(0, 1),
        output_dir / "matched_label_outputs_clamped.png",
        nrow=batch_size,
        pad_value=1,
    )


def _print_table(rows: Sequence[Mapping[str, float]]) -> None:
    header = (
        "       t    jit_t     x0_mse  batch/sp  sample_corr  input_gain "
        "noise_gain clean_gain patch_frac"
    )
    print(header, flush=True)
    for row in rows:
        print(
            f"{row['t']:8.5f} {row['jit_t']:8.5f} "
            f"{row['matched_x0_mse']:11.5g} "
            f"{row['batch_std_over_spatial_std']:9.4f} "
            f"{row['cross_sample_corr']:12.4f} "
            f"{row['input_response_gain']:11.4f} "
            f"{row['noise_response_gain']:10.4f} "
            f"{row['clean_response_gain']:10.4f} "
            f"{row['patch_template_explained_frac']:10.4f}",
            flush=True,
        )


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep physical TrigFlow time and measure direct-x input dependence"
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--checkpoint-key",
        default="model_ema",
        choices=["auto", "model_ema", "model", "training_model"],
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--fixed-class", type=int, default=207)
    parser.add_argument("--times", default=DEFAULT_TIMES)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=["fp32", "bf16", "fp16"], default="bf16"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "work_dirs/diagnostics/scm_xpred_t_sweep",
    )
    parser.add_argument(
        "--constant-gain-max",
        type=float,
        default=0.05,
        help="maximum finite-change input response for a constant-like output",
    )
    parser.add_argument(
        "--constant-batch-ratio-max",
        type=float,
        default=0.1,
        help="maximum batch/spatial output std ratio for a constant-like output",
    )
    parser.add_argument("--no-save-images", action="store_true")
    return parser


def main() -> None:
    args = get_parser().parse_args()
    if args.batch_size < 2:
        raise ValueError("--batch-size must be at least 2 for correlation metrics")
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Run on a GPU node or pass --device cpu "
            "--dtype fp32 (CPU execution is very slow for JiT-B)."
        )
    if device.type == "cpu" and args.dtype == "fp16":
        raise ValueError("CPU diagnostics do not support --dtype fp16")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    print(f"Loading checkpoint: {checkpoint_path}", flush=True)
    checkpoint = _load_checkpoint(checkpoint_path)
    checkpoint_step = int(checkpoint.get("step", -1))
    config = _checkpoint_args(checkpoint)
    state_dict = _select_state_dict(checkpoint, args.checkpoint_key)
    model = _build_model(config, args.batch_size)
    load_result = model.load_state_dict(state_dict, strict=True)
    del state_dict, checkpoint
    model = model.to(device).eval()
    print(f"Loaded {args.checkpoint_key}: {load_result}", flush=True)

    image_size = int(config.get("img_size", 256))
    num_classes = int(config.get("num_classes", 1000))
    if not 0 <= args.fixed_class < num_classes:
        raise ValueError(
            f"--fixed-class must be in [0, {num_classes}), got {args.fixed_class}"
        )
    x0, matched_labels, dataset_indices = _load_real_batch(
        args.data_path.expanduser(), image_size, args.batch_size, args.seed
    )
    x0 = x0.to(device)
    matched_labels = matched_labels.to(device)
    fixed_labels = torch.full_like(matched_labels, args.fixed_class)
    alternate_labels = (
        args.fixed_class
        + torch.arange(1, args.batch_size + 1, device=device, dtype=torch.long)
        * 97
    ) % num_classes

    noise_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    sigma_data = float(config.get("sigma_data", 0.5))
    sigma_max = float(config.get("sigma_max", 80.0))
    noise_a = torch.randn(
        x0.shape, device=device, generator=noise_generator, dtype=x0.dtype
    ) * sigma_data
    noise_b = torch.randn(
        x0.shape, device=device, generator=noise_generator, dtype=x0.dtype
    ) * sigma_data
    times = _parse_times(args.times, sigma_data, sigma_max)

    run_name = f"{checkpoint_path.stem}-{args.checkpoint_key}"
    output_dir = args.output_dir.expanduser().resolve() / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "panel_row_times.txt").write_text(
        "\n".join(f"row {index}: t={value:.9f}" for index, value in enumerate(times))
        + "\n"
    )

    rows = []
    fixed_outputs = []
    matched_outputs = []
    for index, physical_t in enumerate(times):
        metrics, fixed_output, matched_output = _evaluate_time(
            model=model,
            x0=x0,
            matched_labels=matched_labels,
            fixed_labels=fixed_labels,
            alternate_labels=alternate_labels,
            noise_a=noise_a,
            noise_b=noise_b,
            physical_t=physical_t,
            amp_dtype=args.dtype,
        )
        rows.append(metrics)
        fixed_outputs.append(fixed_output)
        matched_outputs.append(matched_output)
        print(
            f"[{index + 1:02d}/{len(times):02d}] t={physical_t:.6f} "
            f"x0_mse={metrics['matched_x0_mse']:.6g} "
            f"sample_corr={metrics['cross_sample_corr']:.4f} "
            f"input_gain={metrics['input_response_gain']:.4f}",
            flush=True,
        )

    constant_like_times = [
        row["t"]
        for row in rows
        if row["input_response_gain"] <= args.constant_gain_max
        and row["batch_std_over_spatial_std"]
        <= args.constant_batch_ratio_max
    ]

    _write_csv(output_dir / "metrics.csv", rows)
    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_key": args.checkpoint_key,
        "checkpoint_step": checkpoint_step,
        "data_path": str(args.data_path.expanduser().resolve()),
        "dataset_indices": dataset_indices,
        "matched_labels": [int(value) for value in matched_labels.cpu()],
        "fixed_class": args.fixed_class,
        "alternate_labels": [int(value) for value in alternate_labels.cpu()],
        "batch_size": args.batch_size,
        "seed": args.seed,
        "dtype": args.dtype,
        "sigma_data": sigma_data,
        "sigma_max": sigma_max,
        "boundary_band_max": float(config.get("boundary_band_max", 0.02)),
        "network_time_mode": str(config.get("network_time_mode", "legacy_reversed")),
        "constant_like_thresholds": {
            "input_response_gain_max": args.constant_gain_max,
            "batch_std_over_spatial_std_max": args.constant_batch_ratio_max,
        },
        "constant_like_times": constant_like_times,
        "metrics": rows,
    }
    with (output_dir / "report.json").open("w") as handle:
        json.dump(report, handle, indent=2)

    if not args.no_save_images:
        _save_panels(
            output_dir,
            fixed_outputs,
            matched_outputs,
            args.batch_size,
        )

    print(flush=True)
    _print_table(rows)
    if constant_like_times:
        formatted_times = ", ".join(f"{value:.6f}" for value in constant_like_times)
        print(
            "\nConstant-like output detected at swept t values: "
            f"{formatted_times}",
            flush=True,
        )
    else:
        print(
            "\nNo swept t value met the input-independent thresholds "
            f"(input_gain <= {args.constant_gain_max:g}, "
            "batch_std/spatial_std <= "
            f"{args.constant_batch_ratio_max:g}).",
            flush=True,
        )
    print(f"\nSaved diagnostic report to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
