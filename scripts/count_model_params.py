#!/usr/bin/env python3
"""Count parameters of the B/L generation models used in Table 3.

The models are created on PyTorch's ``meta`` device by default, so this script
does not need a GPU, checkpoint, or enough RAM to materialize the weights.
Counts cover only the generation model.  They intentionally exclude the VAE,
EMA copies, FD representation networks, and adversarial critics.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.denoiser_imf import iMFDenoiser_models  # noqa: E402
from models.denoiser_jit import JiTDenoiser_models  # noqa: E402
from models.denoiser_pmf import pMFDenoiser_models  # noqa: E402


DEFAULT_MODELS = ("JiT_B", "JiT_L", "pMF_B", "pMF_L", "iMF_B", "iMF_L")
MODEL_ALIASES = {
    alias: model_name
    for model_name in DEFAULT_MODELS
    for alias in (model_name, model_name.replace("_", "-"))
}


def build_table3_model(model_name: str) -> torch.nn.Module:
    """Build one model with the corresponding 256px Table 3 configuration."""
    if model_name in JiTDenoiser_models:
        return JiTDenoiser_models[model_name](
            img_size=256,
            in_channels=3,
            num_classes=1000,
            rope_2d=True,
            learned_pe=True,
            legacy_time_convention=True,
            grad_checkpointing=False,
        )

    if model_name in pMFDenoiser_models:
        return pMFDenoiser_models[model_name](
            img_size=256,
            patch_size=16,
            in_channels=3,
            tokenizer_patch_size=1,
            num_classes=1000,
            rope_2d=True,
            learned_pe=True,
            disable_v_head=True,
            grad_checkpointing=False,
        )

    if model_name in iMFDenoiser_models:
        return iMFDenoiser_models[model_name](
            img_size=256,
            patch_size=2,
            in_channels=4,
            tokenizer_patch_size=8,
            num_classes=1000,
            rope_2d=False,
            learned_pe=False,
            disable_v_head=True,
            grad_checkpointing=False,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def normalize_model_names(values: list[str]) -> list[str]:
    normalized = []
    for value in values:
        try:
            model_name = MODEL_ALIASES[value]
        except KeyError as exc:
            valid = ", ".join(DEFAULT_MODELS)
            raise ValueError(f"Unknown model {value!r}; choose from: {valid}") from exc
        if model_name not in normalized:
            normalized.append(model_name)
    return normalized


def format_table(rows: list[tuple[str, int, int, int]]) -> str:
    headers = ("Model", "Total params", "Trainable", "Frozen", "Total (M)")
    values = [
        (
            model_name.replace("_", "-"),
            f"{total:,}",
            f"{trainable:,}",
            f"{frozen:,}",
            f"{total / 1_000_000:.2f}",
        )
        for model_name, total, trainable, frozen in rows
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in values))
        for index in range(len(headers))
    ]

    def render(row: tuple[str, ...]) -> str:
        return "  ".join(
            value.ljust(widths[index]) if index == 0 else value.rjust(widths[index])
            for index, value in enumerate(row)
        )

    separator = "  ".join("-" * width for width in widths)
    return "\n".join((render(headers), separator, *(render(row) for row in values)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        metavar="MODEL",
        help="models to count; accepts names with '-' or '_' (default: all six)",
    )
    parser.add_argument(
        "--device",
        choices=("meta", "cpu"),
        default="meta",
        help="construction device (default: meta; cpu materializes full weights)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        model_names = normalize_model_names(args.models)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    rows = []
    for model_name in model_names:
        with torch.device(args.device):
            model = build_table3_model(model_name)

        total = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        if not 0 < trainable <= total:
            raise RuntimeError(
                f"Invalid parameter counts for {model_name}: "
                f"total={total}, trainable={trainable}"
            )
        rows.append((model_name, total, trainable, total - trainable))
        del model
        gc.collect()

    print("Table 3 architecture: 256px; iMF uses 32x32 SD-VAE latents.")
    print("Scope: generator only (no VAE, EMA, FD network, or adversarial critic).")
    print(format_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
