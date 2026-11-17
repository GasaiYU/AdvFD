#!/usr/bin/env python3
"""Probe candidate frozen encoders for a W3-style "add more frozen encoders" run.

For every candidate this script, on one GPU:
  * loads it through frechet_distance.repr_models.load_repr_model exactly as
    main_fd.py does (frozen, eval, fp32 forward), i.e. checks the timm name;
  * reports parameter count, feature dim and input size;
  * times forward + backward to the input on a batch of images and records
    the extra peak memory, with and without gradient checkpointing;
  * flags candidates that overlap the six evaluation encoders (FDr6) or the
    three held-out ones (FDr3), which must never be used for training.

It does not train anything and needs no reference statistics. Example:

    CUDA_VISIBLE_DEVICES=0 python scripts/rebuttal/probe_frozen_encoders.py \
        --batch_size 64 --out work_dirs/rebuttal_w3/encoder_probe.csv

Pass --models to probe other names. Per-GPU batch in the 16-GPU W3 runs is 64.
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# (timm name, family, pretraining data); ImageNet-1K-only supervised models
# first, so the "more parameters" control stays close to Inception's data.
CANDIDATES = [
    ("resnet50.a1_in1k", "CNN", "IN-1k sup"),
    ("resnet101.a1_in1k", "CNN", "IN-1k sup"),
    ("resnet152.a1_in1k", "CNN", "IN-1k sup"),
    ("wide_resnet50_2.tv2_in1k", "CNN", "IN-1k sup"),
    ("regnety_160.deit_in1k", "CNN", "IN-1k sup"),
    ("efficientnet_b3.ra2_in1k", "CNN", "IN-1k sup"),
    ("densenet121.tv_in1k", "CNN", "IN-1k sup"),
    ("vgg16.tv_in1k", "CNN", "IN-1k sup"),
    ("swin_tiny_patch4_window7_224.ms_in1k", "Transformer", "IN-1k sup"),
    ("swin_base_patch4_window7_224.ms_in1k", "Transformer", "IN-1k sup"),
    ("deit3_base_patch16_224.fb_in1k", "Transformer", "IN-1k sup"),
    ("deit_base_patch16_224.fb_in1k", "Transformer", "IN-1k sup"),
    ("vit_base_patch16_224.augreg_in1k", "Transformer", "IN-1k sup"),
    ("beit_base_patch16_224.in22k_ft_in22k_in1k", "Transformer", "IN-22k MIM+sup"),
]
# Substrings that mark the six evaluation encoders and their close relatives.
EVAL_FAMILIES = {
    "dinov2": "FDr3 held-out (DINOv2)", "dino": "FDr3 held-out (DINO family)",
    "clip": "FDr3 held-out (CLIP)", "convnext": "FDr3 held-out (ConvNeXt)",
    "siglip": "FDr6, already in SIM", "mae": "FDr6, already in SIM",
    "inception": "FDr6, already in SIM",
}


def eval_overlap(name):
    low = name.lower()
    return next((why for key, why in EVAL_FAMILIES.items() if key in low), "")


def probe(name, batch_size, img_size, grad_checkpointing, repeats):
    import torch
    from frechet_distance.repr_models import load_repr_model

    torch.cuda.empty_cache()
    model, feat_dim, _, target_size = load_repr_model(
        name, device="cuda", target_size=224, grad_checkpointing=grad_checkpointing)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    gc_active = bool(getattr(model, "grad_checkpointing", False))
    x = torch.rand(batch_size, 3, img_size, img_size, device="cuda")

    def step():
        inp = x.clone().requires_grad_(True)
        feats = model(inp)[0].float()
        # Same reduction to the input as the FD loss: a scalar of all features.
        feats.square().mean().backward()
        return inp.grad

    step()  # warm-up: cudnn autotune, lazy init
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(repeats):
        grad = step()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) / repeats * 1000
    peak_gib = (torch.cuda.max_memory_allocated() - base) / 1024**3
    finite = bool(torch.isfinite(grad).all())
    del model, x, grad
    gc.collect()
    torch.cuda.empty_cache()
    return {"params_m": params, "feat_dim": feat_dim, "target_size": target_size,
            "grad_ckpt_active": gc_active, "fwd_bwd_ms": ms, "extra_peak_gib": peak_gib,
            "grad_finite": finite}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=None, help="timm names; default: built-in candidates")
    parser.add_argument("--batch_size", type=int, default=64, help="per-GPU batch of the training run")
    parser.add_argument("--img_size", type=int, default=256, help="generator output resolution")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None, help="optional CSV path")
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA GPU")
    info = {name: (family, data) for name, family, data in CANDIDATES}
    names = args.models or [name for name, _, _ in CANDIDATES]
    rows = []
    for name in names:
        family, data = info.get(name, ("?", "?"))
        row = {"model": name, "family": family, "pretrain": data, "eval_overlap": eval_overlap(name)}
        for gc_flag in (False, True):
            tag = "ckpt" if gc_flag else "plain"
            try:
                result = probe(name, args.batch_size, args.img_size, gc_flag, args.repeats)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                row[f"{tag}_status"] = "OOM"
                continue
            except Exception as exc:  # unknown timm name, download failure, ...
                row[f"{tag}_status"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:80]}"
                break
            row.update({k: result[k] for k in ("params_m", "feat_dim", "target_size")})
            row[f"{tag}_status"] = "ok" if result["grad_finite"] else "non-finite grad"
            row[f"{tag}_ms"] = result["fwd_bwd_ms"]
            row[f"{tag}_gib"] = result["extra_peak_gib"]
            row[f"{tag}_active"] = result["grad_ckpt_active"]
        rows.append(row)
        print(f"[probe] {name}: {row}", flush=True)

    print(f"\nbatch={args.batch_size}, input {args.img_size}px -> resized to each model's size, "
          "fp32 forward + backward to the input (as in main_fd.py)")
    print(f"{'model':44} {'Mparam':>7} {'dim':>5} {'ms':>8} {'GiB':>6} {'ckpt ms':>8} {'ckpt GiB':>8}  note")
    for row in rows:
        def num(key, fmt):
            return format(row[key], fmt) if key in row else row.get(key.split("_")[0] + "_status", "-")[:8]
        note = row["eval_overlap"] or ("" if row.get("plain_status") == "ok" else row.get("plain_status", ""))
        if row.get("ckpt_status") == "ok" and not row.get("ckpt_active"):
            note = (note + "; " if note else "") + "no grad ckpt (same as plain)"
        print(f"{row['model']:44} {num('params_m', '7.1f'):>7} {str(row.get('feat_dim', '-')):>5} "
              f"{num('plain_ms', '8.1f'):>8} {num('plain_gib', '6.2f'):>6} "
              f"{num('ckpt_ms', '8.1f'):>8} {num('ckpt_gib', '8.2f'):>8}  {note}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fields = sorted({key for row in rows for key in row}, key=lambda k: (k != "model", k))
        with args.out.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[probe] saved {args.out}")


if __name__ == "__main__":
    main()
