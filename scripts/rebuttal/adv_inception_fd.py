#!/usr/bin/env python3
"""FD of image folders in the adaptive (AdvFD) Inception space and in the
pretrained Inception space.

For each encoder the real statistics (mu_r, Sigma_r) are re-estimated from
ImageNet train in that same space: the stored Inception reference stats are
not valid for the fine-tuned network. Two numbers are reported per folder:

  fd        standard Frechet distance, same formula as the usual FID.
            Its scale follows the feature scale, so the value is only
            comparable within one encoder.
  fd_white  real-whitened FD, the quantity AdvFD optimizes
            (frechet_distance.adversarial, eps = --fd_adv_whiten_eps).
            Features are whitened by the real covariance, so the scale of
            the encoder drops out and the two spaces can be compared.

A "real_holdout" row (real images disjoint from the reference set) gives the
finite-sample floor in each space. Compare folders only at the same N.

Usage (single GPU):
    python scripts/rebuttal/adv_inception_fd.py \
        --checkpoint work_dirs/.../JiT_B-fd-sim-advinc-w0.1-from-base-ADVLR-1e-6/checkpoints \
        --data_root /path/to/ImageNet-1K \
        --image_folder original=/path/to/jit_b_fd_loss \
        --image_folder noise=/path/to/jit_b_fd_loss_noise \
        --out_dir work_dirs/rebuttal_adv_fd/jit_b

Each --image_folder is LABEL=PATH; add --panel K when the images are
horizontal strips of square panels (e.g. jit_l_threeway, FD-loss = panel 1).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adaptive_flag_quality import (  # noqa: E402  (also puts the repo root on sys.path)
    ENCODERS,
    IMAGE_EXTS,
    PanelDataset,
    extract_features,
    list_samples,
    load_encoders,
    write_csv,
)

STATS_VERSION = 1


def split_real_paths(data_root, per_class, holdout_per_class, seed):
    """Class-balanced reference set and a disjoint holdout set from ImageNet train."""
    train_dir = Path(data_root)
    if (train_dir / "train").is_dir():
        train_dir = train_dir / "train"
    rng = random.Random(seed)
    reference, holdout = [], []
    for wnid in sorted(os.listdir(train_dir)):
        class_dir = train_dir / wnid
        if not class_dir.is_dir():
            continue
        files = sorted(f for f in os.listdir(class_dir) if Path(f).suffix.lower() in IMAGE_EXTS)
        picked = rng.sample(files, min(per_class + holdout_per_class, len(files)))
        reference += [str(class_dir / f) for f in picked[:per_class]]
        holdout += [str(class_dir / f) for f in picked[per_class:]]
    if not reference:
        raise FileNotFoundError(f"no ImageNet class folders under {train_dir}")
    return reference, holdout


def moments(features):
    """Mean and unbiased covariance of an [N, D] CPU tensor, accumulated in float64."""
    import torch

    n, dim = features.shape
    if n < 2:
        raise ValueError("need at least 2 images to estimate a covariance")
    total = torch.zeros(dim, dtype=torch.float64).cuda()
    second = torch.zeros(dim, dim, dtype=torch.float64).cuda()
    for chunk in features.split(8192):
        chunk = chunk.cuda().double()
        total += chunk.sum(dim=0)
        second.addmm_(chunk.T, chunk)
    mu = total / n
    cov = (second - n * torch.outer(mu, mu)) / (n - 1)
    return mu, 0.5 * (cov + cov.T)


def frechet_values(real, fake, whitening, eps):
    from frechet_distance.adversarial import real_whitened_frechet_distance_from_stats
    from frechet_distance.metrics import compute_fid

    (mu_r, cov_r), (mu_f, cov_f) = real, fake
    raw = compute_fid(mu_r.cpu().numpy(), cov_r.cpu().numpy(), mu_f.cpu().numpy(), cov_f.cpu().numpy())
    white = real_whitened_frechet_distance_from_stats(
        mu_r, cov_r, mu_f, cov_f, eps=eps, real_whitening=whitening
    )
    return float(raw), float(white)


def real_statistics(args, encoders, checkpoint_path):
    """Reference and holdout moments per encoder, cached on disk."""
    import torch
    from frechet_distance.datasets import ImageListDataset

    stat = os.stat(checkpoint_path)
    meta = {
        "version": STATS_VERSION, "checkpoint": str(checkpoint_path),
        "checkpoint_size": stat.st_size, "checkpoint_mtime_ns": stat.st_mtime_ns,
        "data_root": str(args.data_root), "real_per_class": args.real_per_class,
        "holdout_per_class": args.holdout_per_class, "img_size": args.img_size, "seed": args.seed,
    }
    cache = Path(args.out_dir) / "real_stats.npz"
    if cache.exists() and not args.recompute_real:
        saved = np.load(cache, allow_pickle=False)
        if json.loads(str(saved["meta"])) == meta:
            print(f"[real] reusing {cache}", flush=True)
            out = {}
            for name in ENCODERS:
                out[name] = {"counts": {p: int(saved[f"{p}_count"]) for p in ("reference", "holdout")}}
                for part in ("reference", "holdout"):
                    if f"{name}_{part}_mu" in saved:
                        out[name][part] = tuple(
                            torch.from_numpy(saved[f"{name}_{part}_{k}"]).cuda() for k in ("mu", "cov")
                        )
            return out
        print("[real] cache does not match this run; recomputing", flush=True)

    reference, holdout = split_real_paths(args.data_root, args.real_per_class,
                                          args.holdout_per_class, args.seed)
    print(f"[real] {len(reference)} reference + {len(holdout)} holdout ImageNet images", flush=True)
    out = {name: {"counts": {"reference": len(reference), "holdout": len(holdout)}} for name in ENCODERS}
    arrays = {"meta": np.array(json.dumps(meta)),
              "reference_count": len(reference), "holdout_count": len(holdout)}
    for part, paths in (("reference", reference), ("holdout", holdout)):
        if not paths:
            continue
        feats = extract_features(encoders, ImageListDataset(paths, img_size=args.img_size),
                                 args.batch_size, args.num_workers, f"real {part}")
        for name in ENCODERS:
            out[name][part] = moments(feats[name])
            for key, value in zip(("mu", "cov"), out[name][part]):
                arrays[f"{name}_{part}_{key}"] = value.cpu().numpy()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    np.savez(cache, **arrays)
    return out


def folder_names(path, num_images, seed):
    names = list_samples(path)
    if 0 < num_images < len(names):  # seeded random subset: file order is class-major
        names = sorted(random.Random(seed).sample(names, num_images))
    return names


def parse_folder(spec):
    label, sep, path = spec.partition("=")
    if not sep or not label or not path:
        raise argparse.ArgumentTypeError(f"expected LABEL=PATH, got {spec!r}")
    return label, path


def print_table(rows, baseline_label):
    base = {r["encoder"]: r for r in rows if r["label"] == baseline_label}
    for name in ENCODERS:
        print(f"\n[{name}]  baseline = {baseline_label}")
        print(f"{'folder':28} {'N':>7} {'FD':>10} {'FD/base':>8} {'FD_white':>10} {'white/base':>10}")
        for row in (r for r in rows if r["encoder"] == name):
            b = base.get(name)
            rel = f"{row['fd'] / b['fd']:8.3f}" if b and b["fd"] else f"{'-':>8}"
            rel_w = f"{row['fd_white'] / b['fd_white']:10.3f}" if b and b["fd_white"] else f"{'-':>10}"
            print(f"{row['label']:28} {row['num_images']:7d} {row['fd']:10.3f} {rel} "
                  f"{row['fd_white']:10.3f} {rel_w}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="AdvFD checkpoint (file or dir) with fd_adv_states")
    parser.add_argument("--data_root", required=True, help="ImageNet root containing train/")
    parser.add_argument("--image_folder", type=parse_folder, action="append", required=True,
                        metavar="LABEL=PATH", help="repeatable; the first one is the baseline for ratios")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--panel", type=int, default=0, help="panel index when images are strips of square panels")
    parser.add_argument("--num_images", type=int, default=0,
                        help="seeded random subset per folder, 0 = all; use the same N for every folder")
    parser.add_argument("--real_per_class", type=int, default=100)
    parser.add_argument("--holdout_per_class", type=int, default=50)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--whiten_eps", type=float, default=1e-3, help="match --fd_adv_whiten_eps")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--recompute_real", action="store_true")
    args = parser.parse_args()

    import torch
    from frechet_distance.adversarial import build_real_whitening

    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA GPU")
    labels = [label for label, _ in args.image_folder]
    if len(set(labels)) != len(labels):
        parser.error("--image_folder labels must be unique")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    encoders, checkpoint_path = load_encoders(args.checkpoint)
    real = real_statistics(args, encoders, checkpoint_path)
    whitening = {
        name: build_real_whitening(*real[name]["reference"], eps=args.whiten_eps) for name in ENCODERS
    }

    rows = []
    for name in ENCODERS:
        if "holdout" in real[name]:
            fd, white = frechet_values(real[name]["reference"], real[name]["holdout"],
                                       whitening[name], args.whiten_eps)
            rows.append({"label": "real_holdout", "encoder": name,
                         "num_images": real[name]["counts"]["holdout"], "fd": fd, "fd_white": white})

    for label, path in args.image_folder:
        names = folder_names(path, args.num_images, args.seed)
        feats = extract_features(encoders, PanelDataset(path, names, args.panel),
                                 args.batch_size, args.num_workers, label)
        for name in ENCODERS:
            fd, white = frechet_values(real[name]["reference"], moments(feats[name]),
                                       whitening[name], args.whiten_eps)
            rows.append({"label": label, "encoder": name, "num_images": len(names), "fd": fd, "fd_white": white})
        del feats

    sizes = {r["num_images"] for r in rows if r["label"] in labels}
    if len(sizes) > 1:
        print(f"[warn] folders have different N {sorted(sizes)}; FD is biased by N, pass --num_images")
    write_csv(Path(args.out_dir) / "adv_inception_fd.csv", rows)
    print_table(rows, labels[0])
    print(f"\nwhiten_eps={args.whiten_eps}, reference={args.real_per_class * 1000} real images/space")
    print(f"[fd] saved {Path(args.out_dir) / 'adv_inception_fd.csv'}")


if __name__ == "__main__":
    main()
