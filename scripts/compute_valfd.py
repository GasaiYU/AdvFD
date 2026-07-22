"""Reproduce the validation-set raw FD normalizers used for FDr/FDr-6.

Run from the repository root:
    torchrun --nproc_per_node=8 scripts/compute_valfd.py --data_root /path/to/imagenet

The script evaluates ImageNet validation images against the released paper
reference statistics in data/fid_stats/ and writes raw validation FD values.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frechet_distance.metrics import compute_fid
from frechet_distance.repr_models import load_repr_model
from utils.data_util import center_crop_arr
from utils.distributed_util import enable_distributed, get_global_rank, get_world_size


logger = logging.getLogger("FD_loss")


@dataclass(frozen=True)
class ValFDSpec:
    label: str
    model: str
    stats_path: str
    mu_key: str = "mu"
    sigma_key: str = "sigma"
    img_size: int = 256
    target_size: int | None = None
    expected: float | None = None


DEFAULT_SPECS = [
    ValFDSpec(
        "Inception",
        "inception",
        "data/fid_stats/guided_diffusion_stats.npz",
        expected=1.68,
    ),
    ValFDSpec(
        "ConvNeXt",
        "convnext",
        "data/fid_stats/convnext_in256_t224_stats.npz",
        target_size=224,
        expected=56.87,
    ),
    ValFDSpec(
        "DINOv2",
        "vit_large_patch14_dinov2.lvd142m",
        "data/fid_stats/vit_large_patch14_dinov2_lvd142m_in256_t256_stats.npz",
        target_size=256,
        expected=14.19,
    ),
    ValFDSpec(
        "MAE",
        "vit_large_patch16_224.mae",
        "data/fid_stats/vit_large_patch16_224_mae_in256_t224_stats.npz",
        target_size=224,
        expected=0.04,
    ),
    ValFDSpec(
        "SigLIP",
        "vit_so400m_patch16_siglip_256.v2_webli",
        "data/fid_stats/vit_so400m_patch16_siglip_256_v2_webli_in256_t224_stats.npz",
        target_size=224,
        expected=0.60,
    ),
    ValFDSpec(
        "InternViT",
        os.environ.get(
            "INTERNVIT",
            "/mmu-vcg/gaomingju/data/models/OpenGVLab/InternViT-300M-448px",
        ),
        "data/fid_stats/internvit_300m_in256_t224_stats.npz",
        target_size=224,
    ),
    ValFDSpec(
        "CLIP",
        "vit_large_patch14_clip_224.openai",
        "data/fid_stats/vit_large_patch14_clip_224_openai_in256_t256_stats.npz",
        target_size=256,
        expected=5.60,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute validation-set raw FD values used for FDr/FDr-6."
    )
    parser.add_argument("--data_root", default=os.environ.get("DATA_ROOT", "data/imagenet"))
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--output_json", default="data/fid_stats/valfd.json")
    parser.add_argument("--output_csv", default="data/fid_stats/valfd.csv")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[s.label for s in DEFAULT_SPECS],
        choices=[s.label for s in DEFAULT_SPECS],
        help="Subset of validation FD normalizers to compute.",
    )
    return parser.parse_args()


def setup_distributed() -> tuple[int, int]:
    enable_distributed()
    rank = get_global_rank()
    world_size = get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    return rank, world_size


def build_loader(data_root: str, img_size: int, batch_size: int, num_workers: int,
                 rank: int, world_size: int) -> tuple[DataLoader, int]:
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, img_size)),
        transforms.ToTensor(),
    ])
    val_dir = os.path.join(data_root, "val")
    dataset = datasets.ImageFolder(val_dir, transform=transform)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    ) if world_size > 1 else None
    loader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    return loader, len(dataset)


@torch.inference_mode()
def compute_stats(spec: ValFDSpec, model: torch.nn.Module, loader: DataLoader,
                  feat_dim: int, has_logits: bool,
                  rank: int, world_size: int) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    device = torch.device("cuda")
    feat_sum = torch.zeros(feat_dim, dtype=torch.float64, device=device)
    feat_outer = torch.zeros(feat_dim, feat_dim, dtype=torch.float64, device=device)
    count = 0

    pbar = tqdm(loader, desc=f"{spec.label} val features", disable=rank != 0)
    for images, _ in pbar:
        images = images.to(device, non_blocking=True)
        if has_logits:
            feats, _ = model(images)
        else:
            with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                feats, _ = model(images)
        feats64 = feats.double()
        feat_sum.add_(feats64.sum(0))
        feat_outer.addmm_(feats64.T, feats64)
        count += feats.shape[0]

    if world_size > 1:
        dist.reduce(feat_sum, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(feat_outer, dst=0, op=dist.ReduceOp.SUM)
        count_t = torch.tensor([count], dtype=torch.long, device=device)
        dist.reduce(count_t, dst=0, op=dist.ReduceOp.SUM)
        count = int(count_t.item())

    if rank != 0:
        return None, None, count

    s = feat_sum.cpu().numpy()
    mu = s / count
    sigma = (feat_outer.cpu().numpy() - np.outer(s, s) / count) / (count - 1)
    return mu, sigma, count


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name in ("httpx", "timm", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.WARNING)
    args = parse_args()
    rank, world_size = setup_distributed()
    if rank != 0:
        logger.setLevel(logging.WARNING)

    selected = {name.lower() for name in args.models}
    specs = [s for s in DEFAULT_SPECS if s.label.lower() in selected]
    results = []

    for spec in specs:
        if rank == 0:
            logger.info(f"\nComputing valFD for {spec.label}")
            if not os.path.exists(spec.stats_path):
                raise FileNotFoundError(
                    f"Missing {spec.stats_path}. Run scripts/extract_paper_ref_stats.py first."
                )

        repr_model, feat_dim, has_logits, _ = load_repr_model(
            spec.model, device="cuda", target_size=spec.target_size
        )

        loader, n_images = build_loader(
            args.data_root, spec.img_size, args.batch_size,
            args.num_workers, rank, world_size,
        )
        mu, sigma, count = compute_stats(
            spec, repr_model, loader, feat_dim, has_logits, rank, world_size
        )

        if rank == 0:
            ref = np.load(spec.stats_path)
            fd = compute_fid(mu, sigma, ref[spec.mu_key], ref[spec.sigma_key])
            row = {
                "representation": spec.label,
                "model": spec.model,
                "valfd": round(float(fd), 6),
                "paper_valfd": spec.expected,
                "n": int(count),
                "stats_path": spec.stats_path,
            }
            results.append(row)
            logger.info(
                f"{spec.label}: valFD={row['valfd']:.6f} "
                f"(paper rounded={spec.expected}, n={count}/{n_images})"
            )

        del repr_model, loader
        torch.cuda.empty_cache()

    if rank == 0:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
            f.write("\n")
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["representation", "model", "valfd", "paper_valfd", "n", "stats_path"]
            )
            writer.writeheader()
            writer.writerows(results)
        logger.info(f"\nWrote {args.output_json}")
        logger.info(f"Wrote {args.output_csv}")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
