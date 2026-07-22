#!/usr/bin/env python3
"""Compute ImageNet real-image stats in the frozen projector feature space."""

import argparse
import logging
import math
import os
import sys

import numpy as np
import torch
import torch.distributed as dist
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.projector import (
    ViTSMultiHeadProjector,
    canonical_projector_backbone,
    infer_projector_backbone,
    projector_backbone_config,
    projector_backbone_tag,
)
from utils.data_util import center_crop_arr
from utils.distributed_util import enable_distributed, get_global_rank, get_world_size


logger = logging.getLogger("FD_loss")

DEFAULT_CHECKPOINT = (
    "/mmu-vcg/gaomingju/workspace/foundation/FD-Loss-Ours/"
    "work_dirs/projector/vits-token-vII-inception-mae-siglip-mse-fdema-muon/"
    "checkpoints/last.pt"
)
DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}
HEAD_SOURCE_MODES = {
    "inception": "patches",
    "mae": "prefix1",
    "siglip": "prefix0",
}
HEAD_VARIANTS = ("default", "siglip_deep_inception_conv")
DEFAULT_OUTPUT_NAMES = {
    "inception": "projector_vits_vII_inception_in256_stats.npz",
    "mae": "projector_vits_vII_mae_in256_stats.npz",
    "siglip": "projector_vits_vII_siglip_in256_stats.npz",
}


def parse_args():
    parser = argparse.ArgumentParser("Compute FD projector reference stats")
    parser.add_argument("--data_path", type=str, required=True,
                        help="ImageNet root with train/ subdirectory")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output_dir", type=str, default="data/fid_stats")
    parser.add_argument("--heads", type=str, nargs="+",
                        default=["inception", "mae", "siglip"],
                        choices=["inception", "mae", "siglip"])
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--num_images", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--dtype", type=str, default="bf16",
                        choices=["bf16", "fp16", "fp32"])

    parser.add_argument("--projector_version", type=str, default="II",
                        choices=["I", "II", "III"])
    parser.add_argument("--backbone", type=str, default="vit_s",
                        help="projector backbone preset: vit_s or vit_b")
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--embed_dim", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument("--head_variant", type=str, default="default", choices=HEAD_VARIANTS)
    parser.add_argument("--head_mlp_layers", type=int, default=1)
    parser.add_argument("--head_hidden_dim", type=int, default=None)
    parser.add_argument("--head_conv_layers", type=int, default=2)
    parser.add_argument("--head_conv_kernel_size", type=int, default=3)
    parser.add_argument("--no_input_normalize", action="store_true")
    return parser.parse_args()


def setup_distributed():
    enable_distributed()
    rank = get_global_rank()
    world_size = get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    return rank, world_size


def strip_state_prefix(state_dict, prefix):
    if not any(key.startswith(prefix) for key in state_dict):
        return state_dict
    return {
        (key[len(prefix):] if key.startswith(prefix) else key): value
        for key, value in state_dict.items()
    }


def infer_head_dims(state_dict):
    candidates = {}
    for key, value in state_dict.items():
        if not key.startswith("heads.") or not key.endswith(".weight"):
            continue
        if getattr(value, "ndim", 0) not in (2, 4):
            continue
        parts = key.split(".")
        if len(parts) < 4 or parts[2] != "net":
            continue
        head = parts[1]
        order = int(parts[3]) if len(parts) > 4 and parts[3].isdigit() else 0
        candidates.setdefault(head, []).append((order, int(value.shape[0])))

    head_dims = {}
    for head, entries in candidates.items():
        _, dim = max(entries, key=lambda item: item[0])
        head_dims[head] = dim
    if not head_dims:
        raise RuntimeError("Could not infer projector head dims from checkpoint")
    return head_dims


def head_source_modes_for_version(projector_version, head_variant="default"):
    modes = dict(HEAD_SOURCE_MODES)
    if projector_version == "I":
        modes["mae"] = "prefix0_patches"
        modes["siglip"] = "prefix0_patches"
    elif projector_version == "III":
        modes["inception"] = "prefix2"
    if head_variant == "siglip_deep_inception_conv":
        modes["inception"] = "patches"
    return modes


def head_config_for_variant(head_dims, head_variant, base_mlp_layers, conv_layers):
    head_types = {}
    head_mlp_layer_overrides = {}
    if head_variant == "default":
        return head_types, head_mlp_layer_overrides
    if head_variant != "siglip_deep_inception_conv":
        raise NotImplementedError(head_variant)
    if "inception" in head_dims:
        head_types["inception"] = "conv"
        head_mlp_layer_overrides["inception"] = conv_layers
    if "siglip" in head_dims:
        head_mlp_layer_overrides["siglip"] = base_mlp_layers + 2
    return head_types, head_mlp_layer_overrides


def head_target_grid(head, source_mode, grid, projector_version, head_variant):
    if (
        head_variant == "siglip_deep_inception_conv"
        and projector_version == "III"
        and head == "inception"
    ):
        return (1, 1)
    if source_mode in ("patches", "prefix0_patches"):
        return grid
    return (1, 1)


def resolve_head_variant(args, checkpoint):
    ckpt_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    ckpt_variant = ckpt_args.get("head_variant") if ckpt_args else None
    if args.head_variant == "default" and ckpt_variant in HEAD_VARIANTS:
        return ckpt_variant
    return args.head_variant


def apply_projector_backbone_defaults(args, checkpoint):
    ckpt_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    ckpt_backbone = ckpt_args.get("backbone") if ckpt_args else None
    args.backbone = canonical_projector_backbone(ckpt_backbone or args.backbone)
    config = projector_backbone_config(args.backbone)

    for key in ("img_size", "patch_size", "embed_dim", "depth", "num_heads", "mlp_ratio"):
        ckpt_value = ckpt_args.get(key) if ckpt_args else None
        if ckpt_value is not None:
            setattr(args, key, ckpt_value)
        elif getattr(args, key) is None:
            setattr(args, key, config[key])

    for key in ("head_mlp_layers", "head_hidden_dim", "head_conv_layers", "head_conv_kernel_size"):
        ckpt_value = ckpt_args.get(key) if ckpt_args else None
        if ckpt_value is not None:
            setattr(args, key, ckpt_value)
        elif key == "head_hidden_dim" and getattr(args, key) is None:
            setattr(args, key, config[key])

    if ckpt_args.get("projector_version") is not None:
        args.projector_version = ckpt_args["projector_version"]
    if ckpt_backbone is None:
        args.backbone = infer_projector_backbone(
            args.patch_size,
            args.embed_dim,
            args.depth,
            args.num_heads,
            default=args.backbone,
        )


def default_output_name(head, projector_version, img_size, head_variant="default", backbone="vit_s"):
    tag = projector_backbone_tag(backbone)
    version = projector_version
    if version == "II" and tag == "vits":
        name = DEFAULT_OUTPUT_NAMES[head]
    else:
        name = f"projector_{tag}_v{version}_{head}_in{img_size}_stats.npz"
    if head_variant != "default":
        return name.replace("_stats.npz", f"_{head_variant}_stats.npz")
    return name


def load_projector(args):
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)
    try:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    state_dict = strip_state_prefix(state_dict, "module.")
    state_dict = strip_state_prefix(state_dict, "_orig_mod.")

    apply_projector_backbone_defaults(args, checkpoint)
    head_variant = resolve_head_variant(args, checkpoint)
    args.head_variant = head_variant
    head_dims = infer_head_dims(state_dict)
    missing = sorted(set(args.heads) - set(head_dims))
    if missing:
        raise ValueError(
            "Projector checkpoint is missing requested heads: "
            + ", ".join(missing)
            + f". Available heads: {', '.join(sorted(head_dims))}"
        )

    grid = (args.img_size // args.patch_size, args.img_size // args.patch_size)
    source_modes = head_source_modes_for_version(args.projector_version, head_variant)
    head_source_modes = {head: source_modes[head] for head in head_dims}
    head_target_grids = {
        head: head_target_grid(
            head,
            head_source_modes[head],
            grid,
            args.projector_version,
            head_variant,
        )
        for head in head_dims
    }
    head_types, head_mlp_layer_overrides = head_config_for_variant(
        head_dims,
        head_variant,
        args.head_mlp_layers,
        args.head_conv_layers,
    )
    projector = ViTSMultiHeadProjector(
        head_dims=head_dims,
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_chans=3,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=0.0,
        attn_dropout=0.0,
        projector_version=args.projector_version,
        head_source_modes=head_source_modes,
        head_target_grids=head_target_grids,
        head_types=head_types,
        head_mlp_layers=args.head_mlp_layers,
        head_mlp_layer_overrides=head_mlp_layer_overrides,
        head_hidden_dim=args.head_hidden_dim,
        head_conv_kernel_size=args.head_conv_kernel_size,
        head_dropout=0.0,
        normalize_input=not args.no_input_normalize,
        grad_checkpointing=False,
    )
    projector.load_state_dict(state_dict, strict=True)
    projector.cuda().eval().requires_grad_(False)
    return projector, head_dims


def build_dataloader(args, rank, world_size):
    transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, args.img_size)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(os.path.join(args.data_path, "train"), transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    ) if world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader, len(dataset)


def projector_head_to_features(head, tokens):
    if tokens.ndim == 2:
        return tokens.float()
    if tokens.ndim != 3:
        raise RuntimeError(f"Head {head} returned shape {tuple(tokens.shape)}")
    if head == "inception":
        if tokens.shape[1] == 1:
            return tokens[:, 0].float()
        return tokens.mean(dim=1).float()
    # Version I token-level MAE/SigLIP heads return primary + patch tokens.
    # Stats are computed from the primary token.
    return tokens[:, 0].float()


@torch.inference_mode()
def extract_stats(projector, head_dims, loader, args, rank, world_size, total_images):
    stats = {}
    for head in args.heads:
        dim = head_dims[head]
        stats[head] = {
            "sum": torch.zeros(dim, dtype=torch.float64, device="cuda"),
            "outer": torch.zeros(dim, dim, dtype=torch.float64, device="cuda"),
            "count": 0,
        }

    max_per_rank = None
    if args.num_images is not None:
        total_images = min(total_images, args.num_images)
        max_per_rank = math.ceil(total_images / world_size)

    amp_dtype = DTYPES[args.dtype]
    use_amp = amp_dtype != torch.float32
    desc = f"[rank {rank}] projector stats" if world_size > 1 else "projector stats"
    pbar = tqdm(loader, desc=desc, position=rank, disable=False)

    local_count = 0
    for images, _ in pbar:
        if max_per_rank is not None and local_count >= max_per_rank:
            break
        if max_per_rank is not None:
            images = images[: max_per_rank - local_count]
        images = images.cuda(non_blocking=True)

        with torch.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            outputs = projector(images, head_names=args.heads)

        batch_count = images.shape[0]
        for head in args.heads:
            feats = projector_head_to_features(head, outputs[head]).double()
            stats[head]["sum"].add_(feats.sum(dim=0))
            stats[head]["outer"].addmm_(feats.T, feats)
            stats[head]["count"] += batch_count

        local_count += batch_count
        pbar.set_postfix({"images": local_count})

    if world_size > 1:
        for head in args.heads:
            dist.reduce(stats[head]["sum"], dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(stats[head]["outer"], dst=0, op=dist.ReduceOp.SUM)
            count_t = torch.tensor([stats[head]["count"]], dtype=torch.long, device="cuda")
            dist.reduce(count_t, dst=0, op=dist.ReduceOp.SUM)
            stats[head]["count"] = int(count_t.item())

    if rank != 0:
        return None

    result = {}
    for head in args.heads:
        count = stats[head]["count"]
        feat_sum = stats[head]["sum"].cpu().numpy()
        feat_outer = stats[head]["outer"].cpu().numpy()
        mu = feat_sum / count
        sigma = (feat_outer - np.outer(feat_sum, feat_sum) / count) / (count - 1)
        result[head] = {"mu": mu, "sigma": sigma, "count": count}
    return result


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    rank, world_size = setup_distributed()
    if rank != 0:
        logger.setLevel(logging.WARNING)

    logger.info(
        "[projector-stats] data=%s checkpoint=%s heads=%s gpus=%d",
        args.data_path,
        args.checkpoint,
        ",".join(args.heads),
        world_size,
    )
    projector, head_dims = load_projector(args)
    logger.info("[projector-stats] head_variant=%s", args.head_variant)
    loader, total_images = build_dataloader(args, rank, world_size)
    results = extract_stats(projector, head_dims, loader, args, rank, world_size, total_images)

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        for head, item in results.items():
            output_name = default_output_name(
                head,
                args.projector_version,
                args.img_size,
                args.head_variant,
                args.backbone,
            )
            output_path = os.path.join(args.output_dir, output_name)
            np.savez(output_path, mu=item["mu"], sigma=item["sigma"], count=item["count"])
            logger.info(
                "[projector-stats] saved %s (n=%d, dim=%d)",
                output_path,
                item["count"],
                item["mu"].shape[0],
            )

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
