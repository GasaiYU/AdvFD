import argparse
import datetime
import logging
import os
import sys
import time

import torch
import torch.distributed
import torch.nn.functional as F

from utils.builders import create_generation_model, create_tokenizer
from utils.checkpoint_util import AsyncCheckpointSaver, ckpt_resume, save_checkpoint
from utils.distributed_util import all_reduce_mean, preempt_requested, register_preempt_handler
from utils.distributed_util import broadcast_module_params
from utils.eval_util import evaluate_all_emas
from utils.grad_util import get_grad_norm
from utils.logging_util import MetricLogger, SmoothedValue
from utils.optimizer_util import create_optimizer
from frechet_distance.evaluator import FDEvaluator
from frechet_distance.queue import FeatureQueue
from frechet_distance.losses import (
    compute_frechet_distance_loss,
    diff_all_gather,
    load_mu_and_sigma_reference, precompute_sigma_ref_sqrt,
)
from frechet_distance.repr_models import load_repr_model, model_short_name
from frechet_distance.judges import (
    extract_judge_features,
    resolve_per_model_args, save_fd_queue_states, load_fd_queue_states,
    fill_all_queues, run_sanity_check,
)
from frechet_distance.adversarial import (
    FeatureStatsEMA,
    build_real_whitening,
    load_fd_adv_states,
    real_whitened_frechet_distance_from_stats,
    save_fd_adv_states,
)
from utils.rng_util import RNGStateManager
from utils.schedule_util import adjust_learning_rate
from utils.setup_util import setup
from utils.vis_util import visualize
from models.patch_gan import (
    PatchDiscriminator,
    patch_discriminator_loss,
    patch_generator_loss,
)
from utils.dmd_util import DMDGuidance

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.cache_size_limit = 128
torch._dynamo.config.optimize_ddp = False

logger = logging.getLogger("FD_loss")


# ---------------------------------------------------------------------------
# FD train step
# ---------------------------------------------------------------------------

def build_real_image_batch_fn(args):
    """Return an infinite distributed ImageNet train batch supplier."""
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, DistributedSampler
    from utils.data_util import center_crop_arr

    train_dir = os.path.join(args.data_path, "train")
    if not os.path.isdir(train_dir):
        train_dir = args.data_path
    transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, args.img_size)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(train_dir, transform=transform)
    sampler = DistributedSampler(
        dataset, num_replicas=args.world_size, rank=args.rank,
        shuffle=True, drop_last=True,
    ) if args.world_size > 1 else None
    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        shuffle=(sampler is None), num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=True,
    )
    logger.info(
        f"[FD timestep] real-image batches from {train_dir}: "
        f"{len(dataset)} images, batch_size={args.batch_size}"
    )

    state = {"epoch": 0, "iterator": iter(loader)}

    def next_batch():
        try:
            images, labels = next(state["iterator"])
        except StopIteration:
            state["epoch"] += 1
            if sampler is not None:
                sampler.set_epoch(state["epoch"])
            state["iterator"] = iter(loader)
            images, labels = next(state["iterator"])
        return images.cuda(non_blocking=True), labels.cuda(non_blocking=True)

    return next_batch


def _set_requires_grad(module, requires_grad):
    for p in module.parameters():
        p.requires_grad_(requires_grad)


def _features_to_mean_cov(feats: torch.Tensor):
    n_samples = feats.shape[0]
    if n_samples < 2:
        logger.warning(f"[FD] Only {n_samples} sample(s) for whitened FD stats - need >= 2")
        fallback = torch.tensor(1e6, device=feats.device, dtype=torch.float32, requires_grad=True)
        return None, None, fallback
    mu = feats.mean(dim=0)
    feats_c = feats - mu
    sigma = (feats_c.T @ feats_c) / (n_samples - 1)
    return mu, sigma, None


def _make_mind_directions(feat_dim: int, num_projections: int, seed: int, device="cuda"):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    directions = torch.randn(
        num_projections, feat_dim,
        device=device, generator=generator, dtype=torch.float32,
    )
    return directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-12)


def _sample_mind_features(feats: torch.Tensor, sample_size: int, seed: int) -> torch.Tensor:
    if sample_size <= 0:
        raise ValueError("MIND sample size must be positive")
    if feats.shape[0] < sample_size:
        raise ValueError(f"Cannot sample {sample_size} features from bank of {feats.shape[0]}")
    if feats.shape[0] == sample_size:
        return feats
    generator = torch.Generator(device=feats.device)
    generator.manual_seed(seed)
    index = torch.randperm(feats.shape[0], device=feats.device, generator=generator)[:sample_size]
    return feats.index_select(0, index)


def _sample_mind_real_triplet(
    real_bank: torch.Tensor,
    sample_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if real_bank.shape[0] < sample_size:
        raise ValueError(
            f"Cannot sample {sample_size} MIND real features from bank of {real_bank.shape[0]}"
        )
    generator = torch.Generator(device=real_bank.device)
    generator.manual_seed(seed)
    if real_bank.shape[0] >= 3 * sample_size:
        index = torch.randperm(real_bank.shape[0], device=real_bank.device, generator=generator)
        real_a = real_bank.index_select(0, index[:sample_size])
        real_b = real_bank.index_select(0, index[sample_size:2 * sample_size])
        real_c = real_bank.index_select(0, index[2 * sample_size:3 * sample_size])
        return real_a, real_b, real_c
    real_a = _sample_mind_features(real_bank, sample_size, seed)
    real_b = _sample_mind_features(real_bank, sample_size, seed + 1)
    real_c = _sample_mind_features(real_bank, sample_size, seed + 2)
    return real_a, real_b, real_c


def compute_mind_queue_loss(
    real_feats: torch.Tensor,
    gen_feats: torch.Tensor,
    directions: torch.Tensor,
    projection_batch_size: int = 1000,
):
    """Differentiable queue-based MIND loss.

    ``real_feats`` is a detached real feature queue snapshot. ``gen_feats`` is
    the generated feature queue snapshot, with autograd only through the current
    generated queue slice.
    """
    if real_feats.dim() != 2 or gen_feats.dim() != 2:
        raise ValueError("MIND queue loss expects 2D feature tensors")
    if real_feats.shape[1] != gen_feats.shape[1]:
        raise ValueError(
            f"Feature dims differ: real={real_feats.shape[1]}, gen={gen_feats.shape[1]}"
        )
    n = min(real_feats.shape[0], gen_feats.shape[0])
    if n < 2:
        return torch.tensor(1e6, device=gen_feats.device, dtype=torch.float32)

    real = real_feats[:n].detach().to(device=gen_feats.device, dtype=torch.float32)
    gen = gen_feats[:n].to(dtype=torch.float32)
    directions = directions.to(device=gen_feats.device, dtype=torch.float32)

    alpha = 3.0 * gen.shape[1]
    total = gen.new_zeros(())
    num_proj = directions.shape[0]
    for lo in range(0, num_proj, projection_batch_size):
        dirs = directions[lo:lo + projection_batch_size]
        real_sorted = (real @ dirs.T).sort(dim=0).values
        gen_sorted = (gen @ dirs.T).sort(dim=0).values
        total = total + (real_sorted - gen_sorted).square().mean(dim=0).sum()
    return (alpha * total / num_proj).float()


@torch.no_grad()
def fill_mind_real_queues(judges, args, real_batch_fn):
    """Fill shuffled real-feature queues for queue-based MIND training."""
    if args.fd_loss_type != "mind":
        return
    if args.fd_ema_beta > 0.0 or args.fd_online_accum:
        raise ValueError("MIND queue loss needs raw feature queues; disable fd_ema_beta/fd_online_accum")
    if real_batch_fn is None:
        raise RuntimeError("MIND queue loss requires a shuffled real image dataloader")
    if args.queue_size < args.mind_sample_size:
        raise ValueError(
            f"MIND fake queue_size={args.queue_size} must be >= mind_sample_size={args.mind_sample_size}"
        )
    if args.mind_real_queue_size < args.mind_sample_size:
        raise ValueError(
            "MIND real queue must be at least as large as the per-step sample size: "
            f"{args.mind_real_queue_size} < {args.mind_sample_size}"
        )
    if args.mind_real_queue_size < 3 * args.mind_sample_size:
        logger.warning(
            "[MIND] real queue is smaller than 3x sample size; real_a/real_b/real_c may overlap"
        )

    logger.info(
        f"[MIND] Filling shuffled real feature queue: "
        f"target={args.mind_real_queue_size}, sample/step={args.mind_sample_size}, "
        f"projections/step={args.mind_num_projections}"
    )

    for judge in judges:
        judge["mind_real_queue"] = FeatureQueue(
            size=args.mind_real_queue_size,
            feat_dim=judge["feat_dim"],
            online_accum=False,
            ema_beta=0.0,
        ).cuda()

    filled = 0
    while filled < args.mind_real_queue_size:
        images, _ = real_batch_fn()
        for judge in judges:
            feats = diff_all_gather(extract_judge_features(judge, images)).float()
            count = min(feats.shape[0], args.mind_real_queue_size - filled)
            judge["mind_real_queue"].feats[filled:filled + count].copy_(feats[:count])
        filled += count
        logger.info(f"[MIND] Real feature queue: {filled}/{args.mind_real_queue_size}")

    for judge in judges:
        judge["mind_real_queue"].ptr.zero_()


@torch.no_grad()
def refresh_mind_real_queues(judges, args, real_batch_fn):
    """Refresh the real-feature bank by enqueueing fresh real batches."""
    if args.fd_loss_type != "mind":
        return
    if args.mind_real_queue_refresh_steps <= 0:
        return
    if args.current_step <= 0 or args.current_step % args.mind_real_queue_refresh_steps != 0:
        return
    if real_batch_fn is None:
        raise RuntimeError("MIND real queue refresh requires a real image dataloader")

    refreshed = 0
    for _ in range(max(1, args.mind_real_queue_refresh_batches)):
        images, _ = real_batch_fn()
        for judge in judges:
            feats = diff_all_gather(extract_judge_features(judge, images)).float()
            judge["mind_real_queue"].enqueue(feats)
        refreshed += feats.shape[0]
    logger.info(
        f"[MIND] Refreshed real feature queue: enqueued={refreshed}, "
        f"step={args.current_step}, interval={args.mind_real_queue_refresh_steps}"
    )


def _freeze_batchnorm(module):
    frozen_params = 0
    for submodule in module.modules():
        if isinstance(submodule, torch.nn.modules.batchnorm._BatchNorm):
            submodule.eval()
            for p in submodule.parameters(recurse=False):
                if p.requires_grad:
                    frozen_params += p.numel()
                p.requires_grad_(False)
    return frozen_params


def _set_fd_adv_requires_grad(module, requires_grad, freeze_batchnorm=False):
    _set_requires_grad(module, requires_grad)
    if freeze_batchnorm:
        return _freeze_batchnorm(module)
    return 0


def _all_reduce_grads(module):
    if not torch.distributed.is_initialized():
        return
    for p in module.parameters():
        if p.grad is not None:
            torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.AVG)


def _get_generator_last_layer(model):
    net = getattr(model, "net", None)
    final_layer = getattr(net, "final_layer", None)
    linear = getattr(final_layer, "linear", None)
    return getattr(linear, "weight", None)


def _extract_adv_features(judge, images):
    if judge.get("adv_backbone") == "patchgan":
        features = judge["adv_model"].forward_features(images.clamp(0.0, 1.0))
        features = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
        max_patches = judge.get("adv_max_patches", 0)
        if max_patches > 0 and features.shape[0] > max_patches:
            idx = torch.randperm(features.shape[0], device=features.device)[:max_patches]
            features = features.index_select(0, idx)
        return features
    primary, secondary = judge["adv_model"](images)
    return secondary if judge.get("pool_type") == "avg" else primary


@torch.no_grad()
def _degrade_real_images_for_fd_adv(images):
    """Lightweight real-image degradation for calibrated FD adversarial negatives."""
    x = images.detach().clamp(0.0, 1.0)
    if x.numel() == 0:
        return x

    batch_size = x.shape[0]
    dtype = x.dtype
    device = x.device
    mask_shape = (batch_size, 1, 1, 1)
    applied = torch.zeros(mask_shape, device=device, dtype=torch.bool)

    # Use per-image random masks so degraded real negatives stay diverse and
    # do not all share the same artifact stack.
    h, w = x.shape[-2:]
    if h >= 32 and w >= 32:
        down_mask = torch.rand(mask_shape, device=device) < 0.5
        down_h = max(1, h // 2)
        down_w = max(1, w // 2)
        low = F.interpolate(x.float(), size=(down_h, down_w), mode="bilinear", align_corners=False)
        low = F.interpolate(low, size=(h, w), mode="bilinear", align_corners=False).to(dtype=dtype)
        x = torch.where(down_mask, 0.5 * x + 0.5 * low, x)
        applied = applied | down_mask

    blur_mask = torch.rand(mask_shape, device=device) < 0.5
    blur = F.avg_pool2d(x.float(), kernel_size=3, stride=1, padding=1).to(dtype=dtype)
    x = torch.where(blur_mask, 0.7 * x + 0.3 * blur, x)
    applied = applied | blur_mask

    color_mask = torch.rand(mask_shape, device=device) < 0.7
    mean = x.mean(dim=(2, 3), keepdim=True)
    contrast = torch.empty(mask_shape, device=device, dtype=torch.float32).uniform_(0.75, 1.15).to(dtype)
    brightness = torch.empty(mask_shape, device=device, dtype=torch.float32).uniform_(-0.08, 0.08).to(dtype)
    jittered = (x - mean) * contrast + mean + brightness
    x = torch.where(color_mask, jittered, x)
    applied = applied | color_mask

    noise_mask = torch.rand(mask_shape, device=device) < 0.7
    noise_std = torch.empty(mask_shape, device=device, dtype=torch.float32).uniform_(0.02, 0.08).to(dtype)
    noised = x + torch.randn_like(x) * noise_std
    x = torch.where(noise_mask, noised, x)
    applied = applied | noise_mask

    fallback_mask = ~applied
    fallback = x + torch.randn_like(x) * 0.03
    x = torch.where(fallback_mask, fallback, x)
    return x.clamp(0.0, 1.0)


def _build_fd_adv_negative_images(fake_images, real_images, degraded_real_ratio):
    ratio = float(degraded_real_ratio)
    if ratio <= 0.0:
        return fake_images.detach(), 0, fake_images.shape[0]

    batch_size = fake_images.shape[0]
    real_count = min(real_images.shape[0], max(1, round(batch_size * min(ratio, 1.0))))
    fake_count = max(0, batch_size - real_count)

    pieces = []
    if fake_count > 0:
        pieces.append(fake_images[:fake_count].detach())
    if real_count > 0:
        pieces.append(_degrade_real_images_for_fd_adv(real_images[:real_count]))
    if not pieces:
        return fake_images.detach(), 0, batch_size
    return torch.cat(pieces, dim=0), real_count, fake_count


@torch.no_grad()
def init_adv_fake_stats_from_fd(judges, source: str = "FD stats"):
    """Initialize adv fake EMA from the statistics used by the base FD loss."""
    for judge in judges:
        adv_fake_stats = judge.get("adv_fake_stats")
        if adv_fake_stats is None:
            continue
        if judge.get("adv_backbone") == "patchgan":
            judge["_adv_fake_fd_stats_initialized"] = True
            logger.info(f"[FD-Adv] PatchGAN fake stats for '{judge['name']}' will initialize from patch features")
            continue
        q = judge["queue"]
        if q.ema_stats:
            mu = q.mu_ema
            m2 = q.m2_ema
        else:
            feats = q.feats.double()
            mu = feats.mean(dim=0)
            m2 = feats.T @ feats / feats.shape[0]
        adv_fake_stats.initialize_from_mean_m2(mu, m2)
        judge["_adv_fake_fd_stats_initialized"] = True
        logger.info(f"[FD-Adv] Initialized fake stats from {source} for '{judge['name']}'")


@torch.no_grad()
def _sample_fd_training_images(model, args, batch_size, y, tokenizer=None):
    input_shape = (
        getattr(model, "in_channels"),
        getattr(model, "input_size"),
        getattr(model, "input_size"),
    )
    z = torch.randn(batch_size, *input_shape, device="cuda") * args.noise_scale
    sampling_args = {
        "t_min": args.interval_min,
        "t_max": args.interval_max,
        "cfg": args.cfg,
        "num_steps": args.num_sampling_steps,
    }
    sampled = model.sample_images_with_grad(z, y, sampling_args=sampling_args)
    if tokenizer is not None:
        sampled = tokenizer.decode(tokenizer.denormalize_z(sampled))
    sampled = sampled * 0.5 + 0.5
    return sampled.clamp(0.0, 1.0)


@torch.no_grad()
def init_patch_adv_stats(judges, model, args, real_batch_fn, tokenizer=None):
    patch_judges = [
        j for j in judges
        if j.get("adv_backbone") == "patchgan"
        and j.get("adv_real_stats") is not None
        and j.get("adv_fake_stats") is not None
    ]
    target_images = args.queue_size
    if not patch_judges or target_images == 0:
        return
    if real_batch_fn is None:
        raise RuntimeError("PatchGAN adversarial FD init requires a real image dataloader")
    use_neg_stats = (
        args.fd_adv_neg_real_degrade_ratio > 0.0
        and any(j.get("adv_neg_stats") is not None for j in patch_judges)
    )

    stats = {}
    for judge in patch_judges:
        dim = judge["adv_dim"]
        device = judge["adv_real_stats"].mu_ema.device
        entry = {
            "real_sum": torch.zeros(dim, device=device, dtype=torch.float64),
            "real_m2": torch.zeros(dim, dim, device=device, dtype=torch.float64),
            "real_count": 0,
            "fake_sum": torch.zeros(dim, device=device, dtype=torch.float64),
            "fake_m2": torch.zeros(dim, dim, device=device, dtype=torch.float64),
            "fake_count": 0,
        }
        if use_neg_stats:
            entry.update({
                "neg_sum": torch.zeros(dim, device=device, dtype=torch.float64),
                "neg_m2": torch.zeros(dim, dim, device=device, dtype=torch.float64),
                "neg_count": 0,
            })
        stats[judge["name"]] = entry

    was_training = model.training
    model.eval()
    adv_was_training = {
        judge["name"]: judge["adv_model"].training
        for judge in patch_judges
    }
    for judge in patch_judges:
        judge["adv_model"].eval()
    filled = 0
    world_size = (
        torch.distributed.get_world_size()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else 1
    )
    while filled < target_images:
        remaining = target_images - filled
        target_local_bsz = min(args.fd_queue_fill_bsz, (remaining + world_size - 1) // world_size)
        real_imgs, _ = real_batch_fn()
        batch_size = min(real_imgs.shape[0], target_local_bsz)
        real_imgs = real_imgs[:batch_size]
        y = torch.randint(0, args.num_classes, (batch_size,), device="cuda")
        fake_imgs = _sample_fd_training_images(
            model,
            args,
            batch_size,
            y,
            tokenizer=tokenizer,
        )
        neg_imgs = None
        if use_neg_stats:
            neg_imgs, _, _ = _build_fd_adv_negative_images(
                fake_imgs,
                real_imgs,
                args.fd_adv_neg_real_degrade_ratio,
            )

        for judge in patch_judges:
            real_feats = diff_all_gather(_extract_adv_features(judge, real_imgs.detach()))
            fake_feats = diff_all_gather(_extract_adv_features(judge, fake_imgs.detach()))
            stat_items = [("real", real_feats), ("fake", fake_feats)]
            if use_neg_stats:
                neg_feats = diff_all_gather(_extract_adv_features(judge, neg_imgs.detach()))
                stat_items.append(("neg", neg_feats))
            entry = stats[judge["name"]]
            for prefix, feats in stat_items:
                feats_d = feats.detach().double()
                entry[f"{prefix}_sum"].add_(feats_d.sum(dim=0))
                entry[f"{prefix}_m2"].add_(feats_d.T @ feats_d)
                entry[f"{prefix}_count"] += feats_d.shape[0]

        filled += batch_size * world_size
        if filled > target_images:
            filled = target_images
        logger.info(
            f"[FD-Adv] PatchGAN stats init: {filled}/"
            f"{target_images} real/fake images"
        )

    for judge in patch_judges:
        entry = stats[judge["name"]]
        real_count = entry["real_count"]
        fake_count = entry["fake_count"]
        judge["adv_real_stats"].initialize_from_mean_m2(
            entry["real_sum"] / real_count,
            entry["real_m2"] / real_count,
        )
        judge["adv_fake_stats"].initialize_from_mean_m2(
            entry["fake_sum"] / fake_count,
            entry["fake_m2"] / fake_count,
        )
        if use_neg_stats and judge.get("adv_neg_stats") is not None:
            neg_count = entry["neg_count"]
            judge["adv_neg_stats"].initialize_from_mean_m2(
                entry["neg_sum"] / neg_count,
                entry["neg_m2"] / neg_count,
            )
            judge["_adv_neg_stats_initialized"] = True
        judge["_adv_fake_fd_stats_initialized"] = True
        neg_msg = f", neg_patches={entry['neg_count']}" if use_neg_stats else ""
        logger.info(
            f"[FD-Adv] Initialized PatchGAN stats for '{judge['name']}': "
            f"real_patches={real_count}, fake_patches={fake_count}{neg_msg}"
        )
    if was_training:
        model.train()
    for judge in patch_judges:
        if adv_was_training[judge["name"]]:
            judge["adv_model"].train()


@torch.no_grad()
def init_adv_neg_stats(judges, model, args, real_batch_fn, tokenizer=None):
    """Initialize the critic-only mixed negative EMA stats."""
    if args.fd_adv_neg_real_degrade_ratio <= 0.0:
        return

    neg_judges = [
        j for j in judges
        if j.get("adv_neg_stats") is not None
        and int(j["adv_neg_stats"].initialized.item()) == 0
    ]
    if not neg_judges:
        return

    target_images = args.queue_size
    if target_images == 0:
        return
    if real_batch_fn is None:
        raise RuntimeError("FD-Adv mixed negative stats init requires a real image dataloader")

    stats = {}
    for judge in neg_judges:
        dim = judge["adv_dim"]
        device = judge["adv_neg_stats"].mu_ema.device
        stats[judge["name"]] = {
            "neg_sum": torch.zeros(dim, device=device, dtype=torch.float64),
            "neg_m2": torch.zeros(dim, dim, device=device, dtype=torch.float64),
            "neg_count": 0,
        }

    was_training = model.training
    model.eval()
    adv_was_training = {
        judge["name"]: judge["adv_model"].training
        for judge in neg_judges
    }
    for judge in neg_judges:
        judge["adv_model"].eval()

    filled = 0
    neg_real_images = 0
    neg_fake_images = 0
    world_size = (
        torch.distributed.get_world_size()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else 1
    )
    while filled < target_images:
        remaining = target_images - filled
        target_local_bsz = min(args.fd_queue_fill_bsz, (remaining + world_size - 1) // world_size)
        real_imgs, _ = real_batch_fn()
        batch_size = min(real_imgs.shape[0], target_local_bsz)
        real_imgs = real_imgs[:batch_size]
        y = torch.randint(0, args.num_classes, (batch_size,), device="cuda")
        fake_imgs = _sample_fd_training_images(
            model,
            args,
            batch_size,
            y,
            tokenizer=tokenizer,
        )
        neg_imgs, neg_real_count, neg_fake_count = _build_fd_adv_negative_images(
            fake_imgs,
            real_imgs,
            args.fd_adv_neg_real_degrade_ratio,
        )

        for judge in neg_judges:
            neg_feats = diff_all_gather(_extract_adv_features(judge, neg_imgs.detach()))
            entry = stats[judge["name"]]
            feats_d = neg_feats.detach().double()
            entry["neg_sum"].add_(feats_d.sum(dim=0))
            entry["neg_m2"].add_(feats_d.T @ feats_d)
            entry["neg_count"] += feats_d.shape[0]

        neg_real_images += neg_real_count * world_size
        neg_fake_images += neg_fake_count * world_size
        filled += batch_size * world_size
        if filled > target_images:
            filled = target_images
        logger.info(
            f"[FD-Adv] Mixed negative stats init: {filled}/{target_images} images"
        )

    for judge in neg_judges:
        entry = stats[judge["name"]]
        neg_count = entry["neg_count"]
        judge["adv_neg_stats"].initialize_from_mean_m2(
            entry["neg_sum"] / neg_count,
            entry["neg_m2"] / neg_count,
        )
        judge["_adv_neg_stats_initialized"] = True
        denom = max(1, neg_real_images + neg_fake_images)
        logger.info(
            f"[FD-Adv] Initialized mixed negative stats for '{judge['name']}': "
            f"neg_features={neg_count}, neg_real_image_ratio={neg_real_images / denom:.4f}"
        )

    if was_training:
        model.train()
    for judge in neg_judges:
        if adv_was_training[judge["name"]]:
            judge["adv_model"].train()


def _patch_adv_judges(judges):
    return [
        j for j in judges
        if j.get("adv_backbone") == "patchgan"
        and j.get("adv_model") is not None
        and j.get("adv_optimizer") is not None
    ]


def pretrain_patch_adv_step(judges, args, real_imgs, fake_imgs):
    patch_judges = _patch_adv_judges(judges)
    if not patch_judges:
        return None

    local_step = args.current_step - args.fd_adv_patch_pretrain_start_step + 1
    last_loss = None
    last_real = None
    last_fake = None
    for judge in patch_judges:
        adv_model = judge["adv_model"]
        adv_optimizer = judge["adv_optimizer"]
        adv_model.train()
        _set_fd_adv_requires_grad(
            adv_model,
            True,
            freeze_batchnorm=args.fd_adv_freeze_batchnorm,
        )
        old_lrs = [group["lr"] for group in adv_optimizer.param_groups]
        if args.fd_adv_patch_pretrain_lr > 0:
            for group in adv_optimizer.param_groups:
                group["lr"] = args.fd_adv_patch_pretrain_lr

        adv_optimizer.zero_grad(set_to_none=True)
        d_loss, d_real, d_fake = patch_discriminator_loss(
            adv_model,
            real_imgs,
            fake_imgs.detach(),
        )
        d_loss.backward()
        _all_reduce_grads(adv_model)
        adv_optimizer.step()
        adv_optimizer.zero_grad(set_to_none=True)

        for group, lr in zip(adv_optimizer.param_groups, old_lrs):
            group["lr"] = lr
        judge["_adv_patch_pretrained"] = True
        last_loss = d_loss.detach()
        last_real = d_real.detach().mean()
        last_fake = d_fake.detach().mean()

    log_freq = args.fd_adv_patch_pretrain_log_freq
    if (
        log_freq > 0
        and last_loss is not None
        and (
            local_step == 1
            or local_step == args.fd_adv_patch_pretrain_steps
            or local_step % log_freq == 0
        )
    ):
        logger.info(
            f"[FD-Adv] PatchGAN pretrain "
            f"{local_step}/{args.fd_adv_patch_pretrain_steps}: "
            f"d_loss={float(last_loss):.4f}, "
            f"d_real={float(last_real):.4f}, "
            f"d_fake={float(last_fake):.4f}"
        )

    return last_loss, last_real, last_fake


def _adaptive_adversarial_weight(main_loss, adv_loss, last_layer, max_weight):
    if last_layer is None:
        return adv_loss.new_tensor(1.0)
    main_grads = torch.autograd.grad(main_loss, last_layer, retain_graph=True)[0]
    adv_grads = torch.autograd.grad(adv_loss, last_layer, retain_graph=True)[0]
    weight = main_grads.norm() / (adv_grads.norm() + 1e-4)
    return weight.clamp(0.0, max_weight).detach()


def _load_checkpoint_model_state_for_current_arch(args, checkpoint_path):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Could not find checkpoint at {checkpoint_path}")

    import models

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

    if args.model in models.iMFDenoiser_models:
        from models.denoiser_imf import convert_imf_checkpoint
        logger.info("[DMD] Converting official iMF teacher checkpoint keys")
        state_dict = convert_imf_checkpoint(state_dict)

    if args.model in models.pMFDenoiser_models:
        from models.denoiser_pmf import convert_pmf_checkpoint
        logger.info("[DMD] Converting official pMF teacher checkpoint keys")
        state_dict = convert_pmf_checkpoint(state_dict)

    return state_dict


def get_fd_train_step(
    model_wo_ddp,
    judges,
    sampling_args,
    args,
    tokenizer=None,
    real_batch_fn=None,
    patch_discriminator=None,
    patch_d_optimizer=None,
    dmd_guidance=None,
    dmd_optimizer=None,
):
    fid_norm_eps = args.fd_fid_norm_eps
    jit_loss_weight = args.jit_loss_weight
    batch_size = args.batch_size
    num_classes = args.num_classes
    input_shape = (args.input_channels, args.input_size, args.input_size)

    def fd_train_step():
        x0, t_view, noise, velocity_pred = None, None, None, None
        real_images_for_gan = None
        if args.fd_random_timestep_training:
            if real_batch_fn is None:
                raise RuntimeError("fd_random_timestep_training requires a real image dataloader")
            real_images_for_gan, y = real_batch_fn()
            x0 = real_images_for_gan.mul(2.0).sub(1.0)
            t_min = sampling_args["t_start_min"]
            if args.fd_timestep_logit_normal:
                t = model_wo_ddp.sample_t(batch_size, device="cuda")
                t = t_min + (1.0 - t_min) * t
            else:
                t = torch.empty(batch_size, device="cuda").uniform_(t_min, 1.0)
            noise = torch.randn_like(x0) * args.noise_scale
            t_view = t.view(batch_size, 1, 1, 1)
            z = (1.0 - t_view) * x0 + t_view * noise
            sampling_args["t_start"] = t
        else:
            z = torch.randn(batch_size, *input_shape, device="cuda") * args.noise_scale
            y = torch.randint(0, num_classes, (batch_size,), device="cuda")
            sampling_args.pop("t_start", None)
        sampling_args["return_velocity"] = jit_loss_weight > 0 and args.fd_random_timestep_training
        sampled = model_wo_ddp.sample_images_with_grad(z, y, sampling_args=sampling_args)
        if sampling_args["return_velocity"]:
            sampled, velocity_pred = sampled

        sampled_model_space = sampled

        if tokenizer is not None:
            sampled = tokenizer.decode(tokenizer.denormalize_z(sampled))
        sampled = sampled * 0.5 + 0.5  # [-1,1] -> [0,1]

        loss_dict = {}
        patch_adv_pretrain_active = (
            args.fd_adv_weight > 0
            and args.fd_adv_backbone == "patchgan"
            and args.fd_adv_patch_pretrain_steps > 0
            and args.current_step >= args.fd_adv_patch_pretrain_start_step
            and args.current_step < (
                args.fd_adv_patch_pretrain_start_step
                + args.fd_adv_patch_pretrain_steps
            )
            and args.current_step < args.fd_adv_start_step
            and any(j.get("adv_backbone") == "patchgan" for j in judges)
        )

        gan_active = (
            patch_discriminator is not None
            and args.patch_gan_weight > 0
            and args.current_step >= args.patch_gan_start_step
        )

        if gan_active:
            if real_batch_fn is None:
                raise RuntimeError("patch_gan_weight > 0 requires a real image dataloader")
            if real_images_for_gan is None:
                real_images_for_gan, _ = real_batch_fn()
            sampled_for_gan = sampled.clamp(0.0, 1.0)

            _set_requires_grad(patch_discriminator, True)
            patch_d_optimizer.zero_grad(set_to_none=True)
            d_loss, d_real, d_fake = patch_discriminator_loss(
                patch_discriminator,
                real_images_for_gan,
                sampled_for_gan.detach(),
            )
            d_loss.backward()
            _all_reduce_grads(patch_discriminator)
            patch_d_optimizer.step()
            patch_d_optimizer.zero_grad(set_to_none=True)
        else:
            d_loss, d_real, d_fake = None, None, None

        if patch_adv_pretrain_active:
            if real_batch_fn is None:
                raise RuntimeError("PatchGAN adversarial FD pretrain requires a real image dataloader")
            real_images_for_patch_adv = real_images_for_gan
            if real_images_for_patch_adv is None:
                real_images_for_patch_adv, _ = real_batch_fn()
            patch_adv_pretrain = pretrain_patch_adv_step(
                judges,
                args,
                real_images_for_patch_adv,
                sampled.clamp(0.0, 1.0),
            )
            if patch_adv_pretrain is not None:
                adv_pre_loss, adv_pre_real, adv_pre_fake = patch_adv_pretrain
                loss_dict["fd_adv_patch_pretrain_loss"] = float(adv_pre_loss)
                loss_dict["fd_adv_patch_pretrain_real"] = float(adv_pre_real)
                loss_dict["fd_adv_patch_pretrain_fake"] = float(adv_pre_fake)

        loss = torch.tensor(0.0, device="cuda")

        dmd_active = dmd_guidance is not None
        dmd_generator_turn = (
            dmd_active
            and not args.dmd_discriminator_only
            and args.dmd_guidance_weight > 0
            and args.current_step % args.dmd_generator_update_ratio == 0
        )
        dmd_fake_batch = (
            (sampled_model_space.detach(), y.detach())
            if dmd_active
            else None
        )

        all_new_feats = []
        for judge in judges:
            feats = extract_judge_features(judge, sampled)
            new_feats = diff_all_gather(feats)
            all_new_feats.append(new_feats)

        adv_active = (
            args.fd_adv_weight > 0
            and args.current_step >= args.fd_adv_start_step
            and any(j.get("adv_model") is not None for j in judges)
        )
        if args.fd_adv_weight > 0:
            if args.fd_adv_warmup_steps > 0:
                adv_warmup_progress = (
                    args.current_step - args.fd_adv_start_step
                ) / args.fd_adv_warmup_steps
                fd_adv_warmup = max(0.0, min(1.0, adv_warmup_progress))
            else:
                fd_adv_warmup = 1.0 if args.current_step >= args.fd_adv_start_step else 0.0
            fd_adv_effective_weight = args.fd_adv_weight * fd_adv_warmup
            loss_dict["fd_adv_warmup"] = float(fd_adv_warmup)
            loss_dict["fd_adv_effective_weight"] = float(fd_adv_effective_weight)
        else:
            fd_adv_effective_weight = 0.0
        real_images_for_adv = None
        if adv_active:
            patch_stats_needed = any(
                j.get("adv_backbone") == "patchgan"
                and j.get("adv_real_stats") is not None
                and int(j["adv_real_stats"].initialized.item()) == 0
                for j in judges
            )
            if patch_stats_needed:
                init_patch_adv_stats(
                    judges,
                    model_wo_ddp,
                    args,
                    real_batch_fn,
                    tokenizer=tokenizer,
                )
            init_needed = any(
                j.get("adv_fake_stats") is not None
                and j.get("adv_backbone") != "patchgan"
                and not j.get("_adv_fake_fd_stats_initialized", False)
                for j in judges
            )
            if init_needed:
                init_adv_fake_stats_from_fd(
                    judges,
                    source=f"FD stats at adv start_step={args.fd_adv_start_step}",
                )
            neg_init_needed = any(
                j.get("adv_neg_stats") is not None
                and int(j["adv_neg_stats"].initialized.item()) == 0
                for j in judges
            )
            if neg_init_needed:
                init_adv_neg_stats(
                    judges,
                    model_wo_ddp,
                    args,
                    real_batch_fn,
                    tokenizer=tokenizer,
                )
            if real_batch_fn is None:
                raise RuntimeError("fd_adv_weight > 0 requires a real image dataloader")
            real_images_for_adv = real_images_for_gan
            if real_images_for_adv is None:
                real_images_for_adv, _ = real_batch_fn()

            for judge in judges:
                adv_model = judge.get("adv_model")
                adv_optimizer = judge.get("adv_optimizer")
                if adv_model is None or adv_optimizer is None:
                    continue

                if judge.get("adv_backbone") == "patchgan":
                    adv_model.train()
                else:
                    adv_model.eval()
                _set_fd_adv_requires_grad(
                    adv_model,
                    True,
                    freeze_batchnorm=args.fd_adv_freeze_batchnorm,
                )
                last_adv_fd = None
                last_adv_raw_fd = None
                last_patch_d_loss = None
                last_patch_d_real = None
                last_patch_d_fake = None
                last_adv_grad_norm = None
                neg_real_count = 0
                neg_fake_count = 0
                for _ in range(args.fd_adv_steps):
                    adv_optimizer.zero_grad(set_to_none=True)
                    patch_hinge_update = (
                        judge.get("adv_backbone") == "patchgan"
                        and args.fd_adv_patch_train_loss == "hinge"
                    )
                    if patch_hinge_update:
                        d_loss, d_real, d_fake = patch_discriminator_loss(
                            adv_model,
                            real_images_for_adv,
                            sampled.detach().clamp(0.0, 1.0),
                        )
                        d_loss.backward()
                        last_patch_d_loss = d_loss.detach()
                        last_patch_d_real = d_real.detach().mean()
                        last_patch_d_fake = d_fake.detach().mean()
                        log_raw_adv_fd = (
                            args.fd_adv_log_raw
                            and args.fd_adv_log_raw_freq > 0
                            and args.current_step % args.fd_adv_log_raw_freq == 0
                        )
                        if log_raw_adv_fd:
                            with torch.no_grad():
                                real_adv = diff_all_gather(
                                    _extract_adv_features(judge, real_images_for_adv.detach())
                                )
                                fake_adv = diff_all_gather(
                                    _extract_adv_features(judge, sampled.detach())
                                )
                                real_mu, real_cov = judge["adv_real_stats"].build_stats(real_adv)
                                fake_mu, fake_cov = judge["adv_fake_stats"].build_stats(fake_adv)
                                adv_fd = compute_frechet_distance_loss(
                                    real_mu.detach(),
                                    real_cov.detach(),
                                    mu=fake_mu.detach(),
                                    sigma=fake_cov.detach(),
                                )
                                adv_raw_fd = adv_fd.detach()
                                last_adv_fd = adv_fd.detach()
                                last_adv_raw_fd = adv_raw_fd.detach()
                        else:
                            adv_fd = None
                            adv_raw_fd = None
                    else:
                        neg_images, neg_real_count, neg_fake_count = _build_fd_adv_negative_images(
                            sampled.detach(),
                            real_images_for_adv,
                            args.fd_adv_neg_real_degrade_ratio,
                        )
                        real_adv = diff_all_gather(
                            _extract_adv_features(judge, real_images_for_adv.detach())
                        )
                        neg_adv = diff_all_gather(
                            _extract_adv_features(judge, neg_images)
                        )
                        real_mu, real_cov = judge["adv_real_stats"].build_stats(real_adv)
                        fake_stats = judge.get("adv_neg_stats") or judge["adv_fake_stats"]
                        fake_mu, fake_cov = fake_stats.build_stats(neg_adv)
                        if args.fd_adv_no_whiten:
                            adv_fd = compute_frechet_distance_loss(
                                real_mu,
                                real_cov,
                                mu=fake_mu,
                                sigma=fake_cov,
                            )
                        else:
                            real_whitening = build_real_whitening(
                                real_mu,
                                real_cov,
                                eps=args.fd_adv_whiten_eps,
                            )
                            adv_fd = real_whitened_frechet_distance_from_stats(
                                real_mu, real_cov, fake_mu, fake_cov,
                                eps=args.fd_adv_whiten_eps,
                                real_whitening=real_whitening,
                            )
                        log_raw_adv_fd = (
                            args.fd_adv_log_raw
                            and args.fd_adv_log_raw_freq > 0
                            and args.current_step % args.fd_adv_log_raw_freq == 0
                        )
                        if log_raw_adv_fd and args.fd_adv_no_whiten:
                            adv_raw_fd = adv_fd.detach()
                        elif log_raw_adv_fd:
                            with torch.no_grad():
                                adv_raw_fd = compute_frechet_distance_loss(
                                    real_mu.detach(),
                                    real_cov.detach(),
                                    mu=fake_mu.detach(),
                                    sigma=fake_cov.detach(),
                                )
                        else:
                            adv_raw_fd = None
                        if judge.get("adv_neg_stats") is not None:
                            judge["_adv_neg_update"] = neg_adv.detach()
                        (-adv_fd).backward()
                    _all_reduce_grads(adv_model)
                    if args.fd_adv_grad_clip > 0.0:
                        adv_grad_norm = torch.nn.utils.clip_grad_norm_(
                            adv_model.parameters(),
                            args.fd_adv_grad_clip,
                        )
                        last_adv_grad_norm = adv_grad_norm.detach()
                    adv_optimizer.step()
                    if adv_fd is not None:
                        last_adv_fd = adv_fd.detach()
                    if adv_raw_fd is not None:
                        last_adv_raw_fd = adv_raw_fd.detach()
                adv_optimizer.zero_grad(set_to_none=True)
                if last_adv_fd is not None:
                    loss_dict[f"fd_adv_critic_{judge['name']}"] = float(last_adv_fd)
                if last_adv_raw_fd is not None:
                    loss_dict[f"fd_adv_critic_raw_{judge['name']}"] = float(last_adv_raw_fd)
                if last_adv_grad_norm is not None:
                    loss_dict[f"fd_adv_critic_grad_norm_{judge['name']}"] = float(last_adv_grad_norm)
                if args.fd_adv_neg_real_degrade_ratio > 0 and last_adv_fd is not None:
                    denom = max(1, neg_real_count + neg_fake_count)
                    loss_dict[f"fd_adv_critic_neg_real_ratio_{judge['name']}"] = float(neg_real_count / denom)
                if last_patch_d_loss is not None:
                    loss_dict[f"fd_adv_patch_d_loss_{judge['name']}"] = float(last_patch_d_loss)
                    loss_dict[f"fd_adv_patch_d_real_{judge['name']}"] = float(last_patch_d_real)
                    loss_dict[f"fd_adv_patch_d_fake_{judge['name']}"] = float(last_patch_d_fake)

        for i, judge in enumerate(judges):
            new_feats = all_new_feats[i]

            _ns_kwargs = dict(sigma_ref_sqrt=judge.get("sigma_ref_sqrt"))
            normalize_main_loss = True
            if args.fd_loss_type == "mind":
                if judge["queue"].online_accum or judge["queue"].ema_stats:
                    raise RuntimeError("MIND queue loss needs snapshot queues, not EMA/online accumulators")
                all_feats = judge["queue"].build_feats_snapshot(new_feats)
                fake_feats = _sample_mind_features(
                    all_feats,
                    args.mind_sample_size,
                    args.mind_projection_seed + 1000003 + args.current_step * len(judges) + i,
                )
                real_a, real_b, real_c = _sample_mind_real_triplet(
                    judge["mind_real_queue"].feats.detach(),
                    args.mind_sample_size,
                    args.mind_projection_seed + 2000003 + args.current_step * len(judges) + i,
                )
                mind_directions = _make_mind_directions(
                    judge["feat_dim"],
                    args.mind_num_projections,
                    args.mind_projection_seed + args.current_step * len(judges) + i,
                    device=new_feats.device,
                )
                fake_real_mind = compute_mind_queue_loss(
                    real_a,
                    fake_feats,
                    mind_directions,
                    projection_batch_size=args.mind_projection_batch_size,
                )
                with torch.no_grad():
                    real_real_mind = compute_mind_queue_loss(
                        real_b,
                        real_c,
                        mind_directions,
                        projection_batch_size=args.mind_projection_batch_size,
                    )
                fid = fake_real_mind
                fid_loss = F.relu(fake_real_mind - args.mind_real_baseline_gamma * real_real_mind)
                normalize_main_loss = False
                loss_key = f"mind_{judge['name']}"
                loss_dict[f"mind_real_real_{judge['name']}"] = float(real_real_mind.detach())
                loss_dict[f"mind_margin_{judge['name']}"] = float(fid_loss.detach())
            elif judge["queue"].online_accum or judge["queue"].ema_stats:
                mu, sigma = judge["queue"].build_feats_stats(new_feats)
                if args.fd_whiten:
                    real_whitening = build_real_whitening(
                        judge["mu_ref"],
                        judge["sigma_ref"],
                        eps=args.fd_whiten_eps,
                    )
                    fid = real_whitened_frechet_distance_from_stats(
                        judge["mu_ref"],
                        judge["sigma_ref"],
                        mu,
                        sigma,
                        eps=args.fd_whiten_eps,
                        real_whitening=real_whitening,
                    )
                else:
                    fid = compute_frechet_distance_loss(judge["mu_ref"], judge["sigma_ref"],
                                                        mu=mu, sigma=sigma,
                                                        **_ns_kwargs)
                loss_key = f"fid_{judge['name']}"
            else:
                all_feats = judge["queue"].build_feats_snapshot(new_feats)
                if args.fd_whiten:
                    mu, sigma, fallback = _features_to_mean_cov(all_feats)
                    if fallback is not None:
                        fid = fallback
                    else:
                        real_whitening = build_real_whitening(
                            judge["mu_ref"],
                            judge["sigma_ref"],
                            eps=args.fd_whiten_eps,
                        )
                        fid = real_whitened_frechet_distance_from_stats(
                            judge["mu_ref"],
                            judge["sigma_ref"],
                            mu,
                            sigma,
                            eps=args.fd_whiten_eps,
                            real_whitening=real_whitening,
                        )
                else:
                    fid = compute_frechet_distance_loss(judge["mu_ref"], judge["sigma_ref"],
                                                        all_feats=all_feats,
                                                        **_ns_kwargs)
                loss_key = f"fid_{judge['name']}"
            if normalize_main_loss:
                fid_loss = fid / (fid.detach() + fid_norm_eps)
            loss = loss + judge["weight"] * fid_loss
            loss_dict[loss_key] = float(fid.detach())

            adv_model = judge.get("adv_model")
            if adv_active and adv_model is not None:
                _set_requires_grad(adv_model, False)
                adv_model.eval()
                with torch.no_grad():
                    real_adv = diff_all_gather(
                        _extract_adv_features(judge, real_images_for_adv.detach())
                    ).detach()
                fake_adv = diff_all_gather(
                    _extract_adv_features(judge, sampled)
                )
                real_mu, real_cov = judge["adv_real_stats"].build_stats(real_adv)
                fake_mu, fake_cov = judge["adv_fake_stats"].build_stats(fake_adv)
                if args.fd_adv_no_whiten:
                    adv_fd = compute_frechet_distance_loss(
                        real_mu,
                        real_cov,
                        mu=fake_mu,
                        sigma=fake_cov,
                    )
                else:
                    real_whitening = build_real_whitening(
                        real_mu,
                        real_cov,
                        eps=args.fd_adv_whiten_eps,
                    )
                    adv_fd = real_whitened_frechet_distance_from_stats(
                        real_mu, real_cov, fake_mu, fake_cov,
                        eps=args.fd_adv_whiten_eps,
                        real_whitening=real_whitening,
                    )
                log_raw_adv_fd = (
                    args.fd_adv_log_raw
                    and args.fd_adv_log_raw_freq > 0
                    and args.current_step % args.fd_adv_log_raw_freq == 0
                )
                if log_raw_adv_fd and args.fd_adv_no_whiten:
                    adv_raw_fd = adv_fd.detach()
                elif log_raw_adv_fd:
                    with torch.no_grad():
                        adv_raw_fd = compute_frechet_distance_loss(
                            real_mu.detach(),
                            real_cov.detach(),
                            mu=fake_mu.detach(),
                            sigma=fake_cov.detach(),
                        )
                else:
                    adv_raw_fd = None
                judge["_adv_real_update"] = real_adv.detach()
                judge["_adv_fake_update"] = fake_adv.detach()
                adv_fd_loss = adv_fd / (adv_fd.detach() + fid_norm_eps)
                loss = loss + fd_adv_effective_weight * judge["weight"] * adv_fd_loss
                loss_dict[f"fd_adv_{judge['name']}"] = float(adv_fd.detach())
                if adv_raw_fd is not None:
                    loss_dict[f"fd_adv_raw_{judge['name']}"] = float(adv_raw_fd.detach())

        fd_main_loss = loss

        if dmd_generator_turn:
            dmd_loss, dmd_dict = dmd_guidance.compute_distribution_matching_loss(
                sampled_model_space,
                y,
            )
            dmd_effective_loss = args.dmd_guidance_weight * dmd_loss
            loss = loss + dmd_effective_loss
            loss_dict.update(dmd_dict)
            loss_dict["dmd_effective_loss"] = float(dmd_effective_loss.detach())

        if gan_active:
            _set_requires_grad(patch_discriminator, False)
            patch_g_loss = patch_generator_loss(patch_discriminator, sampled_for_gan)
            if args.patch_gan_adaptive_weight:
                patch_g_adaptive_weight = _adaptive_adversarial_weight(
                    fd_main_loss,
                    patch_g_loss,
                    _get_generator_last_layer(model_wo_ddp),
                    args.patch_gan_max_adaptive_weight,
                )
            else:
                patch_g_adaptive_weight = patch_g_loss.new_tensor(1.0)
            if args.patch_gan_warmup_steps > 0:
                warmup_progress = (
                    args.current_step - args.patch_gan_start_step + 1
                ) / args.patch_gan_warmup_steps
                patch_g_warmup = max(0.0, min(1.0, warmup_progress))
            else:
                patch_g_warmup = 1.0
            patch_g_effective_weight = (
                args.patch_gan_weight * patch_g_warmup * patch_g_adaptive_weight
            )
            patch_g_effective_loss = patch_g_effective_weight * patch_g_loss
            loss = loss + patch_g_effective_loss
            loss_dict["patch_g_loss"] = float(patch_g_loss.detach())
            loss_dict["patch_g_adaptive_weight"] = float(patch_g_adaptive_weight.detach())
            loss_dict["patch_g_warmup"] = float(patch_g_warmup)
            loss_dict["patch_g_effective_weight"] = float(patch_g_effective_weight.detach())
            loss_dict["patch_g_effective_loss"] = float(patch_g_effective_loss.detach())
            loss_dict["patch_d_loss"] = float(d_loss.detach())
            loss_dict["patch_d_real"] = float(d_real.detach().mean())
            loss_dict["patch_d_fake"] = float(d_fake.detach().mean())
            _set_requires_grad(patch_discriminator, True)

        if jit_loss_weight > 0:
            if not args.fd_random_timestep_training or velocity_pred is None:
                raise RuntimeError("jit_loss_weight > 0 requires fd_random_timestep_training")
            velocity_target = t_view * (noise - x0) / t_view.clamp_min(model_wo_ddp.t_eps)
            jit_loss = ((velocity_pred - velocity_target) ** 2).mean()
            loss = loss + jit_loss_weight * jit_loss
            loss_dict["jit_loss"] = float(jit_loss.detach())

        loss.backward(create_graph=False)

        if torch.distributed.is_initialized():
            _all_reduce_grads(model_wo_ddp)

        for i, judge in enumerate(judges):
            judge["queue"].enqueue(all_new_feats[i].detach())
            if adv_active:
                real_update = judge.pop("_adv_real_update", None)
                fake_update = judge.pop("_adv_fake_update", None)
                neg_update = judge.pop("_adv_neg_update", None)
                if real_update is not None:
                    judge["adv_real_stats"].update(real_update)
                if fake_update is not None:
                    judge["adv_fake_stats"].update(fake_update)
                if neg_update is not None:
                    judge["adv_neg_stats"].update(neg_update)

        refresh_mind_real_queues(judges, args, real_batch_fn)

        return loss, loss_dict, dmd_fake_batch

    if args.compile:
        from utils.runtime_util import _warmup
        logger.info("[Compilation] Compiling fd_train_step ...")
        t0 = time.perf_counter()
        fd_train_step = torch.compile(fd_train_step)
        _warmup(lambda: fd_train_step(), n=2)
        logger.info(f"[Compilation] fd_train_step compiled in {time.perf_counter() - t0:.2f}s")

    return fd_train_step


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_and_evaluate(args):
    wandb_logger = setup(args)
    register_preempt_handler()
    if args.fd_adv_grad_clip < 0.0:
        raise ValueError("fd_adv_grad_clip must be >= 0")
    if not 0.0 <= args.fd_adv_neg_real_degrade_ratio <= 1.0:
        raise ValueError("fd_adv_neg_real_degrade_ratio must be in [0, 1]")

    # -- models, optimizer, checkpoint --
    tokenizer = create_tokenizer(args)
    model, ema_model = create_generation_model(args)
    model_wo_ddp = model

    patch_discriminator = None
    patch_d_optimizer = None
    if args.patch_gan_weight > 0:
        patch_discriminator = PatchDiscriminator(
            in_channels=3,
            base_channels=args.patch_gan_channels,
            num_layers=args.patch_gan_layers,
        ).cuda()
        broadcast_module_params(patch_discriminator, src=0)
        patch_d_optimizer = torch.optim.AdamW(
            patch_discriminator.parameters(),
            lr=args.patch_gan_lr,
            betas=(0.5, 0.9),
            weight_decay=args.patch_gan_weight_decay,
        )
        logger.info(
            f"[PatchGAN] enabled: weight={args.patch_gan_weight}, "
            f"lr={args.patch_gan_lr}, channels={args.patch_gan_channels}, "
            f"layers={args.patch_gan_layers}, start_step={args.patch_gan_start_step}"
        )

    dmd_enabled = args.dmd_guidance_weight > 0 or args.dmd_discriminator_only

    extra_keys = ["fd_queue_states"]
    if patch_discriminator is not None:
        extra_keys.append("patch_gan_state")
    if dmd_enabled:
        extra_keys.append("dmd_guidance_state")
    if args.fd_adv_weight > 0:
        extra_keys.append("fd_adv_states")

    optimizer = create_optimizer(args, model_wo_ddp, print_trainable_params=True)
    extra = ckpt_resume(args, model_wo_ddp, optimizer, ema_model,
                        extra_keys=extra_keys)

    if patch_discriminator is not None and extra is not None and "patch_gan_state" in extra:
        gan_state = extra["patch_gan_state"]
        patch_discriminator.load_state_dict(gan_state["discriminator"])
        if "optimizer" in gan_state:
            patch_d_optimizer.load_state_dict(gan_state["optimizer"])
        logger.info("[PatchGAN] Restored discriminator and optimizer state")

    dmd_guidance = None
    dmd_optimizer = None
    if dmd_enabled:
        if args.dmd_fake_update_ratio is not None:
            logger.warning(
                "[DMD] --dmd_fake_update_ratio is deprecated; interpreting it as "
                "--dmd_generator_update_ratio to match DMD2 terminology."
            )
            args.dmd_generator_update_ratio = args.dmd_fake_update_ratio
        if args.dmd_generator_update_ratio < 1:
            raise ValueError("dmd_generator_update_ratio must be >= 1")
        if args.dmd_decoupled and args.dmd_fake_loss_type == "asd":
            raise ValueError("ASD fake loss is only supported for original DMD; do not combine it with --dmd_decoupled")
        if not (0.0 < args.dmd_min_t <= args.dmd_max_t <= 1.0):
            raise ValueError("DMD timesteps must satisfy 0 < dmd_min_t <= dmd_max_t <= 1")
        if not (0.0 <= args.dmd_fake_min_t <= args.dmd_fake_max_t <= 1.0):
            raise ValueError("DMD fake-score timesteps must satisfy 0 <= dmd_fake_min_t <= dmd_fake_max_t <= 1")
        if args.dmd_decoupled:
            if args.dmd_dm_grad_mode != "original":
                raise ValueError("--dmd_dm_grad_mode is only supported for non-decoupled DMD")
            if args.dmd_ca_min_t is None:
                args.dmd_ca_min_t = args.dmd_min_t
            if args.dmd_ca_max_t is None:
                args.dmd_ca_max_t = args.dmd_max_t
            if args.dmd_dm_min_t is None:
                args.dmd_dm_min_t = args.dmd_min_t
            if args.dmd_dm_max_t is None:
                args.dmd_dm_max_t = args.dmd_max_t
            if not (0.0 < args.dmd_ca_min_t <= args.dmd_ca_max_t <= 1.0):
                raise ValueError("DMD CA timesteps must satisfy 0 < dmd_ca_min_t <= dmd_ca_max_t <= 1")
            if not (0.0 < args.dmd_dm_min_t <= args.dmd_dm_max_t <= 1.0):
                raise ValueError("DMD DM timesteps must satisfy 0 < dmd_dm_min_t <= dmd_dm_max_t <= 1")
            if args.dmd_ca_weight == 0.0 and args.dmd_dm_weight == 0.0:
                raise ValueError("At least one of dmd_ca_weight or dmd_dm_weight must be nonzero")
        if args.dmd_real_guidance_scale is None:
            args.dmd_real_guidance_scale = args.cfg
        if args.dmd_fake_guidance_scale != 1.0:
            raise ValueError("DMD fake guidance scale should stay 1.0 to match DMD2")
        if args.num_sampling_steps != 1:
            logger.warning(
                "[DMD] num_sampling_steps=%s; DMD2's ImageNet setup trains a 1-step generator. "
                "Set --num_sampling_steps 1 for the intended 1-step mode.",
                args.num_sampling_steps,
            )
        dmd_guidance = DMDGuidance(model_wo_ddp, args).cuda()
        teacher_path = args.dmd_teacher_load_from
        if teacher_path is None and args.resume_from and args.load_from:
            teacher_path = args.load_from
        if teacher_path is not None:
            teacher_state = _load_checkpoint_model_state_for_current_arch(args, teacher_path)
            dmd_guidance.load_real_score_state_dict(
                teacher_state,
                strict=args.dmd_teacher_strict_load,
            )
            logger.info(f"[DMD] fixed real score teacher loaded from {teacher_path}")
        else:
            logger.info("[DMD] fixed real score teacher initialized from current model weights")

        broadcast_module_params(dmd_guidance.real_score_model, src=0)
        broadcast_module_params(dmd_guidance.fake_score_model, src=0)
        dmd_optimizer = torch.optim.AdamW(
            dmd_guidance.fake_score_model.parameters(),
            lr=args.dmd_guidance_lr,
            betas=(0.9, 0.999),
            weight_decay=args.dmd_guidance_weight_decay,
        )
        if extra is not None and "dmd_guidance_state" in extra:
            dmd_state = extra["dmd_guidance_state"]
            dmd_guidance.load_state_dict_from_checkpoint(dmd_state)
            if "optimizer" in dmd_state:
                dmd_optimizer.load_state_dict(dmd_state["optimizer"])
            logger.info("[DMD] Restored guidance state")
        logger.info(
            f"[DMD] enabled: weight={args.dmd_guidance_weight}, "
            f"discriminator_only={args.dmd_discriminator_only}, "
            f"decoupled={args.dmd_decoupled}, "
            f"dm_grad_mode={args.dmd_dm_grad_mode}, "
            f"fake_loss_type={args.dmd_fake_loss_type}, "
            f"fake_prediction_type={args.dmd_fake_prediction_type}, "
            f"asd_gamma={args.dmd_asd_gamma}, "
            f"guidance_lr={args.dmd_guidance_lr}, dm_t=[{args.dmd_min_t}, {args.dmd_max_t}], "
            f"ca_t=[{args.dmd_ca_min_t}, {args.dmd_ca_max_t}], "
            f"decoupled_dm_t=[{args.dmd_dm_min_t}, {args.dmd_dm_max_t}], "
            f"ca_weight={args.dmd_ca_weight}, dm_weight={args.dmd_dm_weight}, "
            f"fake_t=[{args.dmd_fake_min_t}, {args.dmd_fake_max_t}], "
            f"generator_update_ratio={args.dmd_generator_update_ratio}, "
            f"real_guidance_scale={args.dmd_real_guidance_scale}, "
            f"fake_guidance_scale={args.dmd_fake_guidance_scale}"
        )

    rng = RNGStateManager()
    rng.save()
    if (not args.disable_vis) or args.vis_only:
        visualize(args, model_wo_ddp, ema_model, args.current_step, rng=rng, tokenizer=tokenizer)
        if args.vis_only:
            return 0

    # -- frechet distance evaluator --
    repr_model_eval, feat_dim_eval, _, _ = load_repr_model("inception")
    fid_evaluator = FDEvaluator(repr_model_eval, feat_dim_eval, args.fid_stats_path)

    # -- frechet distance system: repr models, queues --
    resolve_per_model_args(args)

    judges = []
    for name, stats_path, weight, pool_type, ts in zip(
        args.fd_repr_models, args.fd_repr_stats_paths,
        args.fd_repr_weights, args.fd_repr_pool_types, args.fd_target_sizes,
    ):
        repr_model, feat_dim, _, _ = load_repr_model(
            name,
            target_size=ts,
            grad_checkpointing=args.fd_repr_grad_checkpointing,
        )
        mu_ref, sigma_ref = load_mu_and_sigma_reference(stats_path, pool_type=pool_type)
        queue = FeatureQueue(size=args.queue_size, feat_dim=feat_dim,
                             online_accum=args.fd_online_accum,
                             ema_beta=args.fd_ema_beta).cuda()
        short = model_short_name(name)
        sigma_ref_sqrt = None
        if args.fd_eigvalsh:
            sigma_ref_sqrt = precompute_sigma_ref_sqrt(sigma_ref)
        judge = {
            "name": short, "model": repr_model,
            "feat_dim": feat_dim,
            "pool_type": pool_type,
            "mu_ref": mu_ref, "sigma_ref": sigma_ref,
            "sigma_ref_sqrt": sigma_ref_sqrt,
            "queue": queue, "weight": weight,
        }
        if args.fd_adv_weight > 0:
            if args.fd_adv_backbone == "patchgan":
                adv_model = PatchDiscriminator(
                    in_channels=3,
                    base_channels=args.fd_adv_patch_channels,
                    max_channels=args.fd_adv_patch_max_channels,
                    num_layers=args.fd_adv_patch_layers,
                    use_batchnorm=args.fd_adv_patch_batchnorm,
                ).cuda()
                adv_feat_dim = adv_model.feature_dim
                adv_backbone_label = "patchgan"
            else:
                adv_model, adv_feat_dim, _, _ = load_repr_model(
                    name,
                    target_size=ts,
                    grad_checkpointing=args.fd_repr_grad_checkpointing,
                )
                adv_backbone_label = "repr"
            adv_model.requires_grad_(True).eval()
            frozen_bn_params = 0
            if args.fd_adv_freeze_batchnorm:
                frozen_bn_params = _freeze_batchnorm(adv_model)
            broadcast_module_params(adv_model, src=0)
            adv_trainable_params = [p for p in adv_model.parameters() if p.requires_grad]
            if not adv_trainable_params:
                raise ValueError("FD-Adv model has no trainable parameters after freezing BatchNorm")
            adv_optimizer = torch.optim.AdamW(
                adv_trainable_params,
                lr=args.fd_adv_lr,
                betas=(args.fd_adv_beta1, args.fd_adv_beta2),
                weight_decay=args.fd_adv_weight_decay,
            )
            judge["adv_model"] = adv_model
            judge["adv_optimizer"] = adv_optimizer
            judge["adv_dim"] = adv_feat_dim
            judge["adv_backbone"] = adv_backbone_label
            judge["adv_max_patches"] = args.fd_adv_patch_max_patches
            judge["adv_real_stats"] = FeatureStatsEMA(
                adv_feat_dim,
                beta=args.fd_adv_ema_beta,
            ).cuda()
            if args.fd_adv_backbone == "repr":
                judge["adv_real_stats"].initialize_from_mean_cov(mu_ref, sigma_ref)
            judge["adv_fake_stats"] = FeatureStatsEMA(
                adv_feat_dim,
                beta=args.fd_adv_ema_beta,
            ).cuda()
            if args.fd_adv_neg_real_degrade_ratio > 0.0:
                judge["adv_neg_stats"] = FeatureStatsEMA(
                    adv_feat_dim,
                    beta=args.fd_adv_ema_beta,
                ).cuda()
            judge["_adv_fake_fd_stats_initialized"] = False
            if judge.get("adv_neg_stats") is not None:
                judge["_adv_neg_stats_initialized"] = False
        judges.append(judge)
        eig_mode = "eigvalsh" if args.fd_eigvalsh else "eigvals"
        stats_mode = f"ema(beta={args.fd_ema_beta})" if args.fd_ema_beta > 0 else ("online_accum" if args.fd_online_accum else "snapshot")
        logger.info(f"[FD] Repr '{short}' ({name}): feat_dim={feat_dim}, "
                     f"weight={weight}, pool={pool_type}, stats={stats_path}, "
                     f"eig_mode={eig_mode}, stats_mode={stats_mode}, "
                     f"whiten={args.fd_whiten}, whiten_eps={args.fd_whiten_eps}")
        if args.fd_adv_weight > 0:
            logger.info(
                f"[FD-Adv] Repr '{short}': backbone={judge['adv_backbone']}, "
                f"dim={judge['adv_dim']}, "
                f"weight={args.fd_adv_weight}, lr={args.fd_adv_lr}, "
                f"steps={args.fd_adv_steps}, grad_clip={args.fd_adv_grad_clip}, "
                f"whiten_eps={args.fd_adv_whiten_eps}, "
                f"whiten={not args.fd_adv_no_whiten}, "
                f"neg_real_degrade_ratio={args.fd_adv_neg_real_degrade_ratio}, "
                f"freeze_batchnorm={args.fd_adv_freeze_batchnorm}, "
                f"frozen_bn_params={frozen_bn_params}, "
                f"ema_beta={args.fd_adv_ema_beta}, "
                f"start_step={args.fd_adv_start_step}, "
                f"warmup_steps={args.fd_adv_warmup_steps}, "
                f"max_patches={judge.get('adv_max_patches', 0)}, "
                f"patch_train_loss={args.fd_adv_patch_train_loss}, "
                f"patch_pretrain_start_step={args.fd_adv_patch_pretrain_start_step}, "
                f"patch_pretrain_steps={args.fd_adv_patch_pretrain_steps}, "
                f"patch_pretrain_lr={args.fd_adv_patch_pretrain_lr}"
            )

    adv_restored = False
    if extra is not None and "fd_adv_states" in extra:
        adv_restored = load_fd_adv_states(judges, extra["fd_adv_states"])

    real_batch_fn = (
        build_real_image_batch_fn(args)
        if (
            args.fd_loss_type == "mind"
            or args.fd_random_timestep_training
            or args.jit_loss_weight > 0
            or args.patch_gan_weight > 0
            or args.fd_adv_weight > 0
        )
        else None
    )

    fd_restored = (extra is not None
                   and "fd_queue_states" in extra
                   and load_fd_queue_states(judges, extra["fd_queue_states"]))
    if fd_restored:
        logger.info("[FD] Restored all queue states from checkpoint — skipping queue fill")
        run_sanity_check(judges, args.queue_size, args=args)
    else:
        logger.info(f"[FD] Filling {len(judges)} feature queue(s) "
                    f"({args.queue_size} entries each) ...")
        fill_all_queues(judges, model_wo_ddp, args, tokenizer=tokenizer)
        run_sanity_check(judges, args.queue_size, args=args)
    fill_mind_real_queues(judges, args, real_batch_fn)
    if args.fd_adv_weight > 0:
        if adv_restored:
            for judge in judges:
                judge["_adv_fake_fd_stats_initialized"] = True
                neg_stats = judge.get("adv_neg_stats")
                if neg_stats is not None:
                    judge["_adv_neg_stats_initialized"] = int(neg_stats.initialized.item()) != 0
        elif args.current_step >= args.fd_adv_start_step:
            if args.fd_adv_backbone == "patchgan":
                init_patch_adv_stats(judges, model_wo_ddp, args, real_batch_fn, tokenizer=tokenizer)
            else:
                init_adv_fake_stats_from_fd(judges, source="current FD stats")
            init_adv_neg_stats(judges, model_wo_ddp, args, real_batch_fn, tokenizer=tokenizer)
        else:
            for judge in judges:
                judge["_adv_fake_fd_stats_initialized"] = False
                if judge.get("adv_neg_stats") is not None:
                    judge["_adv_neg_stats_initialized"] = False
            logger.info(
                f"[FD-Adv] Will initialize adversarial stats "
                f"at start_step={args.fd_adv_start_step}"
            )
    del extra

    torch.distributed.barrier()

    model.train()
    args.input_channels = model_wo_ddp.in_channels
    args.input_size = model_wo_ddp.input_size

    # -- FD train step closure --
    sampling_args = {
        "t_min": args.interval_min,
        "t_max": args.interval_max,
        "cfg": args.cfg,
        "num_steps": args.num_sampling_steps,
        "t_start_min": args.fd_timestep_min_start,
    }
    def _fd_timestep_min(step: int) -> float:
        if args.fd_timestep_anneal_steps <= 0:
            return args.fd_timestep_min_end
        progress = min(max(step / args.fd_timestep_anneal_steps, 0.0), 1.0)
        return args.fd_timestep_min_start + (
            args.fd_timestep_min_end - args.fd_timestep_min_start
        ) * progress

    fd_train_step = get_fd_train_step(
        model_wo_ddp, judges, sampling_args, args,
        tokenizer=tokenizer, real_batch_fn=real_batch_fn,
        patch_discriminator=patch_discriminator,
        patch_d_optimizer=patch_d_optimizer,
        dmd_guidance=dmd_guidance,
        dmd_optimizer=dmd_optimizer,
    )

    # -- training loop --
    logger.info(f"training from step {args.current_step:,} -> {args.total_steps:,} "
                f"({args.start_epoch} -> {args.epochs} epochs)")

    global_bsz = args.batch_size * args.world_size
    ckpt_saver = AsyncCheckpointSaver()
    session_start = time.time()
    step_start = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # dynamic checkpoint frequency: target ~10 min between saves
    ckpt_target_minutes = 10.0
    ckpt_measure_interval = 1000
    ckpt_timer_start = time.perf_counter()
    ckpt_timer_step = args.current_step
    last_ckpt_step = args.current_step

    # metric logger
    metric_file = os.path.join(args.log_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metric_file, prefetch=True)
    for name, window, fmt in [
        ("lr",               1,               "{value:.6f}"),
        ("samples/s/device", args.print_freq, "{avg:.2f}"),
        ("samples/s",        args.print_freq, "{avg:.2f}"),
        ("samples_seen(M)",  args.print_freq, "{value:.2f}"),
        ("device_mem(GB)",   args.print_freq, "{value:.2f}"),
    ]:
        metric_logger.add_meter(name, SmoothedValue(window, fmt))

    def _infinite():
        while True:
            yield None

    for step, _ in metric_logger.log_every(
        _infinite(), args.print_freq, header="Train:",
        start_iteration=args.current_step, n_iterations=args.total_steps,
    ):
        model.train()
        adjust_learning_rate(optimizer, step, args)
        if args.fd_random_timestep_training:
            sampling_args["t_start_min"] = _fd_timestep_min(step)

        loss, loss_dict, dmd_fake_batch = fd_train_step()

        grad_norm = (torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                     if args.grad_clip > 0.0 else get_grad_norm(model.parameters()))

        if torch.isfinite(grad_norm):
            optimizer.step()
            ema_model.step(model)
        else:
            logger.warning(f"[step {step}] NaN/Inf grad_norm — skipping optimizer & EMA update")
        optimizer.zero_grad(set_to_none=True)

        if dmd_fake_batch is not None and torch.isfinite(grad_norm):
            fake_x0, fake_y = dmd_fake_batch
            dmd_guidance.fake_score_model.train()
            _set_requires_grad(dmd_guidance.fake_score_model, True)
            dmd_optimizer.zero_grad(set_to_none=True)
            dmd_fake_loss, dmd_fake_dict = dmd_guidance.compute_fake_score_loss(fake_x0, fake_y)
            dmd_fake_loss.backward()
            _all_reduce_grads(dmd_guidance.fake_score_model)
            dmd_optimizer.step()
            dmd_optimizer.zero_grad(set_to_none=True)
            loss_dict.update(dmd_fake_dict)

        torch.cuda.synchronize()

        args.current_step = step + 1
        args.samples_seen += global_bsz

        # timing & metrics
        step_time = time.perf_counter() - step_start
        step_start = time.perf_counter()

        loss_value = all_reduce_mean(loss.item())
        loss_dict = {k: all_reduce_mean(v) for k, v in loss_dict.items()}
        sps = args.batch_size / step_time if step_time > 0 else 0.0
        mem_gb = torch.cuda.max_memory_reserved() / (1024 ** 3) if torch.cuda.is_available() else 0.0

        metric_logger.update(
            loss=loss_value, grad_norm=grad_norm,
            lr=optimizer.param_groups[0]["lr"],
            **{"samples/s/device": sps, "samples/s": sps * args.world_size,
               "samples_seen(M)": args.samples_seen / 1e6, "device_mem(GB)": mem_gb},
            **loss_dict,
        )

        # wandb
        if step % args.print_freq == 0 and wandb_logger:
            elapsed = time.time() - session_start + args.last_elapsed_time
            remaining = args.total_steps - args.current_step
            eta = elapsed / args.current_step * remaining if args.current_step > 0 else 0.0
            elapsed_h = elapsed / 3600
            wandb_logger.update({
                "train/loss": loss_value,
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/grad_norm": grad_norm,
                "train/samples_seen_M": args.samples_seen / 1e6,
                "train/fd_timestep_min": sampling_args["t_start_min"],
                "perf/samples_per_sec_per_device": sps,
                "perf/samples_per_sec": sps * args.world_size,
                "perf/max_reserved_mem_gb": mem_gb,
                "perf/elapsed_real_hours": elapsed_h,
                "perf/elapsed_device_hours": elapsed_h * args.world_size,
                "perf/eta_real_hours": eta / 3600,
                "perf/eta_device_hours": eta / 3600 * args.world_size,
                **{f"train/{k}": v for k, v in loss_dict.items()},
            }, step=args.current_step)

        # dynamic checkpoint frequency
        steps_since_timer = args.current_step - ckpt_timer_step
        if steps_since_timer >= ckpt_measure_interval:
            elapsed_minutes = (time.perf_counter() - ckpt_timer_start) / 60.0
            minutes_per_step = elapsed_minutes / steps_since_timer
            new_save_every = max(100, round(ckpt_target_minutes / minutes_per_step / 100) * 100)
            if new_save_every != args.save_every:
                logger.info(f"adjusting save_every: {args.save_every} -> {new_save_every} "
                            f"({minutes_per_step * 1000:.1f} min/1k steps)")
                args.save_every = new_save_every
            ckpt_timer_start = time.perf_counter()
            ckpt_timer_step = args.current_step

        # checkpoint
        def _save(saver=ckpt_saver):
            elapsed = time.time() - session_start + args.last_elapsed_time
            fd_extra = {"fd_queue_states": save_fd_queue_states(judges)} if judges else {}
            if args.fd_adv_weight > 0:
                fd_extra["fd_adv_states"] = save_fd_adv_states(judges)
            if patch_discriminator is not None:
                fd_extra["patch_gan_state"] = {
                    "discriminator": patch_discriminator.state_dict(),
                    "optimizer": patch_d_optimizer.state_dict(),
                }
            if dmd_guidance is not None:
                dmd_state = dmd_guidance.state_dict_for_checkpoint()
                dmd_state["optimizer"] = dmd_optimizer.state_dict()
                fd_extra["dmd_guidance_state"] = dmd_state
            save_checkpoint(args, step, model_wo_ddp, optimizer, ema_model, elapsed,
                            saver=saver, extra=fd_extra)
            torch.distributed.barrier()

        if (args.current_step - last_ckpt_step >= args.save_every
                or args.current_step == args.total_steps):
            _save()
            last_ckpt_step = args.current_step

        if args.milestone_every > 0 and step > 0 and step % args.milestone_every == 0:
            _save()

        # slurm preemption
        if preempt_requested():
            logger.info(f"Preemption at step {args.current_step}: saving checkpoint ...")
            ckpt_saver.wait()
            _save(saver=None)
            logger.info(f"Preemption checkpoint saved at step {args.current_step}. Exiting.")
            return 0

        # visualization
        if args.vis_every > 0 and args.current_step % args.vis_every == 0:
            visualize(args, model_wo_ddp, ema_model, args.current_step, rng=rng, tokenizer=tokenizer)
            model_wo_ddp.train()

        # online evaluation
        if args.eval_every > 0 and args.online_eval and args.current_step % args.eval_every == 0:
            torch.cuda.empty_cache()
            evaluate_all_emas(
                args, model_wo_ddp, ema_model, fid_evaluator, tokenizer,
                step=args.current_step, wandb_logger=wandb_logger,
                cfg=args.cfg, num_images=args.num_images_for_eval_and_search,
            )
            model_wo_ddp.train()

    # -- final --
    ckpt_saver.wait()
    total = time.time() - session_start + args.last_elapsed_time
    metric_logger.synchronize_between_processes()
    logger.info(f"averaged stats: {metric_logger}")
    logger.info(f"Training complete. Total time: {datetime.timedelta(seconds=int(total))} "
                f"on {args.world_size} devices")
    torch.cuda.empty_cache()

    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def get_args_parser():
    parser = argparse.ArgumentParser("FD loss fine-tuning for generation models", add_help=False)

    # training
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--steps_per_epoch", default=1250, type=int)
    parser.add_argument("--batch_size", default=32, type=int, help="batch size per GPU")
    parser.add_argument("--noise_scale", type=float, default=1.0)
    parser.add_argument("--same_noise", action="store_true")

    # model architecture
    parser.add_argument("--model", default="pMF_B", type=str)
    parser.add_argument("--img_size", default=256, type=int)
    parser.add_argument("--patch_size", default=16, type=int)
    parser.add_argument("--label_drop_prob", default=0.1, type=float)
    parser.add_argument("--attn_dropout", type=float, default=0.0)
    parser.add_argument("--proj_dropout", type=float, default=0.0)
    parser.add_argument("--class_tokens", type=int, default=8)
    parser.add_argument("--time_tokens", type=int, default=4)
    parser.add_argument("--guidance_tokens", type=int, default=4)
    parser.add_argument("--interval_tokens", type=int, default=2)
    parser.add_argument("--norm_eps", type=float, default=0.01)
    parser.add_argument("--norm_p", type=float, default=1.0)
    parser.add_argument("--rope_2d", action="store_true")
    parser.add_argument("--learned_pe", action="store_true")
    parser.add_argument("--disable_v_head", action="store_true")
    parser.add_argument("--t_eps", type=float, default=5e-2)

    # tokenizer
    parser.add_argument("--tokenizer", default=None, type=str)
    parser.add_argument("--token_channels", default=3, type=int)
    parser.add_argument("--tokenizer_patch_size", default=1, type=int)

    # optimization
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--lr_sched", type=str, default="constant", choices=["constant", "cosine"])
    parser.add_argument("--warmup_rate", type=float, default=0.0)
    parser.add_argument("--warmup_epochs", type=int, default=-1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=0.0, help="gradient clip, 0.0 means no clip")
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--use_muon", action="store_true")
    parser.add_argument("--muon_lr", type=float, default=1e-3)
    parser.add_argument("--muon_momentum", type=float, default=0.95)
    parser.add_argument("--muon_weight_decay", type=float, default=0.0)
    parser.add_argument("--ema_type", default="edm", type=str, choices=["const", "edm"])
    parser.add_argument("--ema_rates", default=[0.9999, 0.9996], type=float, nargs="+")
    parser.add_argument("--ema_halflife_kimg", default=[250, 500, 1000, 2000], type=float, nargs="+")
    parser.add_argument("--eval_ema_labels", default=None, type=str, nargs="+")

    parser.add_argument("--grad_checkpointing", action="store_true")
    parser.add_argument(
        "--fd_repr_grad_checkpointing",
        action="store_true",
        help="enable gradient checkpointing for FD representation models only",
    )

    # diffusion / flow-matching
    parser.add_argument("--P_mean", type=float, default=0.8)
    parser.add_argument("--P_std", type=float, default=0.8)
    parser.add_argument("--legacy_time_convention", action="store_true")
    parser.add_argument("--tr_uniform", action="store_true")
    parser.add_argument("--ratio_r_neq_t", type=float, default=0.5)
    parser.add_argument("--cfg_beta", type=float, default=1.0)
    parser.add_argument("--cfg_omega_max", type=float, default=7.0)
    parser.add_argument("--aux_head_depth", type=int, default=8)
    parser.add_argument("--loss_type", type=str, default="v", choices=["v", "x"])
    parser.add_argument("--aux_pred_type", type=str, default="v", choices=["v", "x"])
    parser.add_argument("--perceptual_threshold", type=float, default=0.8)
    parser.add_argument("--perceptual_loss_on_aux", action="store_true")

    # sampling & generation
    parser.add_argument("--sampling_method", type=str, default="heun", choices=["euler", "heun"])
    parser.add_argument("--num_sampling_steps", type=int, default=50)
    parser.add_argument("--cfg", default=4.0, type=float)
    parser.add_argument("--cfg_list", type=float, nargs="+",
                        default=[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.5, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0])
    parser.add_argument("--interval_min", type=float, default=0.1)
    parser.add_argument("--interval_max", type=float, default=1.0)
    parser.add_argument("--vis_steps", default=[1], type=int, nargs="+")

    # data
    parser.add_argument("--data_path", default="./data/imagenet/train", type=str)
    parser.add_argument("--num_classes", default=1000, type=int)
    parser.add_argument("--class_of_interest", default=[207, 360, 387, 974, 88, 979, 417, 279],
                        type=int, nargs="+")
    parser.add_argument("--force_class_of_interest", action="store_true")
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # checkpointing
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--resume_from", default=None)
    parser.add_argument("--load_from", type=str, default=None)
    parser.add_argument("--keep_n_ckpts", default=3, type=int)
    parser.add_argument("--milestone_interval", default=20, type=int)

    # evaluation
    parser.add_argument("--online_eval", action="store_true")
    parser.add_argument("--num_images_for_eval_and_search", default=10000, type=int)
    parser.add_argument("--num_images", default=50000, type=int)
    parser.add_argument("--eval_bsz", type=int, default=64)
    parser.add_argument("--fid_stats_path", type=str, default="data/fid_stats/guided_diffusion_stats.npz")
    parser.add_argument("--keep_eval_folder", action="store_true")

    parser.add_argument("--save_eval_images", action="store_true")
    parser.add_argument("--cfg_min", default=1.0, type=float)
    parser.add_argument("--cfg_max", default=25.0, type=float)
    parser.add_argument("--overwrite_cache", action="store_true")

    # FD fine-tuning
    parser.add_argument("--queue_size", type=int, default=10000)
    parser.add_argument("--fd_loss_type", choices=("mind", "frechet"), default="mind",
                        help="main queue loss in this entrypoint")
    parser.add_argument("--mind_num_projections", type=int, default=1000,
                        help="number of random projections sampled each step for queue MIND training")
    parser.add_argument("--mind_projection_batch_size", type=int, default=1000,
                        help="projection chunk size for queue MIND training")
    parser.add_argument("--mind_projection_seed", type=int, default=2026,
                        help="base seed for per-step queue MIND projection directions")
    parser.add_argument("--mind_real_queue_size", type=int, default=50000,
                        help="number of real features kept for queue MIND training")
    parser.add_argument("--mind_sample_size", type=int, default=10000,
                        help="number of fake/real features sampled per MIND step")
    parser.add_argument("--mind_real_baseline_gamma", type=float, default=1.0,
                        help="gamma for relu(MIND(fake, real_a) - gamma * MIND(real_b, real_c))")
    parser.add_argument("--mind_real_queue_refresh_steps", type=int, default=100,
                        help="refresh the real MIND queue every N steps; <=0 disables refresh")
    parser.add_argument("--mind_real_queue_refresh_batches", type=int, default=1,
                        help="number of real dataloader batches enqueued at each MIND real queue refresh")
    parser.add_argument("--fd_fid_norm_eps", type=float, default=0.01)
    parser.add_argument("--fd_queue_fill_bsz", type=int, default=256)
    parser.add_argument("--fd_repr_models", type=str, nargs="+", default=["inception"],
                        help="feature extractors: 'inception', 'inception:Mixed_6e', or timm model names")
    parser.add_argument("--fd_repr_stats_paths", type=str, nargs="+", default=None,
                        help="reference stats (.npz) per repr model; auto-inferred if omitted")
    parser.add_argument("--fd_repr_weights", type=float, nargs="+", default=None,
                        help="per-model FID loss weight (default 1.0 each)")
    parser.add_argument("--fd_repr_pool_types", type=str, nargs="+", default=None,
                        help="pool type per repr model: 'cls' or 'avg' (default 'cls')")
    parser.add_argument("--fd_target_sizes", type=int, nargs="+", default=None,
                        help="per-model target resolution override (default: model's native size)")
    parser.add_argument("--fd_online_accum", action="store_true",
                        help="use online accumulators for FD (avoids cloning 50k queue each step)")
    parser.add_argument("--fd_eigvalsh", action="store_true",
                        help="use eigvalsh on symmetric product instead of eigvals (~8x faster, exact)")
    parser.add_argument("--fd_ema_beta", type=float, default=0.0, metavar="BETA",
                        help="EMA decay for FD stats (0=disabled, use queue). "
                             "Implies online_accum. E.g. 0.999 → ~1000-batch window")
    parser.add_argument("--fd_random_timestep_training", action="store_true",
                        help="train FD on one-step denoising from real-image x_t samples")
    parser.add_argument("--fd_timestep_min_start", type=float, default=0.0,
                        help="initial lower bound for t_start ~ Uniform(t_min, 1)")
    parser.add_argument("--fd_timestep_min_end", type=float, default=1.0,
                        help="final lower bound for t_start ~ Uniform(t_min, 1)")
    parser.add_argument("--fd_timestep_anneal_steps", type=int, default=0,
                        help="steps to linearly anneal t_min from start to end")
    parser.add_argument("--fd_timestep_logit_normal", action="store_true",
                        help="sample FD timestep with JiT's logit-normal P_mean/P_std instead of uniform annealing")
    parser.add_argument("--jit_loss_weight", type=float, default=0.0,
                        help="weight for adding the original JiT denoising loss to FD loss")
    parser.add_argument("--fd_whiten", action="store_true",
                        help="use real-reference whitening for the base FD loss")
    parser.add_argument("--fd_whiten_eps", type=float, default=1e-3,
                        help="diagonal and eigenvalue floor for base FD real-reference whitening")
    parser.add_argument("--fd_adv_weight", type=float, default=0.0,
                        help="weight for real-whitened adversarial FD; 0 disables it")
    parser.add_argument("--fd_adv_backbone", choices=("repr", "patchgan"), default="repr",
                        help="trainable psi for adversarial FD: repr reuses FD model, patchgan uses patch features")
    parser.add_argument("--fd_adv_lr", type=float, default=1e-6,
                        help="learning rate for trainable adversarial psi")
    parser.add_argument("--fd_adv_beta1", type=float, default=0.9,
                        help="AdamW beta1 for adversarial psi")
    parser.add_argument("--fd_adv_beta2", type=float, default=0.999,
                        help="AdamW beta2 for adversarial psi")
    parser.add_argument("--fd_adv_weight_decay", type=float, default=0.0,
                        help="AdamW weight decay for adversarial psi")
    parser.add_argument("--fd_adv_steps", type=int, default=1,
                        help="number of psi maximization steps per generator step")
    parser.add_argument("--fd_adv_grad_clip", type=float, default=0.0,
                        help="clip adversarial psi gradient norm before optimizer step; 0 disables")
    parser.add_argument("--fd_adv_freeze_batchnorm", action="store_true",
                        help="keep BatchNorm layers in FD-Adv psi frozen, including affine parameters")
    parser.add_argument("--fd_adv_start_step", type=int, default=1000,
                        help="global step at which adversarial FD starts")
    parser.add_argument("--fd_adv_warmup_steps", type=int, default=4000,
                        help="linearly ramp adversarial FD generator weight after start_step")
    parser.add_argument("--fd_adv_whiten_eps", type=float, default=1e-1,
                        help="diagonal and eigenvalue floor for real-batch whitening")
    parser.add_argument("--fd_adv_no_whiten", action="store_true",
                        help="disable real whitening for adversarial FD")
    parser.add_argument("--fd_adv_neg_real_degrade_ratio", type=float, default=0.0,
                        help="ratio of degraded real images mixed into the adversarial FD critic negative batch")
    parser.add_argument("--fd_adv_log_raw", action="store_true",
                        help="also log raw adversarial FD; expensive because it adds matrix eigensolves")
    parser.add_argument("--fd_adv_log_raw_freq", type=int, default=1,
                        help="compute raw adversarial FD every N steps when fd_adv_log_raw is enabled")
    parser.add_argument("--fd_adv_ema_beta", type=float, default=0.99,
                        help="EMA decay for adversarial psi real/fake feature stats")
    parser.add_argument("--fd_adv_patch_channels", type=int, default=64,
                        help="base channels for PatchGAN adversarial FD encoder")
    parser.add_argument("--fd_adv_patch_max_channels", type=int, default=256,
                        help="maximum channels for PatchGAN adversarial FD encoder")
    parser.add_argument("--fd_adv_patch_layers", type=int, default=3,
                        help="number of stride-2 layers for PatchGAN adversarial FD encoder")
    parser.add_argument("--fd_adv_patch_max_patches", type=int, default=8192,
                        help="maximum PatchGAN feature patches per rank before all-gather; 0 uses all patches")
    parser.add_argument("--fd_adv_patch_batchnorm", action="store_true",
                        help="use BatchNorm in PatchGAN adversarial FD encoder")
    parser.add_argument("--fd_adv_patch_train_loss", choices=("fd", "hinge"), default="fd",
                        help="PatchGAN adversarial encoder update: maximize FD or train a real/fake hinge discriminator")
    parser.add_argument("--fd_adv_patch_pretrain_start_step", type=int, default=1000,
                        help="global step at which PatchGAN adversarial FD discriminator pretrain starts")
    parser.add_argument("--fd_adv_patch_pretrain_steps", type=int, default=0,
                        help="number of training steps used to pretrain PatchGAN adversarial FD encoder")
    parser.add_argument("--fd_adv_patch_pretrain_lr", type=float, default=0.0,
                        help="temporary lr for PatchGAN adversarial FD pretrain; <=0 reuses --fd_adv_lr")
    parser.add_argument("--fd_adv_patch_pretrain_log_freq", type=int, default=100,
                        help="log PatchGAN adversarial FD pretrain every N steps; 0 disables progress logs")
    parser.add_argument("--patch_gan_weight", type=float, default=0.0,
                        help="weight for PatchGAN generator hinge loss; 0 disables it")
    parser.add_argument("--patch_gan_lr", type=float, default=2e-4,
                        help="PatchGAN discriminator learning rate")
    parser.add_argument("--patch_gan_channels", type=int, default=64,
                        help="base channel count for PatchGAN discriminator")
    parser.add_argument("--patch_gan_layers", type=int, default=3,
                        help="number of stride-2 PatchGAN discriminator stages")
    parser.add_argument("--patch_gan_weight_decay", type=float, default=0.0,
                        help="PatchGAN discriminator AdamW weight decay")
    parser.add_argument("--patch_gan_start_step", type=int, default=0,
                        help="global step at which PatchGAN losses start")
    parser.add_argument("--patch_gan_warmup_steps", type=int, default=0,
                        help="linearly ramp PatchGAN generator weight after start_step")
    parser.add_argument("--patch_gan_adaptive_weight", action="store_true",
                        help="scale generator GAN loss by FD/GAN gradient norm ratio")
    parser.add_argument("--patch_gan_max_adaptive_weight", type=float, default=1e4,
                        help="maximum adaptive adversarial weight")
    parser.add_argument("--dmd_guidance_weight", type=float, default=0.0,
                        help="weight for DMD one-step distribution matching loss; 0 disables it unless --dmd_discriminator_only is set")
    parser.add_argument("--dmd_discriminator_only", action="store_true",
                        help="train the DMD fake-score/ASD discriminator but skip generator-side distribution matching loss")
    parser.add_argument("--dmd_real_guidance_scale", type=float, default=None,
                        help="CFG scale for the frozen DMD real score model; defaults to --cfg")
    parser.add_argument("--dmd_fake_guidance_scale", type=float, default=1.0,
                        help="CFG scale for the DMD fake score model; DMD2 keeps this at 1.0")
    parser.add_argument("--dmd_dm_grad_mode", type=str, default="original",
                        choices=("original", "uncond_real_minus_fake"),
                        help="generator-side DMD pseudo-gradient: original or unconditional real score minus fake score")
    parser.add_argument("--dmd_guidance_lr", type=float, default=2e-6,
                        help="learning rate for the DMD fake score model")
    parser.add_argument("--dmd_guidance_weight_decay", type=float, default=0.01,
                        help="AdamW weight decay for the DMD fake score model")
    parser.add_argument("--dmd_generator_update_ratio", type=int, default=1,
                        help="add DMD generator loss every N steps; fake score still updates every step")
    parser.add_argument("--dmd_fake_update_ratio", type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--dmd_min_t", type=float, default=0.02,
                        help="minimum noisy timestep used by DMD distribution matching")
    parser.add_argument("--dmd_max_t", type=float, default=0.98,
                        help="maximum noisy timestep used by DMD distribution matching")
    parser.add_argument("--dmd_decoupled", action="store_true",
                        help="split DMD into CFG augmentation (CA) and distribution matching (DM) terms")
    parser.add_argument("--dmd_ca_weight", type=float, default=1.0,
                        help="weight for the decoupled CFG augmentation term")
    parser.add_argument("--dmd_dm_weight", type=float, default=1.0,
                        help="weight for the decoupled distribution matching term")
    parser.add_argument("--dmd_ca_min_t", type=float, default=None,
                        help="minimum noisy timestep for decoupled CFG augmentation; defaults to dmd_min_t")
    parser.add_argument("--dmd_ca_max_t", type=float, default=None,
                        help="maximum noisy timestep for decoupled CFG augmentation; defaults to dmd_max_t")
    parser.add_argument("--dmd_dm_min_t", type=float, default=None,
                        help="minimum noisy timestep for decoupled distribution matching; defaults to dmd_min_t")
    parser.add_argument("--dmd_dm_max_t", type=float, default=None,
                        help="maximum noisy timestep for decoupled distribution matching; defaults to dmd_max_t")
    parser.add_argument("--dmd_fake_min_t", type=float, default=0.0,
                        help="minimum noisy timestep used by DMD fake-score regression")
    parser.add_argument("--dmd_fake_max_t", type=float, default=1.0,
                        help="maximum noisy timestep used by DMD fake-score regression")
    parser.add_argument("--dmd_timestep_logit_normal", action="store_true",
                        help="sample DMD timesteps with the model's logit-normal sampler")
    parser.add_argument("--dmd_grad_norm_eps", type=float, default=1e-6,
                        help="epsilon for normalizing DMD distribution-matching gradients")
    parser.add_argument("--dmd_grad_clip", type=float, default=0.0,
                        help="optional elementwise clip for DMD pseudo-gradients; 0 disables")
    parser.add_argument("--dmd_fake_loss_snr_weight", action="store_true",
                        help="weight DMD fake-score regression by linear-interpolation SNR + 1")
    parser.add_argument("--dmd_fake_loss_type", type=str, default="dmd", choices=["dmd", "asd"],
                        help="fake-score training loss: original DMD regression or ASD discriminator loss")
    parser.add_argument("--dmd_fake_prediction_type", type=str, default="x0", choices=["x0", "v"],
                        help="prediction space for DMD fake-score regression/ASD fake loss")
    parser.add_argument("--dmd_asd_gamma", type=float, default=-0.75,
                        help="negative teacher-distance weight for ASD fake-score discriminator loss")
    parser.add_argument("--dmd_teacher_load_from", type=str, default=None,
                        help="checkpoint for the frozen DMD real score model; defaults to load_from when resuming")
    parser.add_argument("--dmd_teacher_strict_load", action="store_true",
                        help="strictly load dmd_teacher_load_from into the frozen real score model")
    # logging & tracking
    parser.add_argument("--output_dir", default="./work_dirs")
    parser.add_argument("--local_eval_dir", type=str, default=None)
    parser.add_argument("--print_freq", type=int, default=50)
    parser.add_argument("--eval_freq", type=int, default=10)
    parser.add_argument("--vis_freq", type=int, default=10)
    parser.add_argument("--val_freq", type=int, default=10)
    parser.add_argument("--save_freq", type=int, default=5)
    parser.add_argument("--vis_only", action="store_true")
    parser.add_argument("--disable_vis", action="store_true")
    parser.add_argument("--last_elapsed_time", type=float, default=0.0)
    parser.add_argument("--current_step", type=int, default=0)
    parser.add_argument("--samples_seen", type=int, default=0)
    parser.add_argument("--project", default="One3", type=str)
    parser.add_argument("--entity", default=None, type=str)
    parser.add_argument("--exp_name", default=None, type=str)
    parser.add_argument("--enable_wandb", action="store_true")
    parser.add_argument("--disable_wandb", action="store_false", dest="enable_wandb")

    # system
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--dtype", default="bf16", type=str, choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--compile", action="store_true")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    sys.exit(train_and_evaluate(args))
