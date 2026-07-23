"""Learn and evaluate a universal dose-response pattern for frozen pMF models.

The generator and representation models are always frozen.  Training updates
only one low-resolution parameter ``u`` and uses Inception FID.  CLIP is loaded
only after training, when the learned pattern is evaluated on held-out noise
and labels over a dense alpha grid.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from frechet_distance.judges import infer_stats_path
from frechet_distance.losses import (
    compute_frechet_distance_loss,
    diff_all_gather,
    load_mu_and_sigma_reference,
    precompute_sigma_ref_sqrt,
)
from frechet_distance.metrics import compute_fid as np_fid
from frechet_distance.queue import FeatureQueue
from frechet_distance.repr_models import load_repr_model
from utils.distributed_util import (
    broadcast_module_params,
    get_global_rank,
    get_world_size,
    is_enabled,
)


logger = logging.getLogger("FD_loss")


class UniversalPattern(nn.Module):
    """A shared bounded RGB pattern periodically tiled over an image."""

    def __init__(self, size: int = 16):
        super().__init__()
        if size < 1:
            raise ValueError("pattern size must be positive")
        self.size = int(size)
        self.u = nn.Parameter(torch.zeros(1, 3, self.size, self.size))

    def bounded_tile(self, height: int, width: int) -> torch.Tensor:
        patch = torch.tanh(self.u)
        repeat_h = math.ceil(height / self.size)
        repeat_w = math.ceil(width / self.size)
        return patch.repeat(1, 1, repeat_h, repeat_w)[..., :height, :width]

    def regularizer(self, mean_weight: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        patch = torch.tanh(self.u)
        tv_h = (patch[:, :, 1:, :] - patch[:, :, :-1, :]).abs().mean()
        tv_w = (patch[:, :, :, 1:] - patch[:, :, :, :-1]).abs().mean()
        tv = tv_h + tv_w
        mean_sq = patch.mean(dim=(2, 3)).square().mean()
        return tv + mean_weight * mean_sq, tv, mean_sq


@dataclass
class Moments:
    feat_sum: torch.Tensor
    feat_outer: torch.Tensor
    count: int = 0

    @classmethod
    def create(cls, feat_dim: int, device: torch.device) -> "Moments":
        return cls(
            feat_sum=torch.zeros(feat_dim, dtype=torch.float64, device=device),
            feat_outer=torch.zeros(
                feat_dim, feat_dim, dtype=torch.float64, device=device
            ),
        )

    @torch.no_grad()
    def update(self, features: torch.Tensor) -> None:
        features64 = features.detach().double()
        self.feat_sum.add_(features64.sum(dim=0))
        self.feat_outer.addmm_(features64.T, features64)
        self.count += features64.shape[0]

    @torch.no_grad()
    def zero_(self) -> None:
        self.feat_sum.zero_()
        self.feat_outer.zero_()
        self.count = 0


def _validate_args(args: argparse.Namespace) -> None:
    if not args.model.startswith("pMF_"):
        raise ValueError(f"This entry point only supports pMF models, got {args.model!r}")
    if args.tokenizer is not None:
        raise ValueError("pMF universal-pattern training operates in pixel space; tokenizer must be None")
    if args.load_from is None:
        raise ValueError("--load_from must point to the frozen pMF checkpoint")
    if args.queue_size < 2:
        raise ValueError("--queue_size must be at least 2")
    if not 0.0 <= args.fd_ema_beta < 1.0:
        raise ValueError("--fd_ema_beta must be in [0, 1)")
    if args.fd_fid_norm_eps <= 0:
        raise ValueError("--fd_fid_norm_eps must be positive")
    if args.pattern_eval_images < 2:
        raise ValueError("--pattern_eval_images must be at least 2")
    if args.pattern_eval_blocks < 1:
        raise ValueError("--pattern_eval_blocks must be positive")
    if args.pattern_eval_images % args.pattern_eval_blocks:
        raise ValueError("--pattern_eval_images must be divisible by --pattern_eval_blocks")
    if args.pattern_eval_images // args.pattern_eval_blocks < 2:
        raise ValueError("each held-out evaluation block must contain at least 2 images")
    if not args.pattern_eval_seeds:
        raise ValueError("--pattern_eval_seeds cannot be empty")
    if args.lr <= 0:
        raise ValueError("--lr must be positive")
    if not 0 <= args.min_lr <= args.lr:
        raise ValueError("--min_lr must be in [0, lr]")
    if args.pattern_mono_margin < 0:
        raise ValueError("--pattern_mono_margin must be non-negative")
    if args.pattern_mono_weight < 0 or args.pattern_reg_weight < 0:
        raise ValueError("pattern loss weights must be non-negative")
    if args.pattern_mean_weight < 0:
        raise ValueError("--pattern_mean_weight must be non-negative")
    if args.pattern_clip_fdr_denominator <= 0:
        raise ValueError("--pattern_clip_fdr_denominator must be positive")
    if args.save_freq < 1:
        raise ValueError("--save_freq must be positive")
    if not 0 < args.pattern_confidence < 1:
        raise ValueError("--pattern_confidence must be in (0, 1)")
    if args.pattern_eval_only and args.pattern_train_only:
        raise ValueError("--pattern_eval_only and --pattern_train_only are mutually exclusive")
    if not args.pattern_eval_only and args.epochs < 1:
        raise ValueError("--epochs must be positive when pattern training is enabled")
    if not args.pattern_eval_only and args.steps_per_epoch < 1:
        raise ValueError("--steps_per_epoch must be positive when training is enabled")

    train_alphas = list(args.pattern_train_alphas)
    eval_alphas = list(args.pattern_eval_alphas)
    for name, values in (
        ("--pattern_train_alphas", train_alphas),
        ("--pattern_eval_alphas", eval_alphas),
    ):
        if not values:
            raise ValueError(f"{name} cannot be empty")
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError(f"{name} values must be finite and non-negative")
        if values != sorted(set(values)):
            raise ValueError(f"{name} values must be unique and strictly increasing")
        if values[0] != 0.0:
            raise ValueError(f"{name} must start at alpha=0")
    if len(train_alphas) < 2 or len(eval_alphas) < 2:
        raise ValueError("training and evaluation both require alpha=0 plus a positive alpha")


def _load_frozen_pmf(args: argparse.Namespace) -> nn.Module:
    import models
    from models.denoiser_pmf import convert_pmf_checkpoint

    if args.model not in models.pMFDenoiser_models:
        raise ValueError(
            f"Unknown pMF model {args.model!r}; "
            f"available={sorted(models.pMFDenoiser_models)}"
        )
    model = models.pMFDenoiser_models[args.model](
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_channels=args.token_channels,
        tokenizer_patch_size=args.tokenizer_patch_size,
        num_classes=args.num_classes,
        label_drop_prob=args.label_drop_prob,
        P_mean=args.P_mean,
        P_std=args.P_std,
        ratio_r_neq_t=args.ratio_r_neq_t,
        cfg_beta=args.cfg_beta,
        tr_uniform=args.tr_uniform,
        cfg_omega_max=args.cfg_omega_max,
        aux_head_depth=args.aux_head_depth,
        class_tokens=args.class_tokens,
        time_tokens=args.time_tokens,
        guidance_tokens=args.guidance_tokens,
        interval_tokens=args.interval_tokens,
        t_eps=args.t_eps,
        perceptual_threshold=args.perceptual_threshold,
        perceptual_loss_on_aux=args.perceptual_loss_on_aux,
        rope_2d=args.rope_2d,
        learned_pe=args.learned_pe,
        disable_v_head=args.disable_v_head,
        noise_scale=args.noise_scale,
        norm_eps=args.norm_eps,
        norm_p=args.norm_p,
        grad_checkpointing=args.grad_checkpointing,
    ).cuda()

    checkpoint = torch.load(args.load_from, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    state_dict = convert_pmf_checkpoint(state_dict)
    message = model.load_state_dict(state_dict, strict=False)
    if message.missing_keys:
        logger.warning("Frozen pMF checkpoint missing keys: %s", message.missing_keys)
    if message.unexpected_keys:
        logger.warning("Frozen pMF checkpoint unexpected keys: %s", message.unexpected_keys)
    del checkpoint, state_dict

    model.eval().requires_grad_(False)
    broadcast_module_params(model, src=0)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable != 0:
        raise RuntimeError(f"Generator freeze failed: {trainable} parameters remain trainable")
    logger.info("Loaded and froze pMF generator from %s", args.load_from)
    return model


def _load_frozen_repr(
    name: str,
    *,
    target_size: int,
) -> tuple[nn.Module, int]:
    model, feat_dim, _, _ = load_repr_model(name, target_size=target_size)
    model.eval().requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable != 0:
        raise RuntimeError(
            f"Representation model {name!r} freeze failed: "
            f"{trainable} parameters remain trainable"
        )
    return model, feat_dim


def _fixed_batch_seed(base_seed: int, batch_index: int, rank: int) -> int:
    # Keep streams stable for a fixed world size and independent across ranks.
    return int(base_seed + 1_000_003 * batch_index + 10_007 * rank)


@torch.no_grad()
def _generate_fixed_batch(
    model: nn.Module,
    args: argparse.Namespace,
    *,
    batch_size: int,
    base_seed: int,
    batch_index: int,
    labels: torch.Tensor | None = None,
) -> torch.Tensor:
    device = torch.device("cuda")
    rank = get_global_rank()
    generator = torch.Generator(device=device)
    generator.manual_seed(_fixed_batch_seed(base_seed, batch_index, rank))
    z = torch.randn(
        batch_size,
        model.in_channels,
        model.input_size,
        model.input_size,
        device=device,
        generator=generator,
    )
    z.mul_(args.noise_scale)
    if labels is None:
        labels = torch.randint(
            0,
            args.num_classes,
            (batch_size,),
            device=device,
            generator=generator,
        )

    with torch.autocast(
        "cuda", enabled=args.enable_amp, dtype=args.amp_dtype
    ):
        images = model.generate(
            batch_size,
            labels,
            cfg=args.cfg,
            args=args,
            verbose=False,
            z_t=z,
        )
    # Use a normal (non-inference) float tensor as the constant input to the
    # differentiable pattern/Inception path.
    return images.detach().float().clone().clamp_(-1.0, 1.0)


def _apply_pattern_model_space(
    images: torch.Tensor,
    pattern: UniversalPattern,
    alpha: float,
) -> torch.Tensor:
    tiled = pattern.bounded_tile(images.shape[-2], images.shape[-1])
    return (images + float(alpha) * tiled).clamp(-1.0, 1.0)


def _to_unit_range(images_model_space: torch.Tensor) -> torch.Tensor:
    return images_model_space.mul(0.5).add(0.5).clamp(0.0, 1.0)


def _extract_primary_features(
    repr_model: nn.Module,
    images: torch.Tensor,
    *,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> torch.Tensor:
    with torch.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
        primary, _ = repr_model(images)
    return primary.float()


def _local_batches(num_images: int, batch_size: int, world_size: int) -> Iterable[tuple[int, int]]:
    if num_images % world_size:
        raise ValueError(
            f"num_images={num_images} must be divisible by world_size={world_size}"
        )
    local_images = num_images // world_size
    num_batches = math.ceil(local_images / batch_size)
    for batch_index in range(num_batches):
        start = batch_index * batch_size
        yield batch_index, min(batch_size, local_images - start)


def _balanced_training_labels(
    args: argparse.Namespace,
    *,
    batch_index: int,
    local_batch_size: int,
) -> torch.Tensor:
    """Assign the fixed training set deterministically and class-balanced."""
    global_batch_start = batch_index * args.batch_size * get_world_size()
    rank_start = global_batch_start + get_global_rank() * local_batch_size
    return (
        torch.arange(
            rank_start,
            rank_start + local_batch_size,
            device="cuda",
            dtype=torch.long,
        )
        % args.num_classes
    )


@torch.no_grad()
def _initialize_training_queues(
    args: argparse.Namespace,
    generator: nn.Module,
    inception: nn.Module,
    feat_dim: int,
    pattern: UniversalPattern,
    mu_ref: torch.Tensor,
    sigma_ref: torch.Tensor,
    sigma_ref_sqrt: torch.Tensor | None,
) -> tuple[float, dict[float, FeatureQueue]]:
    """Initialize every positive-alpha queue on the fixed training set."""
    device = torch.device("cuda")
    world_size = get_world_size()
    alphas = list(args.pattern_train_alphas)
    positive_alphas = alphas[1:]
    queues = {
        alpha: FeatureQueue(
            size=args.queue_size,
            feat_dim=feat_dim,
            online_accum=True,
            ema_beta=args.fd_ema_beta,
        ).cuda()
        for alpha in positive_alphas
    }
    baseline_moments = Moments.create(feat_dim, device)
    zero_pattern = float(pattern.u.detach().abs().max()) == 0.0

    filled = 0
    for batch_index, local_bsz in _local_batches(
        args.queue_size, args.batch_size, world_size
    ):
        labels = _balanced_training_labels(
            args,
            batch_index=batch_index,
            local_batch_size=local_bsz,
        )
        base = _generate_fixed_batch(
            generator,
            args,
            batch_size=local_bsz,
            base_seed=args.pattern_train_seed,
            batch_index=batch_index,
            labels=labels,
        )
        baseline_local = _extract_primary_features(
            inception,
            _to_unit_range(base),
            use_amp=False,
            amp_dtype=args.amp_dtype,
        )
        baseline_global = diff_all_gather(baseline_local)
        count = baseline_global.shape[0]
        baseline_moments.update(baseline_global)

        if zero_pattern:
            for alpha in positive_alphas:
                queue = queues[alpha]
                if queue.ema_stats:
                    queue.accumulate_batch(baseline_global)
                else:
                    queue.feats[filled : filled + count].copy_(baseline_global)
        else:
            for alpha in positive_alphas:
                perturbed = _to_unit_range(
                    _apply_pattern_model_space(base, pattern, alpha)
                )
                local_features = _extract_primary_features(
                    inception,
                    perturbed,
                    use_amp=False,
                    amp_dtype=args.amp_dtype,
                )
                global_features = diff_all_gather(local_features)
                queue = queues[alpha]
                if queue.ema_stats:
                    queue.accumulate_batch(global_features)
                else:
                    queue.feats[filled : filled + count].copy_(global_features)
                del perturbed, local_features, global_features

        filled += count
        if get_global_rank() == 0:
            logger.info(
                "[pattern] initializing fixed train stats: %d/%d",
                filled,
                args.queue_size,
            )
        del base, labels, baseline_local, baseline_global

    baseline_mu = baseline_moments.feat_sum / args.queue_size
    baseline_sigma = (
        baseline_moments.feat_outer
        - baseline_moments.feat_sum.unsqueeze(1)
        * baseline_moments.feat_sum.unsqueeze(0)
        / args.queue_size
    ) / (args.queue_size - 1)
    baseline_fid = float(
        compute_frechet_distance_loss(
            mu_ref,
            sigma_ref,
            mu=baseline_mu,
            sigma=baseline_sigma,
            sigma_ref_sqrt=sigma_ref_sqrt,
        )
    )

    for queue in queues.values():
        if queue.ema_stats:
            queue._finalize_streaming_init()
        else:
            queue.ptr.zero_()
            queue._init_accumulators()
    del baseline_moments, baseline_mu, baseline_sigma
    torch.cuda.empty_cache()
    logger.info(
        "[pattern] alpha=0 fixed-train FID: %.6f; "
        "queue_size=%d; fd_ema_beta=%.6f",
        baseline_fid,
        args.queue_size,
        args.fd_ema_beta,
    )
    return baseline_fid, queues


def _compose_fd_gradient(
    fid_values: list[float],
    fid_gradients: list[torch.Tensor | None],
    *,
    mono_weight: float,
    mono_margin: float,
    fid_norm_eps: float,
) -> tuple[torch.Tensor, float, float, list[float]]:
    """Combine pMF-normalized FD gradients with the new monotonic constraint."""
    num_positive = len(fid_values) - 1
    coefficients = [0.0] + [
        1.0 / (num_positive * (fid + fid_norm_eps))
        for fid in fid_values[1:]
    ]
    hinge_values: list[float] = []
    for index in range(len(fid_values) - 1):
        hinge = max(
            0.0,
            fid_values[index + 1] - fid_values[index] + mono_margin,
        )
        hinge_values.append(hinge)
        if hinge > 0.0:
            coefficients[index] -= mono_weight
            coefficients[index + 1] += mono_weight

    template = next(g for g in fid_gradients if g is not None)
    gradient = torch.zeros_like(template)
    for coefficient, grad in zip(coefficients, fid_gradients):
        if grad is not None and coefficient != 0.0:
            gradient.add_(grad, alpha=coefficient)

    base_fd_loss = float(
        np.mean(
            [
                fid / (fid + fid_norm_eps)
                for fid in fid_values[1:]
            ]
        )
    )
    mono_loss = float(sum(hinge_values))
    return gradient, base_fd_loss, mono_loss, coefficients


def _save_pattern_checkpoint(
    args: argparse.Namespace,
    pattern: UniversalPattern,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    step: int,
    name: str,
) -> Path | None:
    if get_global_rank() != 0:
        return None
    path = Path(args.ckpt_dir) / name
    payload = {
        "pattern": pattern.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "train_alphas": list(args.pattern_train_alphas),
        "pattern_size": args.pattern_size,
        "queue_size": args.queue_size,
        "fd_ema_beta": args.fd_ema_beta,
        "generator_checkpoint": args.load_from,
    }
    torch.save(payload, path)
    logger.info("Saved universal pattern checkpoint: %s", path)
    return path


def _save_pattern_preview(
    args: argparse.Namespace,
    pattern: UniversalPattern,
    name: str = "pattern_preview.png",
) -> None:
    if get_global_rank() != 0:
        return
    from PIL import Image

    with torch.no_grad():
        tiled = pattern.bounded_tile(args.img_size, args.img_size)[0]
        preview = (
            tiled.mul(0.5)
            .add(0.5)
            .clamp(0.0, 1.0)
            .mul(255)
            .byte()
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
    Image.fromarray(preview).save(Path(args.log_dir) / name)


def _load_pattern_resume(
    args: argparse.Namespace,
    pattern: UniversalPattern,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, int]:
    if args.pattern_resume is None:
        return 0, 0
    checkpoint = torch.load(
        args.pattern_resume, map_location="cuda", weights_only=False
    )
    if not args.pattern_eval_only:
        saved_alphas = checkpoint.get("train_alphas")
        if saved_alphas is not None and list(saved_alphas) != list(
            args.pattern_train_alphas
        ):
            raise ValueError(
                "Resumed pattern uses different training alphas: "
                f"{saved_alphas} vs {list(args.pattern_train_alphas)}"
            )
        saved_queue_size = checkpoint.get("queue_size")
        if saved_queue_size is not None and int(saved_queue_size) != args.queue_size:
            raise ValueError(
                "Resumed pattern uses a different queue size: "
                f"{saved_queue_size} vs {args.queue_size}"
            )
        saved_ema_beta = checkpoint.get("fd_ema_beta")
        if (
            saved_ema_beta is not None
            and not math.isclose(float(saved_ema_beta), args.fd_ema_beta)
        ):
            raise ValueError(
                "Resumed pattern uses a different fd_ema_beta: "
                f"{saved_ema_beta} vs {args.fd_ema_beta}"
            )
    pattern.load_state_dict(checkpoint["pattern"])
    if not args.pattern_eval_only and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = int(checkpoint.get("epoch", 0))
    step = int(checkpoint.get("step", 0))
    logger.info(
        "Resumed universal pattern from %s at epoch=%d step=%d",
        args.pattern_resume,
        start_epoch,
        step,
    )
    return start_epoch, step


def _train_pattern(
    args: argparse.Namespace,
    generator: nn.Module,
    inception: nn.Module,
    feat_dim: int,
    pattern: UniversalPattern,
    optimizer: torch.optim.Optimizer,
    *,
    start_epoch: int,
    global_step: int,
    wandb_logger=None,
) -> tuple[int, int]:
    from utils.schedule_util import adjust_learning_rate

    mu_ref, sigma_ref = load_mu_and_sigma_reference(
        args.pattern_inception_stats_path
    )
    sigma_ref_sqrt = (
        precompute_sigma_ref_sqrt(sigma_ref) if args.fd_eigvalsh else None
    )
    baseline_fid, queues = _initialize_training_queues(
        args,
        generator,
        inception,
        feat_dim,
        pattern,
        mu_ref,
        sigma_ref,
        sigma_ref_sqrt,
    )
    positive_alphas = list(args.pattern_train_alphas[1:])
    world_size = get_world_size()
    fixed_batches = list(
        _local_batches(args.queue_size, args.batch_size, world_size)
    )
    updates_since_queue_init = 0
    start_time = time.perf_counter()

    for epoch in range(start_epoch, args.epochs):
        for step_in_epoch in range(args.steps_per_epoch):
            batch_index, local_bsz = fixed_batches[
                updates_since_queue_init % len(fixed_batches)
            ]
            lr = adjust_learning_rate(optimizer, global_step, args)

            optimizer.zero_grad(set_to_none=True)
            labels = _balanced_training_labels(
                args,
                batch_index=batch_index,
                local_batch_size=local_bsz,
            )
            base = _generate_fixed_batch(
                generator,
                args,
                batch_size=local_bsz,
                base_seed=args.pattern_train_seed,
                batch_index=batch_index,
                labels=labels,
            )
            fid_values = [baseline_fid]
            fid_gradients: list[torch.Tensor | None] = [None]

            # Process alphas sequentially and immediately differentiate each
            # FID.  This avoids retaining K Inception/eigendecomposition graphs.
            for alpha in positive_alphas:
                perturbed = _to_unit_range(
                    _apply_pattern_model_space(base, pattern, alpha)
                )
                local_features = _extract_primary_features(
                    inception,
                    perturbed,
                    use_amp=False,
                    amp_dtype=args.amp_dtype,
                )
                global_features = diff_all_gather(local_features)
                mu, sigma = queues[alpha].build_feats_stats(global_features)
                fid = compute_frechet_distance_loss(
                    mu_ref,
                    sigma_ref,
                    mu=mu,
                    sigma=sigma,
                    sigma_ref_sqrt=sigma_ref_sqrt,
                )
                grad = torch.autograd.grad(fid, pattern.u, create_graph=False)[0]
                fid_values.append(float(fid.detach()))
                fid_gradients.append(grad.detach())
                queues[alpha].enqueue(global_features)
                del perturbed, local_features, global_features, mu, sigma, fid, grad

            fd_gradient, base_fd_loss, mono_loss, coefficients = _compose_fd_gradient(
                fid_values,
                fid_gradients,
                mono_weight=args.pattern_mono_weight,
                mono_margin=args.pattern_mono_margin,
                fid_norm_eps=args.fd_fid_norm_eps,
            )
            if is_enabled():
                # Each rank owns the gradient through its local feature chunk.
                # SUM reconstructs the derivative of the globally gathered FID.
                dist.all_reduce(fd_gradient, op=dist.ReduceOp.SUM)

            reg, tv, mean_sq = pattern.regularizer(args.pattern_mean_weight)
            reg_gradient = torch.autograd.grad(reg, pattern.u)[0]
            total_gradient = fd_gradient.add(
                reg_gradient, alpha=args.pattern_reg_weight
            )
            pattern.u.grad = total_gradient
            if args.pattern_grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    pattern.parameters(), args.pattern_grad_clip
                )
            else:
                grad_norm = total_gradient.norm()
            finite = bool(torch.isfinite(grad_norm))
            if finite:
                optimizer.step()
            else:
                logger.warning(
                    "[pattern] non-finite gradient at step %d; update skipped",
                    global_step,
                )

            total_loss = (
                base_fd_loss
                + args.pattern_mono_weight * mono_loss
                + args.pattern_reg_weight * float(reg.detach())
            )
            if (
                get_global_rank() == 0
                and (
                    global_step % args.print_freq == 0
                    or step_in_epoch + 1 == args.steps_per_epoch
                )
            ):
                elapsed = time.perf_counter() - start_time
                fid_text = ", ".join(
                    f"{alpha:.6g}:{value:.4f}"
                    for alpha, value in zip(args.pattern_train_alphas, fid_values)
                )
                logger.info(
                    "[pattern] epoch=%d/%d step=%d loss=%.6f "
                    "mono=%.6f tv=%.6f mean2=%.6g grad=%.6g lr=%.3g "
                    "fids={%s} coeff=%s elapsed=%.1fs",
                    epoch + 1,
                    args.epochs,
                    global_step,
                    total_loss,
                    mono_loss,
                    float(tv.detach()),
                    float(mean_sq.detach()),
                    float(grad_norm),
                    lr,
                    fid_text,
                    [round(v, 4) for v in coefficients],
                    elapsed,
                )
                if wandb_logger is not None:
                    wandb_logger.update(
                        {
                            "train/loss": total_loss,
                            "train/base_fd_loss": base_fd_loss,
                            "train/mono_loss": mono_loss,
                            "train/tv": float(tv.detach()),
                            "train/pattern_mean_sq": float(mean_sq.detach()),
                            "train/grad_norm": float(grad_norm),
                            "train/lr": lr,
                            **{
                                f"train/fid_alpha_{alpha:.8f}": value
                                for alpha, value in zip(
                                    args.pattern_train_alphas,
                                    fid_values,
                                )
                            },
                        },
                        step=global_step,
                    )
            global_step += 1
            updates_since_queue_init += 1
            del base, labels, fid_values, fid_gradients, fd_gradient, reg_gradient
            del total_gradient, reg, tv, mean_sq

        next_epoch = epoch + 1
        if (
            next_epoch % args.save_freq == 0
            or next_epoch == args.epochs
        ):
            _save_pattern_checkpoint(
                args,
                pattern,
                optimizer,
                epoch=next_epoch,
                step=global_step,
                name=f"pattern_epoch_{next_epoch:04d}.pth",
            )
            _save_pattern_checkpoint(
                args,
                pattern,
                optimizer,
                epoch=next_epoch,
                step=global_step,
                name="pattern_latest.pth",
            )
            _save_pattern_preview(args, pattern)
        if is_enabled():
            dist.barrier()

    del queues
    torch.cuda.empty_cache()
    return args.epochs, global_step


def _stats_to_fd(
    feat_sum: np.ndarray,
    feat_outer: np.ndarray,
    count: int,
    ref_mu: np.ndarray,
    ref_sigma: np.ndarray,
) -> float:
    if count < 2:
        raise ValueError("At least two features are required for FD")
    mu = feat_sum / count
    sigma = (
        feat_outer - np.outer(feat_sum, feat_sum) / count
    ) / (count - 1)
    return float(np_fid(mu, sigma, ref_mu, ref_sigma))


def _reduce_block_moments(
    accumulators: dict[str, dict[float, Moments]],
    *,
    block_count: int,
    ref_stats: dict[str, tuple[np.ndarray, np.ndarray]],
    totals: dict[str, dict[float, dict[str, np.ndarray | int]]],
) -> dict[str, dict[float, float]]:
    rank = get_global_rank()
    world_size = get_world_size()
    results: dict[str, dict[float, float]] = {
        metric: {} for metric in accumulators
    }
    global_count = block_count * world_size

    for metric, alpha_accumulators in accumulators.items():
        ref_mu, ref_sigma = ref_stats[metric]
        for alpha, moments in alpha_accumulators.items():
            if is_enabled():
                dist.reduce(moments.feat_sum, dst=0, op=dist.ReduceOp.SUM)
                dist.reduce(moments.feat_outer, dst=0, op=dist.ReduceOp.SUM)
            if rank == 0:
                feat_sum = moments.feat_sum.cpu().numpy().copy()
                feat_outer = moments.feat_outer.cpu().numpy().copy()
                total = totals[metric][alpha]
                total["sum"] += feat_sum
                total["outer"] += feat_outer
                total["count"] = int(total["count"]) + global_count
                results[metric][alpha] = _stats_to_fd(
                    feat_sum,
                    feat_outer,
                    global_count,
                    ref_mu,
                    ref_sigma,
                )
            moments.zero_()
    return results


def _linear_slope(alphas: list[float], values: list[float]) -> float:
    return float(np.polyfit(np.asarray(alphas), np.asarray(values), deg=1)[0])


def _evaluate_one_seed(
    args: argparse.Namespace,
    generator: nn.Module,
    inception: nn.Module,
    inception_dim: int,
    clip_model: nn.Module,
    clip_dim: int,
    pattern: UniversalPattern,
    *,
    eval_seed: int,
    ref_stats: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[list[dict[str, float | int]], dict[str, float], list[dict[str, float]]]:
    device = torch.device("cuda")
    rank = get_global_rank()
    world_size = get_world_size()
    alphas = list(args.pattern_eval_alphas)
    block_images = args.pattern_eval_images // args.pattern_eval_blocks
    if block_images % world_size:
        raise ValueError(
            f"eval block size={block_images} must be divisible by world_size={world_size}"
        )
    local_block_images = block_images // world_size
    accumulators = {
        "inception": {
            alpha: Moments.create(inception_dim, device) for alpha in alphas
        },
        "clip": {
            alpha: Moments.create(clip_dim, device) for alpha in alphas
        },
    }

    totals: dict[str, dict[float, dict[str, np.ndarray | int]]] = {}
    if rank == 0:
        totals = {
            "inception": {
                alpha: {
                    "sum": np.zeros(inception_dim, dtype=np.float64),
                    "outer": np.zeros(
                        (inception_dim, inception_dim), dtype=np.float64
                    ),
                    "count": 0,
                }
                for alpha in alphas
            },
            "clip": {
                alpha: {
                    "sum": np.zeros(clip_dim, dtype=np.float64),
                    "outer": np.zeros(
                        (clip_dim, clip_dim), dtype=np.float64
                    ),
                    "count": 0,
                }
                for alpha in alphas
            },
        }

    block_rows: list[dict[str, float]] = []
    started = time.perf_counter()
    for block_index in range(args.pattern_eval_blocks):
        local_seen = 0
        num_batches = math.ceil(local_block_images / args.eval_bsz)
        for batch_index in range(num_batches):
            local_bsz = min(args.eval_bsz, local_block_images - local_seen)
            global_start = (
                block_index * block_images
                + rank * local_block_images
                + local_seen
            )
            labels = (
                torch.arange(
                    global_start,
                    global_start + local_bsz,
                    device=device,
                    dtype=torch.long,
                )
                % args.num_classes
            )
            deterministic_batch_index = (
                block_index * num_batches + batch_index
            )
            base = _generate_fixed_batch(
                generator,
                args,
                batch_size=local_bsz,
                base_seed=eval_seed,
                batch_index=deterministic_batch_index,
                labels=labels,
            )
            with torch.no_grad():
                for alpha in alphas:
                    images = _to_unit_range(
                        _apply_pattern_model_space(base, pattern, alpha)
                    )
                    inc_features = _extract_primary_features(
                        inception,
                        images,
                        use_amp=False,
                        amp_dtype=args.amp_dtype,
                    )
                    clip_features = _extract_primary_features(
                        clip_model,
                        images,
                        use_amp=args.enable_amp,
                        amp_dtype=args.amp_dtype,
                    )
                    accumulators["inception"][alpha].update(inc_features)
                    accumulators["clip"][alpha].update(clip_features)
                    del images, inc_features, clip_features
            local_seen += local_bsz
            del base, labels

        block_results = _reduce_block_moments(
            accumulators,
            block_count=local_block_images,
            ref_stats=ref_stats,
            totals=totals,
        )
        if rank == 0:
            inc_values = [block_results["inception"][a] for a in alphas]
            clip_values = [block_results["clip"][a] for a in alphas]
            clip_fdr_values = [
                value / args.pattern_clip_fdr_denominator for value in clip_values
            ]
            block_row = {
                "eval_seed": float(eval_seed),
                "block": float(block_index),
                "beta_inception": _linear_slope(alphas, inc_values),
                "beta_clip": _linear_slope(alphas, clip_values),
                "beta_clip_fdr": _linear_slope(alphas, clip_fdr_values),
            }
            block_rows.append(block_row)
            logger.info(
                "[pattern eval] seed=%d block=%d/%d "
                "beta_inc=%.6g beta_clip_fdr=%.6g elapsed=%.1fs",
                eval_seed,
                block_index + 1,
                args.pattern_eval_blocks,
                block_row["beta_inception"],
                block_row["beta_clip_fdr"],
                time.perf_counter() - started,
            )
        if is_enabled():
            dist.barrier()

    rows: list[dict[str, float | int]] = []
    slopes: dict[str, float] = {}
    if rank == 0:
        final_values: dict[str, list[float]] = {"inception": [], "clip": []}
        for metric in ("inception", "clip"):
            ref_mu, ref_sigma = ref_stats[metric]
            for alpha in alphas:
                total = totals[metric][alpha]
                final_values[metric].append(
                    _stats_to_fd(
                        total["sum"],
                        total["outer"],
                        int(total["count"]),
                        ref_mu,
                        ref_sigma,
                    )
                )
        clip_fdr = [
            value / args.pattern_clip_fdr_denominator
            for value in final_values["clip"]
        ]
        for alpha, inc_fd, clip_fd, clip_ratio in zip(
            alphas,
            final_values["inception"],
            final_values["clip"],
            clip_fdr,
        ):
            rows.append(
                {
                    "eval_seed": eval_seed,
                    "alpha": alpha,
                    "inception_fid": inc_fd,
                    "clip_fd": clip_fd,
                    "clip_fdr": clip_ratio,
                    "num_images": args.pattern_eval_images,
                }
            )
        slopes = {
            "beta_inception": _linear_slope(
                alphas, final_values["inception"]
            ),
            "beta_clip": _linear_slope(alphas, final_values["clip"]),
            "beta_clip_fdr": _linear_slope(alphas, clip_fdr),
        }
        slopes["conflict"] = max(0.0, -slopes["beta_inception"]) * max(
            0.0, slopes["beta_clip_fdr"]
        )
    del accumulators, totals
    torch.cuda.empty_cache()
    return rows, slopes, block_rows


def _bootstrap_mean_ci(
    values: list[float],
    *,
    repeats: int,
    confidence: float,
    seed: int,
    point_estimate: float,
) -> list[float] | None:
    if len(values) < 2 or repeats < 1:
        return None
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=np.float64)
    indices = rng.integers(0, len(array), size=(repeats, len(array)))
    means = array[indices].mean(axis=1)
    # Center block-bootstrap fluctuations on the full-set slope.  For
    # independent-seed bootstrap the point estimate already equals the sample
    # mean, so this reduces to the ordinary percentile interval.
    means = point_estimate + (means - array.mean())
    tail = (1.0 - confidence) / 2.0
    return [
        float(np.quantile(means, tail)),
        float(np.quantile(means, 1.0 - tail)),
    ]


def _write_evaluation_outputs(
    args: argparse.Namespace,
    rows: list[dict[str, float | int]],
    seed_slopes: list[dict[str, float]],
    block_rows: list[dict[str, float]],
) -> dict[str, object]:
    if get_global_rank() != 0:
        return {}
    output_dir = Path(args.log_dir)
    csv_path = output_dir / "pattern_dose_response.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Prefer independent held-out seed slopes.  With one evaluation seed, use
    # disjoint held-out blocks as the bootstrap units.
    if len(seed_slopes) >= 2:
        ci_source = "heldout_seeds"
        ci_rows = seed_slopes
    else:
        ci_source = "heldout_blocks"
        ci_rows = block_rows

    mean_slopes: dict[str, float] = {
        key: float(np.mean([row[key] for row in seed_slopes]))
        for key in ("beta_inception", "beta_clip", "beta_clip_fdr")
    }
    mean_slopes["conflict"] = max(
        0.0, -mean_slopes["beta_inception"]
    ) * max(0.0, mean_slopes["beta_clip_fdr"])
    inc_ci = _bootstrap_mean_ci(
        [row["beta_inception"] for row in ci_rows],
        repeats=args.pattern_bootstrap_repeats,
        confidence=args.pattern_confidence,
        seed=args.pattern_bootstrap_seed,
        point_estimate=mean_slopes["beta_inception"],
    )
    clip_ci = _bootstrap_mean_ci(
        [row["beta_clip_fdr"] for row in ci_rows],
        repeats=args.pattern_bootstrap_repeats,
        confidence=args.pattern_confidence,
        seed=args.pattern_bootstrap_seed + 1,
        point_estimate=mean_slopes["beta_clip_fdr"],
    )
    summary: dict[str, object] = {
        **mean_slopes,
        "beta_inception_ci": inc_ci,
        "beta_clip_fdr_ci": clip_ci,
        "confidence": args.pattern_confidence,
        "ci_source": ci_source,
        "num_ci_units": len(ci_rows),
        "eval_seeds": list(args.pattern_eval_seeds),
        "alphas": list(args.pattern_eval_alphas),
        "success": (
            mean_slopes["beta_inception"] < 0
            and mean_slopes["beta_clip_fdr"] > 0
        ),
        "ci_sign_success": (
            inc_ci is not None
            and clip_ci is not None
            and inc_ci[1] < 0
            and clip_ci[0] > 0
        ),
        "clip_fdr_denominator": args.pattern_clip_fdr_denominator,
        "clip_model": args.pattern_clip_model,
        "inception_stats": args.pattern_inception_stats_path,
        "clip_stats": args.pattern_clip_stats_path,
        "queue_size": args.queue_size,
        "fd_ema_beta": args.fd_ema_beta,
        "global_batch_size": args.batch_size * args.world_size,
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "learning_rate": args.lr,
    }
    summary_path = output_dir / "pattern_dose_response_summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    logger.info("Dose-response CSV: %s", csv_path)
    logger.info("Dose-response summary: %s", summary_path)
    logger.info(
        "Result: beta_inc=%.6g CI=%s, beta_clip_fdr=%.6g CI=%s, "
        "conflict=%.6g success=%s",
        mean_slopes["beta_inception"],
        inc_ci,
        mean_slopes["beta_clip_fdr"],
        clip_ci,
        mean_slopes["conflict"],
        summary["success"],
    )
    return summary


def _evaluate_pattern(
    args: argparse.Namespace,
    generator: nn.Module,
    inception: nn.Module,
    inception_dim: int,
    pattern: UniversalPattern,
) -> dict[str, object]:
    # CLIP is deliberately first loaded here, after pattern optimization.
    logger.info(
        "[pattern eval] training is over; now loading frozen CLIP %s",
        args.pattern_clip_model,
    )
    clip_model, clip_dim = _load_frozen_repr(
        args.pattern_clip_model,
        target_size=args.pattern_clip_target_size,
    )
    clip_ref = np.load(args.pattern_clip_stats_path)
    inc_ref = np.load(args.pattern_inception_stats_path)
    ref_stats = {
        "inception": (
            np.asarray(inc_ref["mu"], dtype=np.float64),
            np.asarray(inc_ref["sigma"], dtype=np.float64),
        ),
        "clip": (
            np.asarray(clip_ref["mu"], dtype=np.float64),
            np.asarray(clip_ref["sigma"], dtype=np.float64),
        ),
    }
    if ref_stats["inception"][0].shape[0] != inception_dim:
        raise ValueError("Inception reference-stat dimension does not match the encoder")
    if ref_stats["clip"][0].shape[0] != clip_dim:
        raise ValueError("CLIP reference-stat dimension does not match the encoder")

    all_rows: list[dict[str, float | int]] = []
    all_seed_slopes: list[dict[str, float]] = []
    all_block_rows: list[dict[str, float]] = []
    for eval_seed in args.pattern_eval_seeds:
        rows, slopes, block_rows = _evaluate_one_seed(
            args,
            generator,
            inception,
            inception_dim,
            clip_model,
            clip_dim,
            pattern,
            eval_seed=eval_seed,
            ref_stats=ref_stats,
        )
        if get_global_rank() == 0:
            all_rows.extend(rows)
            all_seed_slopes.append(slopes)
            all_block_rows.extend(block_rows)
    del clip_model
    torch.cuda.empty_cache()
    return _write_evaluation_outputs(
        args, all_rows, all_seed_slopes, all_block_rows
    )


def train_and_evaluate(args: argparse.Namespace) -> dict[str, object]:
    from utils.setup_util import setup

    _validate_args(args)
    expected_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not args.pattern_eval_only:
        if args.queue_size % expected_world_size:
            raise ValueError(
                "--queue_size must be divisible by the torchrun world size"
            )
    wandb_logger = setup(args)
    if expected_world_size != get_world_size():
        raise RuntimeError("WORLD_SIZE changed during distributed initialization")

    if args.pattern_inception_stats_path is None:
        args.pattern_inception_stats_path = args.fid_stats_path
    if not os.path.exists(args.pattern_inception_stats_path):
        raise FileNotFoundError(
            f"Reference statistics not found: {args.pattern_inception_stats_path}"
        )
    if not args.pattern_train_only:
        if args.pattern_clip_stats_path is None:
            args.pattern_clip_stats_path = infer_stats_path(
                args.pattern_clip_model,
                args.img_size,
                args.pattern_clip_target_size,
            )
        if not os.path.exists(args.pattern_clip_stats_path):
            raise FileNotFoundError(
                f"Reference statistics not found: {args.pattern_clip_stats_path}"
            )

    generator = _load_frozen_pmf(args)
    inception, inception_dim = _load_frozen_repr(
        "inception", target_size=299
    )
    pattern = UniversalPattern(args.pattern_size).cuda()
    broadcast_module_params(pattern, src=0)
    optimizer = torch.optim.AdamW(
        [pattern.u],
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=0.0,
    )
    if args.pattern_resume is None and args.auto_resume:
        auto_resume_path = Path(args.ckpt_dir) / "pattern_latest.pth"
        if auto_resume_path.exists():
            args.pattern_resume = str(auto_resume_path)
    start_epoch, global_step = _load_pattern_resume(
        args, pattern, optimizer
    )
    broadcast_module_params(pattern, src=0)

    trainable = [
        (name, parameter.numel())
        for name, parameter in pattern.named_parameters()
        if parameter.requires_grad
    ]
    logger.info(
        "Trainable parameters (the complete list): %s; total=%d",
        trainable,
        sum(count for _, count in trainable),
    )

    if not args.pattern_eval_only:
        _, global_step = _train_pattern(
            args,
            generator,
            inception,
            inception_dim,
            pattern,
            optimizer,
            start_epoch=start_epoch,
            global_step=global_step,
            wandb_logger=wandb_logger,
        )
    _save_pattern_preview(args, pattern, name="pattern_final.png")

    summary: dict[str, object] = {}
    if not args.pattern_train_only:
        summary = _evaluate_pattern(
            args, generator, inception, inception_dim, pattern
        )
        if wandb_logger is not None:
            wandb_logger.update(
                {
                    f"eval/{key}": value
                    for key, value in summary.items()
                    if isinstance(value, (int, float, bool))
                },
                step=global_step,
            )
    if is_enabled():
        dist.barrier()
    if wandb_logger is not None:
        wandb_logger.finish()
    return summary


def get_args_parser() -> argparse.ArgumentParser:
    from main_fd import get_args_parser as get_parent_parser

    parser = argparse.ArgumentParser(
        "Frozen-pMF universal dose-response pattern",
        parents=[get_parent_parser()],
        add_help=True,
        conflict_handler="resolve",
    )
    parser.add_argument("--pattern_size", type=int, default=16)
    parser.add_argument(
        "--pattern_train_alphas",
        type=float,
        nargs="+",
        default=[0.0, 2 / 255, 4 / 255, 6 / 255, 8 / 255],
        help="alpha values in pMF's [-1,1] image space",
    )
    parser.add_argument(
        "--pattern_eval_alphas",
        type=float,
        nargs="+",
        default=[i / 255 for i in range(9)],
        help="held-out dose-response alpha grid in pMF's [-1,1] image space",
    )
    parser.add_argument("--pattern_eval_images", type=int, default=50_000)
    parser.add_argument("--pattern_train_seed", type=int, default=12_345)
    parser.add_argument(
        "--pattern_eval_seeds", type=int, nargs="+", default=[54_321]
    )
    parser.add_argument("--pattern_mono_weight", type=float, default=1.0)
    parser.add_argument("--pattern_mono_margin", type=float, default=0.01)
    parser.add_argument("--pattern_reg_weight", type=float, default=1e-3)
    parser.add_argument("--pattern_mean_weight", type=float, default=10.0)
    parser.add_argument("--pattern_grad_clip", type=float, default=0.0)
    parser.add_argument("--pattern_resume", type=str, default=None)
    parser.add_argument("--pattern_eval_only", action="store_true")
    parser.add_argument("--pattern_train_only", action="store_true")
    parser.add_argument(
        "--pattern_inception_stats_path",
        type=str,
        default=None,
        help="real Inception statistics; defaults to --fid_stats_path",
    )
    parser.add_argument(
        "--pattern_clip_model",
        type=str,
        default="vit_large_patch14_clip_224.openai",
    )
    parser.add_argument("--pattern_clip_target_size", type=int, default=256)
    parser.add_argument("--pattern_clip_stats_path", type=str, default=None)
    parser.add_argument(
        "--pattern_clip_fdr_denominator",
        type=float,
        default=5.60,
        help="ImageNet validation FD used for repository-standard FDr-CLIP",
    )
    parser.add_argument(
        "--pattern_eval_blocks",
        type=int,
        default=10,
        help="disjoint held-out blocks used as fallback bootstrap units",
    )
    parser.add_argument("--pattern_bootstrap_repeats", type=int, default=10_000)
    parser.add_argument("--pattern_bootstrap_seed", type=int, default=20_21)
    parser.add_argument("--pattern_confidence", type=float, default=0.95)
    return parser


def _cleanup_distributed() -> None:
    if is_enabled():
        dist.destroy_process_group()


if __name__ == "__main__":
    parsed_args = get_args_parser().parse_args()
    try:
        train_and_evaluate(parsed_args)
    finally:
        _cleanup_distributed()
