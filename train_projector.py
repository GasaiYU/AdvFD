import argparse
import datetime
import json
import logging
import math
import os
import sys
import time
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from frechet_distance.judges import infer_stats_path
from frechet_distance.losses import (
    compute_frechet_distance_loss,
    diff_all_gather,
    load_mu_and_sigma_reference,
    precompute_sigma_ref_sqrt,
)
from frechet_distance.queue import FeatureQueue
from frechet_distance.repr_models import (
    inception_feature_layer_from_name,
    load_repr_model,
    model_short_name,
)
from models.projector import (
    ViTSMultiHeadProjector,
    canonical_projector_backbone,
    count_trainable_parameters,
    projector_backbone_config,
)
from projector.data import build_projector_train_loader
from utils.distributed_util import (
    all_reduce_mean,
    enable_distributed,
    get_global_rank,
    get_local_rank,
    get_world_size,
    is_main_process,
)
from utils.grad_util import get_grad_norm
from utils.logging_util import MetricLogger, SmoothedValue, setup_logging, setup_wandb
from utils.rng_util import fix_random_seeds


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

logger = logging.getLogger("FD_loss")

HEAD_VARIANTS = ("default", "siglip_deep_inception_conv")


DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def apply_projector_backbone_defaults(args) -> None:
    args.backbone = canonical_projector_backbone(args.backbone)
    config = projector_backbone_config(args.backbone)
    for key in ("patch_size", "embed_dim", "depth", "num_heads", "head_hidden_dim"):
        if getattr(args, key) is None:
            setattr(args, key, config[key])


def expand_arg(values, n: int, default):
    if values is None:
        return [default for _ in range(n)]
    if len(values) == 1 and n > 1:
        return list(values) * n
    if len(values) != n:
        raise ValueError("Expected 1 or %d values, got %d" % (n, len(values)))
    return list(values)


def unique_name(name: str, used: Dict[str, int]) -> str:
    if name not in used:
        used[name] = 1
        return name
    used[name] += 1
    return "%s_%d" % (name, used[name])


def teacher_role(model_name: str, inception_layer: Optional[str]) -> str:
    if inception_layer is not None:
        return "inception"
    lower = model_name.lower()
    if "siglip" in lower:
        return "siglip"
    if "mae" in lower:
        return "mae"
    raise ValueError(
        "Unsupported projector teacher '%s'. Expected Inception, SigLIP, or MAE."
        % model_name
    )


def resolve_teacher_specs(args):
    teacher_models = list(args.teacher_models)
    n = len(teacher_models)
    target_sizes = expand_arg(args.teacher_target_sizes, n, None)
    head_weights = expand_arg(args.head_weights, n, 1.0)

    specs = []
    used = {}
    for model_name, target_size, weight in zip(teacher_models, target_sizes, head_weights):
        base_name = model_short_name(model_name)
        head_name = unique_name(base_name, used)
        inception_layer = inception_feature_layer_from_name(model_name)
        specs.append({
            "model_name": model_name,
            "head_name": head_name,
            "target_size": target_size,
            "weight": float(weight),
            "role": teacher_role(model_name, inception_layer),
        })
    return specs


def load_teachers(specs, device: str):
    teachers = []
    for spec in specs:
        model, feat_dim, _, target_size = load_repr_model(
            spec["model_name"],
            device=device,
            target_size=spec["target_size"],
            grad_checkpointing=False,
        )
        model.eval().requires_grad_(False)
        teacher = dict(spec)
        teacher.update({
            "model": model,
            "feat_dim": feat_dim,
            "inception_layer": inception_feature_layer_from_name(spec["model_name"]),
            "resolved_target_size": target_size,
        })
        teachers.append(teacher)
        logger.info(
            "[projector] teacher head=%s model=%s role=%s feat_dim=%d target_size=%s weight=%.4g",
            teacher["head_name"],
            teacher["model_name"],
            teacher["role"],
            feat_dim,
            str(target_size),
            teacher["weight"],
        )
    return teachers


@torch.no_grad()
def extract_teacher_targets(teachers, images: torch.Tensor, teacher_dtype: torch.dtype,
                            use_amp: bool, projector_version: str):
    targets = {}
    for teacher in teachers:
        model = teacher["model"]
        amp_enabled = use_amp and not getattr(model, "is_inception", False)
        with torch.autocast("cuda", enabled=amp_enabled, dtype=teacher_dtype):
            feat = _extract_teacher_target(teacher, images, projector_version)

        if feat.ndim != 3:
            raise RuntimeError(
                "Teacher '%s' returned shape %s, expected [B, tokens, channels]"
                % (teacher["model_name"], tuple(feat.shape))
            )
        targets[teacher["head_name"]] = feat.detach().float()
    return targets


def _extract_teacher_target(teacher, images: torch.Tensor, projector_version: str) -> torch.Tensor:
    role = teacher["role"]
    model = teacher["model"]

    if role == "inception":
        if hasattr(model, "forward_feature_maps_by_layer"):
            fmap = model.forward_feature_maps_by_layer(
                images,
                [teacher["inception_layer"]],
            )[teacher["inception_layer"]]
            if projector_version == "III":
                return F.adaptive_avg_pool2d(fmap, 1).flatten(1).float().unsqueeze(1)
            tokens = fmap.flatten(2).transpose(1, 2).float()
            return resize_token_grid(tokens, (16, 16))

        if projector_version == "III":
            if not hasattr(model, "forward_features_by_layer"):
                raise RuntimeError("Inception teacher does not provide pooled features")
            primary, _ = model.forward_features_by_layer(
                images,
                [teacher["inception_layer"]],
            )[teacher["inception_layer"]]
            return primary.unsqueeze(1)
        if not hasattr(model, "forward_tokens_by_layer"):
            raise RuntimeError("Inception teacher does not provide token features")
        tokens = model.forward_tokens_by_layer(images, [teacher["inception_layer"]])[
            teacher["inception_layer"]
        ]
        return resize_token_grid(tokens, (16, 16))

    if role == "siglip":
        primary, _ = model(images)
        query = primary.unsqueeze(1)
        if projector_version in ("II", "III"):
            return query
        tokens = model.forward_tokens(images)
        return torch.cat([query, patch_tokens_from_teacher(model, tokens)], dim=1)

    if role == "mae":
        tokens = model.forward_tokens(images)
        prefix = int(getattr(model, "num_prefix_tokens", 0))
        if prefix > 0 and prefix < tokens.shape[1]:
            primary = tokens[:, :1]
            patches = tokens[:, prefix:]
        else:
            primary = tokens.mean(1, keepdim=True)
            patches = tokens
        if projector_version in ("II", "III"):
            return primary
        return torch.cat([primary, patches], dim=1)

    raise NotImplementedError(role)


def patch_tokens_from_teacher(model, tokens: torch.Tensor) -> torch.Tensor:
    prefix = int(getattr(model, "num_prefix_tokens", 0))
    if prefix >= tokens.shape[1]:
        prefix = 0
    return tokens[:, prefix:]


def square_grid(num_tokens: int, label: str) -> Tuple[int, int]:
    grid = int(math.sqrt(num_tokens))
    if grid * grid != num_tokens:
        raise RuntimeError(
            "Cannot infer square token grid for %s: tokens=%d"
            % (label, num_tokens)
        )
    return grid, grid


def resize_token_grid(tokens: torch.Tensor, target_grid: Tuple[int, int]) -> torch.Tensor:
    src_h, src_w = square_grid(tokens.shape[1], "teacher tokens")
    if (src_h, src_w) == target_grid:
        return tokens.float()
    bsz, _, dim = tokens.shape
    tokens = tokens.reshape(bsz, src_h, src_w, dim).permute(0, 3, 1, 2)
    tokens = F.interpolate(tokens, size=target_grid, mode="bicubic", align_corners=False)
    tgt_h, tgt_w = target_grid
    return tokens.permute(0, 2, 3, 1).reshape(bsz, tgt_h * tgt_w, dim).float()


def source_mode_for_teacher(teacher, projector_version: str, head_variant: str = "default") -> str:
    role = teacher["role"]
    if role == "inception":
        if head_variant == "siglip_deep_inception_conv":
            return "patches"
        if projector_version == "III":
            return "prefix2"
        return "patches"
    if projector_version == "I":
        return "prefix0_patches"
    if role == "siglip":
        return "prefix0"
    if role == "mae":
        return "prefix1"
    raise NotImplementedError(role)


def infer_teacher_target_layout(
    teacher,
    target: torch.Tensor,
    projector_version: str,
    head_variant: str = "default",
):
    num_tokens, token_dim = target.shape[1], target.shape[2]
    teacher["token_dim"] = token_dim
    teacher["num_tokens"] = num_tokens
    teacher["source_mode"] = source_mode_for_teacher(teacher, projector_version, head_variant)
    if teacher["source_mode"] == "patches":
        teacher["target_grid"] = square_grid(num_tokens, teacher["head_name"])
    elif teacher["source_mode"] == "prefix0_patches":
        teacher["target_grid"] = square_grid(num_tokens - 1, teacher["head_name"])
    else:
        teacher["target_grid"] = (1, 1)
    logger.info(
        "[projector] target head=%s role=%s tokens=%d dim=%d source=%s grid=%dx%d",
        teacher["head_name"],
        teacher["role"],
        num_tokens,
        token_dim,
        teacher["source_mode"],
        teacher["target_grid"][0],
        teacher["target_grid"][1],
    )


def resolve_projector_head_config(args, teachers):
    head_types = {}
    head_mlp_layer_overrides = {}
    if args.head_variant == "default":
        return head_types, head_mlp_layer_overrides
    if args.head_variant != "siglip_deep_inception_conv":
        raise NotImplementedError(args.head_variant)

    for teacher in teachers:
        name = teacher["head_name"]
        if teacher["role"] == "inception":
            head_types[name] = "conv"
            head_mlp_layer_overrides[name] = args.head_conv_layers
        elif teacher["role"] == "siglip":
            head_mlp_layer_overrides[name] = args.head_mlp_layers + 2
    return head_types, head_mlp_layer_overrides


def projector_tokens_to_fd_feature(role: str, tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim == 2:
        return tokens.float()
    if tokens.ndim != 3:
        raise RuntimeError(
            "Projector role '%s' returned shape %s, expected [B,C] or [B,T,C]"
            % (role, tuple(tokens.shape))
        )
    if role == "inception" and tokens.shape[1] > 1:
        return tokens.mean(dim=1).float()
    return tokens[:, 0].float()


def compute_head_frechet_loss(
    pred: torch.Tensor,
    fd_state,
    args,
):
    pred_feat = projector_tokens_to_fd_feature(fd_state["role"], pred).float()
    all_feats = diff_all_gather(pred_feat.contiguous())
    queue = fd_state["queue"]

    if queue.ema_stats or queue.online_accum:
        mu, sigma = queue.build_feats_stats(all_feats)
        fid = compute_frechet_distance_loss(
            fd_state["mu_ref"],
            fd_state["sigma_ref"],
            mu=mu,
            sigma=sigma,
            sigma_ref_sqrt=fd_state.get("sigma_ref_sqrt"),
        )
    else:
        fid = compute_frechet_distance_loss(
            fd_state["mu_ref"],
            fd_state["sigma_ref"],
            all_feats=queue.build_feats_snapshot(all_feats),
            sigma_ref_sqrt=fd_state.get("sigma_ref_sqrt"),
        )

    fd_loss = fid / (fid.detach() + args.fd_fid_norm_eps)

    return fd_loss, all_feats, {
        "fd_fid": fid.detach(),
        "fd_loss": fd_loss.detach(),
        "fd_feature_pred_norm": pred_feat.norm(dim=1).mean().detach(),
    }


def compute_head_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    args,
    *,
    fd_state=None,
):
    with torch.autocast("cuda", enabled=False):
        pred = pred.float()
        target = target.float()
        if pred.shape != target.shape:
            raise RuntimeError(
                "Projector prediction shape %s does not match target shape %s"
                % (tuple(pred.shape), tuple(target.shape))
            )
        mse = F.mse_loss(pred, target)
        rel_mse = mse / target.pow(2).mean().detach().clamp_min(args.loss_eps)
        pred_flat = pred.reshape(-1, pred.shape[-1])
        target_flat = target.reshape(-1, target.shape[-1])
        cosine_sim = F.cosine_similarity(pred_flat, target_flat, dim=1, eps=args.loss_eps).mean()
        cosine = 1.0 - cosine_sim

        if args.loss_type == "mse":
            base_loss = mse
        elif args.loss_type == "relative_mse":
            base_loss = rel_mse
        elif args.loss_type == "cosine":
            base_loss = cosine
        elif args.loss_type == "relative_mse_cosine":
            base_loss = rel_mse + args.cosine_weight * cosine
        else:
            raise NotImplementedError(args.loss_type)

        if args.fd_loss_weight > 0.0:
            if fd_state is None:
                raise RuntimeError("fd_state is required when fd_loss_weight > 0")
            fd_loss, fd_new_feats, fd_logs = compute_head_frechet_loss(
                pred, fd_state, args
            )
            loss = base_loss + args.fd_loss_weight * fd_loss
        else:
            fd_loss = pred.new_zeros(())
            fd_new_feats = None
            fd_logs = {
                "fd_fid": pred.new_zeros(()),
                "fd_loss": pred.new_zeros(()),
                "fd_feature_pred_norm": pred.new_zeros(()),
            }
            loss = base_loss

        logs = {
            "base_loss": base_loss.detach(),
            "mse": mse.detach(),
            "relative_mse": rel_mse.detach(),
            "cosine": cosine.detach(),
            "cosine_sim": cosine_sim.detach(),
            "fd_weighted": (args.fd_loss_weight * fd_loss).detach(),
            "target_norm": target_flat.norm(dim=1).mean().detach(),
            "pred_norm": pred_flat.norm(dim=1).mean().detach(),
        }
        logs.update(fd_logs)
        return loss, logs, fd_new_feats


def build_fd_feature_states(args, teachers):
    if args.fd_loss_weight <= 0.0:
        return {}

    stats_paths = expand_arg(args.fd_repr_stats_paths, len(teachers), None)
    fd_states = {}
    for teacher, stats_path in zip(teachers, stats_paths):
        if stats_path is None:
            stats_path = infer_stats_path(
                teacher["model_name"],
                args.img_size,
                teacher["resolved_target_size"],
                args.fid_stats_path,
            )
        mu_ref, sigma_ref = load_mu_and_sigma_reference(stats_path, pool_type="cls")
        feat_dim = int(mu_ref.shape[0])
        if feat_dim != int(teacher["token_dim"]):
            raise RuntimeError(
                "FD stats dim %d does not match head '%s' dim %d: %s"
                % (feat_dim, teacher["head_name"], teacher["token_dim"], stats_path)
            )
        queue = FeatureQueue(
            size=args.fd_queue_size,
            feat_dim=feat_dim,
            online_accum=args.fd_online_accum,
            ema_beta=args.fd_ema_beta,
        ).cuda()
        sigma_ref_sqrt = precompute_sigma_ref_sqrt(sigma_ref) if args.fd_eigvalsh else None
        fd_states[teacher["head_name"]] = {
            "head_name": teacher["head_name"],
            "role": teacher["role"],
            "stats_path": stats_path,
            "feat_dim": feat_dim,
            "mu_ref": mu_ref,
            "sigma_ref": sigma_ref,
            "sigma_ref_sqrt": sigma_ref_sqrt,
            "queue": queue,
        }
        stats_mode = (
            "ema(beta=%.6g)" % args.fd_ema_beta
            if args.fd_ema_beta > 0.0
            else ("online_accum" if args.fd_online_accum else "snapshot")
        )
        eig_mode = "eigvalsh" if args.fd_eigvalsh else "eigvals"
        logger.info(
            "[projector] FD head=%s role=%s dim=%d stats=%s queue=%d mode=%s eig=%s weight=%.4g",
            teacher["head_name"],
            teacher["role"],
            feat_dim,
            stats_path,
            args.fd_queue_size,
            stats_mode,
            eig_mode,
            args.fd_loss_weight,
        )
    return fd_states


@torch.no_grad()
def initialize_fd_feature_queues(args, model, fd_states, loader, sampler=None):
    if not fd_states or args.fd_queue_size <= 0:
        return

    was_training = model.training
    model.eval()
    data_iter = iter(loader)
    filled = 0
    logger.info("[projector] initializing FD fake queues with projector features")
    while filled < args.fd_queue_size:
        try:
            images, _ = next(data_iter)
        except StopIteration:
            if sampler is not None:
                sampler.set_epoch(0)
            data_iter = iter(loader)
            images, _ = next(data_iter)

        images = images.cuda(non_blocking=True)
        preds = model(images)
        count = None
        for name, fd_state in fd_states.items():
            feats = projector_tokens_to_fd_feature(fd_state["role"], preds[name])
            all_feats = diff_all_gather(feats.contiguous())
            if count is None:
                count = min(all_feats.shape[0], args.fd_queue_size - filled)
            q = fd_state["queue"]
            if q.ema_stats:
                q.accumulate_batch(all_feats[:count])
            else:
                q.feats[filled:filled + count] = all_feats[:count].float()

        filled += int(count or 0)
        if is_main_process() and (filled == args.fd_queue_size or filled % max(args.global_bsz * 20, 1) == 0):
            logger.info(
                "[projector] FD queue init: %d/%d (%.1f%%)",
                filled,
                args.fd_queue_size,
                100.0 * filled / args.fd_queue_size,
            )

    for fd_state in fd_states.values():
        q = fd_state["queue"]
        if q.ema_stats:
            q._finalize_streaming_init()
        else:
            q.ptr.zero_()
            if q.online_accum:
                q._init_accumulators()

    if was_training:
        model.train()
    logger.info("[projector] FD fake queues initialized")


def fd_queue_state_dict(fd_states):
    return {
        name: state["queue"].state_dict()
        for name, state in fd_states.items()
    }


def load_fd_queue_state_dict(fd_states, state_dict):
    if not fd_states or not state_dict:
        return False
    loaded = 0
    for name, fd_state in fd_states.items():
        if name not in state_dict:
            logger.warning("[projector] no saved FD queue state for head '%s'", name)
            continue
        fd_state["queue"].load_state_dict(state_dict[name])
        fd_state["queue"].cuda()
        loaded += 1
    if loaded:
        logger.info("[projector] restored FD queue states: %d/%d", loaded, len(fd_states))
    return loaded == len(fd_states)


def create_optimizer(args, model):
    from timm.optim import create_optimizer_v2

    opt_name = "muon" if args.use_muon else "adamw"
    if args.use_muon:
        params = create_muon_param_groups(args, model)
        kwargs = {
            "opt": opt_name,
            "lr": args.muon_lr,
            "weight_decay": args.muon_weight_decay,
            "momentum": args.muon_momentum,
            "betas": (args.beta1, args.beta2),
            "nesterov": True,
            "filter_bias_and_bn": False,
        }
    else:
        params = model
        kwargs = {
            "opt": opt_name,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "filter_bias_and_bn": True,
        }
        kwargs["betas"] = (args.beta1, args.beta2)
    logger.info(
        "[projector] optimizer via timm: opt=%s lr=%.6g weight_decay=%.6g",
        opt_name,
        kwargs["lr"],
        kwargs["weight_decay"],
    )

    try:
        optimizer = create_optimizer_v2(params, **kwargs)
    except (AssertionError, ValueError) as exc:
        if args.use_muon:
            raise RuntimeError(
                "USE_MUON=1 requested timm optimizer 'muon', but the installed timm "
                "does not support it. Upgrade timm to a version that provides "
                "timm.optim.create_optimizer_v2(..., opt='muon'), or run with USE_MUON=0."
            ) from exc
        raise

    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])
    return optimizer


def _is_muon_hidden_weight(name: str, param: torch.nn.Parameter) -> bool:
    if param.ndim != 2 or not name.startswith("blocks."):
        return False
    if ".attn." in name:
        return name.endswith((
            "q_proj.weight",
            "k_proj.weight",
            "v_proj.weight",
            "out_proj.weight",
        ))
    if ".mlp." in name:
        return name.endswith(("fc1.weight", "fc2.weight"))
    return False


def _is_projector_no_decay(name: str, param: torch.nn.Parameter, no_weight_decay_names: set) -> bool:
    return (
        param.ndim < 2
        or name.endswith(".bias")
        or "norm" in name
        or name in no_weight_decay_names
    )


def create_muon_param_groups(args, model):
    no_weight_decay_names = (
        set(model.no_weight_decay()) if hasattr(model, "no_weight_decay") else set()
    )
    muon_params, muon_names = [], []
    adamw_decay_params, adamw_decay_names = [], []
    adamw_nodecay_params, adamw_nodecay_names = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _is_muon_hidden_weight(name, param):
            muon_params.append(param)
            muon_names.append(name)
        elif _is_projector_no_decay(name, param, no_weight_decay_names):
            adamw_nodecay_params.append(param)
            adamw_nodecay_names.append(name)
        else:
            adamw_decay_params.append(param)
            adamw_decay_names.append(name)

    groups = []
    if muon_params:
        groups.append({
            "params": muon_params,
            "lr": args.muon_lr,
            "weight_decay": args.muon_weight_decay,
            "nesterov": True,
            "use_fallback": False,
            "use_muon": True,
        })
    if adamw_decay_params:
        groups.append({
            "params": adamw_decay_params,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "nesterov": False,
            "use_fallback": True,
            "use_muon": False,
        })
    if adamw_nodecay_params:
        groups.append({
            "params": adamw_nodecay_params,
            "lr": args.lr,
            "weight_decay": 0.0,
            "nesterov": False,
            "use_fallback": True,
            "use_muon": False,
        })

    counts = {
        "muon_hidden": sum(p.numel() for p in muon_params),
        "adamw_decay": sum(p.numel() for p in adamw_decay_params),
        "adamw_nodecay": sum(p.numel() for p in adamw_nodecay_params),
    }
    total = sum(counts.values())
    logger.info(
        "[projector] timm Muon groups: muon_lr=%.6g, fallback_adamw_lr=%.6g, "
        "muon_wd=%.6g, fallback_adamw_wd=%.6g, "
        "muon_nesterov=True, fallback_nesterov=False",
        args.muon_lr,
        args.lr,
        args.muon_weight_decay,
        args.weight_decay,
    )
    for label, count in counts.items():
        pct = 100.0 * count / total if total else 0.0
        logger.info("[projector] %s: %d params (%.2f%%)", label, count, pct)
    attn_muon = [name for name in muon_names if ".attn." in name]
    logger.info(
        "[projector] Muon attention matrices: %d "
        "(q_proj/k_proj/v_proj/out_proj are separate)",
        len(attn_muon),
    )
    if attn_muon:
        logger.info("[projector] first Muon attention matrices: %s", ", ".join(attn_muon[:6]))
    if adamw_decay_names:
        logger.info("[projector] first AdamW decay tensors: %s", ", ".join(adamw_decay_names[:6]))
    return groups


def adjust_learning_rate(optimizer, step: int, args) -> float:
    if step < args.warmup_steps:
        lr = args.lr * float(step + 1) / max(1, args.warmup_steps)
    elif args.lr_sched == "constant":
        lr = args.lr
    else:
        progress = (step - args.warmup_steps) / max(1, args.total_steps - args.warmup_steps)
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        if args.use_muon:
            group["lr"] = group.get("initial_lr", group["lr"]) * (lr / args.lr)
        else:
            group["lr"] = lr
    return lr


def unwrap_model(model):
    if isinstance(model, DistributedDataParallel):
        model = model.module
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


def strip_state_prefix(state_dict, prefix: str):
    if not any(k.startswith(prefix) for k in state_dict):
        return state_dict
    return {
        (k[len(prefix):] if k.startswith(prefix) else k): v
        for k, v in state_dict.items()
    }


def save_checkpoint(args, model, optimizer, scaler, step: int, elapsed: float,
                    fd_states=None):
    if not is_main_process():
        return
    os.makedirs(args.ckpt_dir, exist_ok=True)
    payload = {
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "args": vars(args),
        "step": step,
        "samples_seen": args.samples_seen,
        "elapsed": elapsed,
    }
    if fd_states:
        payload["fd_queue_states"] = fd_queue_state_dict(fd_states)
    last_path = os.path.join(args.ckpt_dir, "last.pt")
    step_path = os.path.join(args.ckpt_dir, "step_%07d.pt" % step)
    torch.save(payload, last_path)
    torch.save(payload, step_path)
    logger.info("[projector] saved checkpoint: %s", step_path)


def resume_checkpoint(args, model, optimizer=None, scaler=None, fd_states=None):
    path = args.resume_from
    if path is None and args.auto_resume:
        candidate = os.path.join(args.ckpt_dir, "last.pt")
        if os.path.exists(candidate):
            path = candidate
    if path is None:
        return False

    logger.info("[projector] loading checkpoint: %s", path)
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model"]
    state = strip_state_prefix(state, "module.")
    state = strip_state_prefix(state, "_orig_mod.")
    unwrap_model(model).load_state_dict(state, strict=True)
    if optimizer is not None and "optimizer" in ckpt and ckpt["optimizer"] is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    args.current_step = int(ckpt.get("step", 0))
    args.samples_seen = int(ckpt.get("samples_seen", 0))
    args.last_elapsed_time = float(ckpt.get("elapsed", 0.0))
    fd_queues_restored = load_fd_queue_state_dict(
        fd_states,
        ckpt.get("fd_queue_states"),
    )
    logger.info("[projector] resumed from step %d", args.current_step)
    return fd_queues_restored


def setup_experiment(args):
    enable_distributed()
    args.rank = get_global_rank()
    args.local_rank = get_local_rank()
    args.world_size = get_world_size()
    args.global_bsz = args.batch_size * args.world_size
    fix_random_seeds(args.seed + args.rank)

    if args.exp_name is None:
        args.exp_name = datetime.datetime.now().strftime("%Y%m%d_%H%M_projector")
    args.log_dir = os.path.join(args.output_dir, args.project, args.exp_name)
    args.ckpt_dir = os.path.join(args.log_dir, "checkpoints")
    if is_main_process():
        os.makedirs(args.ckpt_dir, exist_ok=True)
        setup_logging(args.log_dir)
        logger.info("[projector] logging to %s", args.log_dir)
        logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
        with open(os.path.join(args.log_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4, sort_keys=True)
    args.amp_dtype = DTYPE_MAP[args.dtype]
    args.teacher_amp_dtype = DTYPE_MAP[args.teacher_dtype]

    wandb_logger = None
    if is_main_process() and args.enable_wandb:
        wandb_logger = setup_wandb(args, args.entity, args.project, args.exp_name, args.log_dir)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return wandb_logger


def train(args):
    apply_projector_backbone_defaults(args)
    wandb_logger = setup_experiment(args)
    device = "cuda"

    loader, sampler = build_projector_train_loader(
        args.data_path,
        args.img_size,
        args.batch_size,
        args.num_workers,
        args.pin_mem,
        world_size=args.world_size,
        rank=args.rank,
        subset_size=args.subset_size,
    )
    if args.steps_per_epoch <= 0:
        args.steps_per_epoch = len(loader)
    args.total_steps = args.epochs * args.steps_per_epoch
    args.warmup_steps = int(args.warmup_epochs * args.steps_per_epoch)

    specs = resolve_teacher_specs(args)
    teachers = load_teachers(specs, device=device)
    dummy_images = torch.zeros(1, 3, args.img_size, args.img_size, device=device)
    dummy_targets = extract_teacher_targets(
        teachers,
        dummy_images,
        teacher_dtype=args.teacher_amp_dtype,
        use_amp=args.teacher_dtype != "fp32",
        projector_version=args.projector_version,
    )
    for teacher in teachers:
        infer_teacher_target_layout(
            teacher,
            dummy_targets[teacher["head_name"]],
            args.projector_version,
            args.head_variant,
        )
    del dummy_images, dummy_targets

    head_dims = {teacher["head_name"]: teacher["token_dim"] for teacher in teachers}
    head_target_grids = {teacher["head_name"]: teacher["target_grid"] for teacher in teachers}
    head_source_modes = {teacher["head_name"]: teacher["source_mode"] for teacher in teachers}
    head_types, head_mlp_layer_overrides = resolve_projector_head_config(args, teachers)
    fd_states = build_fd_feature_states(args, teachers)

    model = ViTSMultiHeadProjector(
        head_dims=head_dims,
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_chans=3,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        projector_version=args.projector_version,
        head_source_modes=head_source_modes,
        head_types=head_types,
        head_hidden_dim=args.head_hidden_dim,
        head_mlp_layers=args.head_mlp_layers,
        head_mlp_layer_overrides=head_mlp_layer_overrides,
        head_conv_kernel_size=args.head_conv_kernel_size,
        head_dropout=args.head_dropout,
        head_target_grids=head_target_grids,
        normalize_input=not args.no_input_normalize,
        grad_checkpointing=args.grad_checkpointing,
    ).to(device)
    backbone_label = args.backbone.replace("_", "-").upper()
    logger.info(
        "[projector] %s projector version=%s params: %.3fM, heads=%s",
        backbone_label,
        args.projector_version,
        count_trainable_parameters(model) / 1e6,
        ", ".join("%s:%d" % (k, v) for k, v in head_dims.items()),
    )
    logger.info(
        "[projector] head variant=%s config=%s",
        args.head_variant,
        ", ".join(
            "%s:type=%s,layers=%d,source=%s,target_grid=%dx%d"
            % (
                teacher["head_name"],
                head_types.get(teacher["head_name"], "mlp"),
                head_mlp_layer_overrides.get(teacher["head_name"], args.head_mlp_layers),
                head_source_modes[teacher["head_name"]],
                head_target_grids[teacher["head_name"]][0],
                head_target_grids[teacher["head_name"]][1],
            )
            for teacher in teachers
        ),
    )

    if args.compile:
        model = torch.compile(model)

    optimizer = create_optimizer(args, model)
    scaler = torch.cuda.amp.GradScaler(enabled=args.dtype == "fp16")
    fd_queues_restored = resume_checkpoint(args, model, optimizer, scaler, fd_states)

    if args.world_size > 1:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False)

    if fd_states and not fd_queues_restored:
        initialize_fd_feature_queues(args, model, fd_states, loader, sampler=sampler)

    metric_file = os.path.join(args.log_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metric_file)
    for name, window, fmt in [
        ("lr", 1, "{value:.6f}"),
        ("samples/s/device", args.print_freq, "{avg:.2f}"),
        ("samples/s", args.print_freq, "{avg:.2f}"),
        ("samples_seen(M)", args.print_freq, "{value:.2f}"),
        ("device_mem(GB)", args.print_freq, "{value:.2f}"),
    ]:
        metric_logger.add_meter(name, SmoothedValue(window, fmt))

    start_time = time.time()
    step_start = time.perf_counter()
    data_iter = iter(loader)
    logger.info(
        "[projector] training from step %d to %d, global_bsz=%d",
        args.current_step,
        args.total_steps,
        args.global_bsz,
    )

    for step in range(args.current_step, args.total_steps):
        model.train()
        if sampler is not None and step % args.steps_per_epoch == 0:
            sampler.set_epoch(step // args.steps_per_epoch)

        try:
            images, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            images, _ = next(data_iter)

        images = images.cuda(non_blocking=True)
        lr = adjust_learning_rate(optimizer, step, args)

        targets = extract_teacher_targets(
            teachers,
            images,
            teacher_dtype=args.teacher_amp_dtype,
            use_amp=args.teacher_dtype != "fp32",
            projector_version=args.projector_version,
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=args.dtype != "fp32", dtype=args.amp_dtype):
            preds = model(images)
            total_loss = torch.zeros((), device=images.device, dtype=torch.float32)
            log_tensors = {}
            fd_queue_updates = {}
            for teacher in teachers:
                name = teacher["head_name"]
                head_loss, head_logs, fd_new_feats = compute_head_loss(
                    preds[name],
                    targets[name],
                    args,
                    fd_state=fd_states.get(name),
                )
                total_loss = total_loss + teacher["weight"] * head_loss
                log_tensors["loss_%s" % name] = head_loss.detach()
                for key, value in head_logs.items():
                    log_tensors["%s_%s" % (key, name)] = value
                if fd_new_feats is not None:
                    fd_queue_updates[name] = fd_new_feats.detach()

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = (
            torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), args.grad_clip)
            if args.grad_clip > 0.0 else get_grad_norm(unwrap_model(model).parameters())
        )
        stepped = False
        if torch.isfinite(grad_norm):
            scaler.step(optimizer)
            scaler.update()
            stepped = True
        else:
            logger.warning("[projector] step %d has NaN/Inf grad_norm; skipping optimizer", step)
            optimizer.zero_grad(set_to_none=True)
        if stepped:
            for name, feats in fd_queue_updates.items():
                fd_states[name]["queue"].enqueue(feats)

        torch.cuda.synchronize()

        args.current_step = step + 1
        args.samples_seen += args.global_bsz

        step_time = time.perf_counter() - step_start
        step_start = time.perf_counter()
        loss_value = all_reduce_mean(total_loss.detach())
        log_dict = {k: all_reduce_mean(v) for k, v in log_tensors.items()}
        grad_norm_value = all_reduce_mean(grad_norm)
        sps = args.batch_size / step_time if step_time > 0 else 0.0
        mem_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)

        metric_logger.update(
            loss=loss_value,
            grad_norm=grad_norm_value,
            lr=lr,
            **{
                "samples/s/device": sps,
                "samples/s": sps * args.world_size,
                "samples_seen(M)": args.samples_seen / 1e6,
                "device_mem(GB)": mem_gb,
            },
            **log_dict
        )

        if step % args.print_freq == 0:
            logger.info(
                "Projector Train: [%d/%d]  %s",
                args.current_step,
                args.total_steps,
                str(metric_logger),
            )
            if wandb_logger:
                wandb_logger.update({
                    "train/loss": loss_value,
                    "train/lr": lr,
                    "train/grad_norm": grad_norm_value,
                    "perf/samples_per_sec_per_device": sps,
                    "perf/samples_per_sec": sps * args.world_size,
                    "perf/max_reserved_mem_gb": mem_gb,
                    **{"train/%s" % k: v for k, v in log_dict.items()},
                }, step=args.current_step)

        if args.save_every > 0 and args.current_step % args.save_every == 0:
            elapsed = time.time() - start_time + args.last_elapsed_time
            save_checkpoint(args, model, optimizer, scaler, args.current_step, elapsed, fd_states)
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

    elapsed = time.time() - start_time + args.last_elapsed_time
    save_checkpoint(args, model, optimizer, scaler, args.current_step, elapsed, fd_states)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    metric_logger.synchronize_between_processes()
    logger.info("[projector] averaged stats: %s", str(metric_logger))
    logger.info("[projector] training complete in %s", datetime.timedelta(seconds=int(elapsed)))
    if wandb_logger:
        wandb_logger.finish()
    return 0


def get_args_parser():
    parser = argparse.ArgumentParser("Train a ViT multi-head feature projector")

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--subset_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    parser.add_argument("--teacher_models", type=str, nargs="+", default=[
        "inception",
        "vit_large_patch16_224.mae",
        "vit_so400m_patch16_siglip_256.v2_webli",
    ])
    parser.add_argument("--teacher_target_sizes", type=int, nargs="+", default=[256, 224, 224])
    parser.add_argument("--head_weights", type=float, nargs="+", default=None)
    parser.add_argument("--teacher_dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])

    parser.add_argument("--backbone", type=str, default="vit_s",
                        help="projector backbone preset: vit_s or vit_b")
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--embed_dim", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--attn_dropout", type=float, default=0.0)
    parser.add_argument("--projector_version", type=str, default="I", choices=["I", "II", "III"],
                        help=("I: 257 tokens, one prefix plus patches. "
                              "II: 258 tokens, two prefixes plus patches. "
                              "III: three prefixes plus patches; prefix2 predicts pooled Inception."))
    parser.add_argument("--head_variant", type=str, default="default", choices=HEAD_VARIANTS,
                        help=("default keeps all heads as configured by --head_mlp_layers; "
                              "siglip_deep_inception_conv makes the SigLIP head two MLP layers "
                              "deeper and uses a conv head for Inception."))
    parser.add_argument("--head_mlp_layers", type=int, default=1)
    parser.add_argument("--head_hidden_dim", type=int, default=None)
    parser.add_argument("--head_conv_layers", type=int, default=2,
                        help="number of conv layers for variant conv heads")
    parser.add_argument("--head_conv_kernel_size", type=int, default=3)
    parser.add_argument("--head_dropout", type=float, default=0.0)
    parser.add_argument("--no_input_normalize", action="store_true")
    parser.add_argument("--grad_checkpointing", action="store_true")

    parser.add_argument("--loss_type", type=str, default="mse",
                        choices=["mse", "relative_mse", "cosine", "relative_mse_cosine"])
    parser.add_argument("--cosine_weight", type=float, default=0.1)
    parser.add_argument("--fd_loss_weight", type=float, default=0.0,
                        help="weight for per-head Frechet loss on projector FD features")
    parser.add_argument("--fid_stats_path", type=str,
                        default="data/fid_stats/guided_diffusion_stats.npz")
    parser.add_argument("--fd_repr_stats_paths", type=str, nargs="+", default=None,
                        help="reference stats paths per teacher; inferred when omitted")
    parser.add_argument("--fd_queue_size", type=int, default=50000)
    parser.add_argument("--fd_fid_norm_eps", type=float, default=0.01)
    parser.add_argument("--fd_online_accum", action="store_true",
                        help="use online queue statistics instead of snapshot replacement")
    parser.add_argument("--fd_eigvalsh", action="store_true",
                        help="use sigma_ref_sqrt + eigvalsh for the Frechet trace term")
    parser.add_argument("--fd_ema_beta", type=float, default=0.0,
                        help="EMA beta for fake feature moments; >0 avoids raw feature queue storage")
    parser.add_argument("--loss_eps", type=float, default=1e-6)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps_per_epoch", type=int, default=1250)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--lr_sched", type=str, default="cosine", choices=["constant", "cosine"])
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--use_muon", action="store_true")
    parser.add_argument("--muon_lr", type=float, default=1e-3)
    parser.add_argument("--muon_momentum", type=float, default=0.95)
    parser.add_argument("--muon_weight_decay", type=float, default=0.0)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--compile", action="store_true")

    parser.add_argument("--output_dir", type=str, default="./work_dirs")
    parser.add_argument("--project", type=str, default="projector")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--print_freq", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--current_step", type=int, default=0)
    parser.add_argument("--samples_seen", type=int, default=0)
    parser.add_argument("--last_elapsed_time", type=float, default=0.0)

    parser.add_argument("--enable_wandb", action="store_true")
    parser.add_argument("--disable_wandb", action="store_false", dest="enable_wandb")
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1)

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    sys.exit(train(args))
