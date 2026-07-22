import argparse
import datetime
import logging
import os
import sys
import time

import torch
import torch.distributed

from utils.builders import create_generation_model, create_tokenizer
from utils.checkpoint_util import AsyncCheckpointSaver, ckpt_resume, save_checkpoint
from utils.distributed_util import all_reduce_mean, preempt_requested, register_preempt_handler
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
from frechet_distance.repr_models import (
    inception_feature_layer_from_name,
    load_repr_model,
    model_short_name,
)
from frechet_distance.judges import (
    resolve_per_model_args, save_fd_queue_states, load_fd_queue_states,
    run_sanity_check,
)
from frechet_distance.adversarial import (
    build_real_whitening,
    real_whitened_frechet_distance_from_stats,
)
from utils.rng_util import RNGStateManager
from utils.schedule_util import adjust_learning_rate
from utils.setup_util import setup
from utils.vis_util import visualize

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.cache_size_limit = 128
torch._dynamo.config.optimize_ddp = False

logger = logging.getLogger("FD_loss")


PROJECTOR_CHECKPOINT_DEFAULT = (
    "/mmu-vcg/gaomingju/workspace/foundation/FD-Loss-Ours/"
    "work_dirs/projector/vits-token-vII-inception-mae-siglip-mse-fdema-muon/"
    "checkpoints/last.pt"
)
PROJECTOR_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}
PROJECTOR_HEAD_SOURCE_MODES = {
    "inception": "patches",
    "mae": "prefix1",
    "siglip": "prefix0",
}
PROJECTOR_HEAD_VARIANTS = ("default", "siglip_deep_inception_conv")


class FDProjectorModel(torch.nn.Module):
    """Frozen multi-head projector exposed as an FD representation model."""

    is_fd_projector = True

    def __init__(self, projector, amp_dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.projector = projector
        self.amp_dtype = amp_dtype
        self.head_dims = dict(projector.head_dims)

    def forward_heads(self, images, head_names):
        enabled = self.amp_dtype != torch.float32
        with torch.autocast("cuda", enabled=enabled, dtype=self.amp_dtype):
            outputs = self.projector(images, head_names=head_names)
        return {name: feat.float() for name, feat in outputs.items()}


def _strip_state_prefix(state_dict, prefix: str):
    if not any(k.startswith(prefix) for k in state_dict):
        return state_dict
    return {
        (k[len(prefix):] if k.startswith(prefix) else k): v
        for k, v in state_dict.items()
    }


def _load_projector_checkpoint(path: str):
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Could not find FD projector checkpoint at {path}")
    logger.info("[FD-Projector] loading checkpoint: %s", path)
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    state_dict = _strip_state_prefix(state_dict, "module.")
    state_dict = _strip_state_prefix(state_dict, "_orig_mod.")
    return checkpoint if isinstance(checkpoint, dict) else {}, state_dict


def _infer_projector_head_dims(state_dict):
    candidates = {}
    for key, value in state_dict.items():
        if not key.startswith("heads.") or not key.endswith(".weight"):
            continue
        if getattr(value, "ndim", 0) not in (2, 4):
            continue
        parts = key.split(".")
        if len(parts) < 4 or parts[2] != "net":
            continue
        head_name = parts[1]
        order = int(parts[3]) if len(parts) > 4 and parts[3].isdigit() else 0
        candidates.setdefault(head_name, []).append((order, key, int(value.shape[0])))

    head_dims = {}
    for head_name, entries in candidates.items():
        _, _, dim = max(entries, key=lambda item: item[0])
        head_dims[head_name] = dim
    if not head_dims:
        raise RuntimeError("Could not infer projector head dimensions from checkpoint")
    return head_dims


def _projector_head_for_repr_model(name: str):
    inception_layer = inception_feature_layer_from_name(name)
    if inception_layer is not None:
        if inception_layer != "pool3":
            raise ValueError(
                "The configured projector checkpoint only has the Inception pool3 "
                f"head; unsupported FD repr model: {name}"
            )
        return "inception"

    low = name.lower()
    if "siglip" in low:
        return "siglip"
    if "mae" in low:
        return "mae"
    return None


def _projector_head_source_modes_for_version(
    projector_version: str,
    head_variant: str = "default",
):
    modes = dict(PROJECTOR_HEAD_SOURCE_MODES)
    if projector_version == "I":
        modes["mae"] = "prefix0_patches"
        modes["siglip"] = "prefix0_patches"
    elif projector_version == "III":
        modes["inception"] = "prefix2"
    if head_variant == "siglip_deep_inception_conv":
        modes["inception"] = "patches"
    return modes


def _projector_head_config_for_variant(
    head_dims,
    head_variant: str,
    base_mlp_layers: int,
    conv_layers: int,
):
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


def _projector_head_target_grid(name, source_mode, grid, projector_version, head_variant):
    if (
        head_variant == "siglip_deep_inception_conv"
        and projector_version == "III"
        and name == "inception"
    ):
        return (1, 1)
    if source_mode in ("patches", "prefix0_patches"):
        return grid
    return (1, 1)


def _resolve_projector_head_variant(args, checkpoint):
    ckpt_args = checkpoint.get("args", {})
    head_variant = args.fd_projector_head_variant
    ckpt_variant = ckpt_args.get("head_variant") if ckpt_args else None
    if head_variant == "default" and ckpt_variant in PROJECTOR_HEAD_VARIANTS:
        head_variant = ckpt_variant
    return head_variant, ckpt_args


def _apply_fd_projector_backbone_defaults(args, checkpoint):
    from models.projector import (
        canonical_projector_backbone,
        infer_projector_backbone,
        projector_backbone_config,
    )

    ckpt_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    ckpt_backbone = ckpt_args.get("backbone") if ckpt_args else None
    args.fd_projector_backbone = canonical_projector_backbone(
        ckpt_backbone or args.fd_projector_backbone
    )
    config = projector_backbone_config(args.fd_projector_backbone)

    key_map = {
        "img_size": "fd_projector_img_size",
        "patch_size": "fd_projector_patch_size",
        "embed_dim": "fd_projector_embed_dim",
        "depth": "fd_projector_depth",
        "num_heads": "fd_projector_num_heads",
        "mlp_ratio": "fd_projector_mlp_ratio",
        "head_mlp_layers": "fd_projector_head_mlp_layers",
        "head_hidden_dim": "fd_projector_head_hidden_dim",
        "head_conv_layers": "fd_projector_head_conv_layers",
        "head_conv_kernel_size": "fd_projector_head_conv_kernel_size",
    }
    for ckpt_key, arg_key in key_map.items():
        ckpt_value = ckpt_args.get(ckpt_key) if ckpt_args else None
        if ckpt_value is not None:
            setattr(args, arg_key, ckpt_value)
        elif getattr(args, arg_key) is None and ckpt_key in config:
            setattr(args, arg_key, config[ckpt_key])

    if ckpt_args.get("projector_version") is not None:
        args.fd_projector_version = ckpt_args["projector_version"]
    if ckpt_backbone is None:
        args.fd_projector_backbone = infer_projector_backbone(
            args.fd_projector_patch_size,
            args.fd_projector_embed_dim,
            args.fd_projector_depth,
            args.fd_projector_num_heads,
            default=args.fd_projector_backbone,
        )


def build_fd_projector(args, required_heads):
    from models.projector import ViTSMultiHeadProjector

    checkpoint, state_dict = _load_projector_checkpoint(args.fd_projector_checkpoint)
    head_variant, ckpt_args = _resolve_projector_head_variant(args, checkpoint)
    args.fd_projector_head_variant = head_variant
    _apply_fd_projector_backbone_defaults(args, checkpoint)
    head_dims = _infer_projector_head_dims(state_dict)
    missing = sorted(set(required_heads) - set(head_dims))
    if missing:
        raise ValueError(
            "FD projector checkpoint is missing required head(s): "
            + ", ".join(missing)
            + f". Available heads: {', '.join(sorted(head_dims))}"
        )

    projector_head_source_modes = _projector_head_source_modes_for_version(
        args.fd_projector_version,
        head_variant,
    )
    unknown_heads = sorted(set(head_dims) - set(projector_head_source_modes))
    if unknown_heads:
        raise ValueError(
            "FD projector checkpoint contains unsupported head(s): "
            + ", ".join(unknown_heads)
        )

    grid = (
        args.fd_projector_img_size // args.fd_projector_patch_size,
        args.fd_projector_img_size // args.fd_projector_patch_size,
    )
    head_source_modes = {
        name: projector_head_source_modes[name]
        for name in head_dims
    }
    head_target_grids = {
        name: _projector_head_target_grid(
            name,
            head_source_modes[name],
            grid,
            args.fd_projector_version,
            head_variant,
        )
        for name in head_dims
    }
    head_types, head_mlp_layer_overrides = _projector_head_config_for_variant(
        head_dims,
        head_variant,
        args.fd_projector_head_mlp_layers,
        args.fd_projector_head_conv_layers,
    )

    projector = ViTSMultiHeadProjector(
        head_dims=head_dims,
        img_size=args.fd_projector_img_size,
        patch_size=args.fd_projector_patch_size,
        in_chans=3,
        embed_dim=args.fd_projector_embed_dim,
        depth=args.fd_projector_depth,
        num_heads=args.fd_projector_num_heads,
        mlp_ratio=args.fd_projector_mlp_ratio,
        dropout=0.0,
        attn_dropout=0.0,
        projector_version=args.fd_projector_version,
        head_source_modes=head_source_modes,
        head_target_grids=head_target_grids,
        head_types=head_types,
        head_mlp_layers=args.fd_projector_head_mlp_layers,
        head_mlp_layer_overrides=head_mlp_layer_overrides,
        head_hidden_dim=args.fd_projector_head_hidden_dim,
        head_conv_kernel_size=args.fd_projector_head_conv_kernel_size,
        head_dropout=0.0,
        normalize_input=not args.fd_projector_no_input_normalize,
        grad_checkpointing=args.fd_projector_grad_checkpointing,
    )
    projector.load_state_dict(state_dict, strict=True)
    projector.cuda().requires_grad_(False)
    if args.fd_projector_grad_checkpointing:
        projector.train()
    else:
        projector.eval()
    amp_dtype = PROJECTOR_DTYPE_MAP[args.fd_projector_dtype]
    wrapped = FDProjectorModel(projector, amp_dtype=amp_dtype).cuda()
    if args.fd_projector_grad_checkpointing:
        wrapped.train()
    else:
        wrapped.eval()
    logger.info(
        "[FD-Projector] ready: backbone=%s version=%s variant=%s ckpt_step=%s heads=%s dtype=%s",
        args.fd_projector_backbone,
        args.fd_projector_version,
        head_variant,
        str(checkpoint.get("step", "unknown")),
        ", ".join("%s:%d" % (name, dim) for name, dim in sorted(head_dims.items())),
        args.fd_projector_dtype,
    )
    if ckpt_args:
        logger.info(
            "[FD-Projector] checkpoint config: img_size=%s patch_size=%s "
            "embed_dim=%s depth=%s num_heads=%s backbone=%s head_mlp_layers=%s head_variant=%s",
            str(ckpt_args.get("img_size")),
            str(ckpt_args.get("patch_size")),
            str(ckpt_args.get("embed_dim")),
            str(ckpt_args.get("depth")),
            str(ckpt_args.get("num_heads")),
            str(ckpt_args.get("backbone")),
            str(ckpt_args.get("head_mlp_layers")),
            str(ckpt_args.get("head_variant")),
        )
    return wrapped


def _projector_tokens_to_fd_features(judge, tokens):
    if tokens.ndim == 2:
        return tokens.float()
    if tokens.ndim != 3:
        raise RuntimeError(
            "FD projector head '%s' returned shape %s, expected [B,C] or [B,T,C]"
            % (judge.get("projector_head"), tuple(tokens.shape))
        )

    head = judge.get("projector_head")
    if head == "inception":
        if tokens.shape[1] == 1:
            return tokens[:, 0].float()
        # Version II was trained on 16x16 Inception pool3 tokens.  The FD
        # reference stats are per-image pool3 vectors, so pool tokens back to
        # one feature per image before queue/Frechet statistics.
        return tokens.mean(dim=1).float()

    # Version I token-level MAE/SigLIP heads return primary + patch tokens.
    # The FD feature is the primary token.
    return tokens[:, 0].float()


def extract_judge_features_projector(judge, images):
    if not getattr(judge.get("model"), "is_fd_projector", False):
        raise RuntimeError(
            "main_fd_projector.py expects projector FD judges; "
            f"judge '{judge.get('name')}' is not backed by FDProjectorModel"
        )
    outputs = judge["model"].forward_heads(images, [judge["projector_head"]])
    return _projector_tokens_to_fd_features(judge, outputs[judge["projector_head"]])


def extract_judge_features_all_projector(judges, images):
    features = [None] * len(judges)
    projector_groups = {}

    for idx, judge in enumerate(judges):
        if not getattr(judge.get("model"), "is_fd_projector", False):
            raise RuntimeError(
                "main_fd_projector.py expects projector FD judges; "
                f"judge '{judge.get('name')}' is not backed by FDProjectorModel"
            )
        key = id(judge["model"])
        projector_groups.setdefault(key, {"model": judge["model"], "entries": []})
        projector_groups[key]["entries"].append((idx, judge, judge["projector_head"]))

    for group in projector_groups.values():
        head_names = []
        for _, _, head_name in group["entries"]:
            if head_name not in head_names:
                head_names.append(head_name)
        outputs = group["model"].forward_heads(images, head_names)
        for idx, judge, head_name in group["entries"]:
            features[idx] = _projector_tokens_to_fd_features(judge, outputs[head_name])

    return features


@torch.no_grad()
def fill_all_queues_projector(judges, model, args, tokenizer=None):
    """Fill FD queues with the projector-aware feature extraction path."""
    queue_size = args.queue_size
    if queue_size == 0:
        logger.info("[FD] queue_size=0: skipping queue fill")
        return

    model.eval()
    filled = 0
    while filled < queue_size:
        batch_size = min(args.fd_queue_fill_bsz, queue_size - filled)
        y = torch.randint(0, args.num_classes, (batch_size,), device="cuda")
        imgs = model.generate(batch_size, y, cfg=args.cfg, args=args, verbose=False)
        if tokenizer is not None:
            imgs = tokenizer.detokenize(imgs)
        else:
            imgs = imgs * 0.5 + 0.5

        local_feats_list = extract_judge_features_all_projector(judges, imgs)
        for judge, local_feats in zip(judges, local_feats_list):
            all_feats = diff_all_gather(local_feats)
            count = min(all_feats.shape[0], queue_size - filled)
            q = judge["queue"]
            if q.ema_stats:
                q.accumulate_batch(all_feats[:count])
            else:
                q.feats[filled:filled + count] = all_feats[:count].float()

        filled += count
        logger.info(f"[FD] Queue fill: {filled}/{queue_size} ({filled / queue_size * 100:.1f}%)")

    for judge in judges:
        q = judge["queue"]
        if q.ema_stats:
            q._finalize_streaming_init()
        else:
            q.ptr.zero_()
            if q.online_accum:
                q._init_accumulators()
    logger.info(f"[FD] All {len(judges)} queues initialized with {filled} features")


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

def _validate_fd_sequential_backward_args(args):
    if not args.fd_sequential_backward:
        return

    incompatible = []
    if args.jit_loss_weight > 0:
        incompatible.append("--jit_loss_weight")
    if args.compile:
        incompatible.append("--compile")
    if incompatible:
        raise ValueError(
            "--fd_sequential_backward currently supports plain FD generator "
            "loss only. Incompatible options: " + ", ".join(incompatible)
        )


def _all_reduce_grads(module):
    if not torch.distributed.is_initialized():
        return
    for param in module.parameters():
        if param.grad is not None:
            torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.AVG)


def get_fd_train_step(
    model_wo_ddp,
    judges,
    sampling_args,
    args,
    tokenizer=None,
    real_batch_fn=None,
):
    fid_norm_eps = args.fd_fid_norm_eps
    jit_loss_weight = args.jit_loss_weight
    batch_size = args.batch_size
    num_classes = args.num_classes
    input_shape = (args.input_channels, args.input_size, args.input_size)

    def fd_train_step():
        x0, t_view, noise, velocity_pred = None, None, None, None
        if args.fd_random_timestep_training:
            if real_batch_fn is None:
                raise RuntimeError("fd_random_timestep_training requires a real image dataloader")
            real_images, y = real_batch_fn()
            x0 = real_images.mul(2.0).sub(1.0)
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

        if tokenizer is not None:
            sampled = tokenizer.decode(tokenizer.denormalize_z(sampled))
        sampled = sampled * 0.5 + 0.5  # [-1,1] -> [0,1]

        loss_dict = {}
        loss = torch.tensor(0.0, device="cuda")

        def _compute_fd_main_loss(judge, new_feats):
            _ns_kwargs = dict(sigma_ref_sqrt=judge.get("sigma_ref_sqrt"))
            if judge["queue"].online_accum or judge["queue"].ema_stats:
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
                    fid = compute_frechet_distance_loss(
                        judge["mu_ref"], judge["sigma_ref"],
                        mu=mu, sigma=sigma, **_ns_kwargs,
                    )
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
                    fid = compute_frechet_distance_loss(
                        judge["mu_ref"], judge["sigma_ref"],
                        all_feats=all_feats, **_ns_kwargs,
                    )
            fid_loss = fid / (fid.detach() + fid_norm_eps)
            return fid, judge["weight"] * fid_loss

        all_new_feats = []
        if not args.fd_sequential_backward:
            for feats in extract_judge_features_all_projector(judges, sampled):
                new_feats = diff_all_gather(feats)
                all_new_feats.append(new_feats)

        if args.fd_sequential_backward:
            for i, judge in enumerate(judges):
                feats = extract_judge_features_projector(judge, sampled)
                new_feats = diff_all_gather(feats)
                all_new_feats.append(new_feats.detach())
                fid, fd_loss = _compute_fd_main_loss(judge, new_feats)
                loss = loss + fd_loss.detach()
                loss_dict[f"fid_{judge['name']}"] = float(fid.detach())
                fd_loss.backward(
                    retain_graph=(i < len(judges) - 1),
                    create_graph=False,
                )
                del feats, new_feats, fid, fd_loss
        else:
            for i, judge in enumerate(judges):
                new_feats = all_new_feats[i]
                fid, fd_loss = _compute_fd_main_loss(judge, new_feats)
                loss = loss + fd_loss
                loss_dict[f"fid_{judge['name']}"] = float(fid.detach())

        if jit_loss_weight > 0:
            if not args.fd_random_timestep_training or velocity_pred is None:
                raise RuntimeError("jit_loss_weight > 0 requires fd_random_timestep_training")
            velocity_target = t_view * (noise - x0) / t_view.clamp_min(model_wo_ddp.t_eps)
            jit_loss = ((velocity_pred - velocity_target) ** 2).mean()
            loss = loss + jit_loss_weight * jit_loss
            loss_dict["jit_loss"] = float(jit_loss.detach())

        if not args.fd_sequential_backward:
            loss.backward(create_graph=False)

        if torch.distributed.is_initialized():
            _all_reduce_grads(model_wo_ddp)

        for i, judge in enumerate(judges):
            judge["queue"].enqueue(all_new_feats[i].detach())

        return loss, loss_dict

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
    _validate_fd_sequential_backward_args(args)

    # -- models, optimizer, checkpoint --
    tokenizer = create_tokenizer(args)
    model, ema_model = create_generation_model(args)
    model_wo_ddp = model

    extra_keys = ["fd_queue_states"]

    optimizer = create_optimizer(args, model_wo_ddp, print_trainable_params=True)
    extra = ckpt_resume(args, model_wo_ddp, optimizer, ema_model,
                        extra_keys=extra_keys)

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

    fd_projector_heads = {}
    for name, pool_type in zip(args.fd_repr_models, args.fd_repr_pool_types):
        head_name = _projector_head_for_repr_model(name)
        if head_name is None:
            raise ValueError(
                f"FD projector checkpoint does not support repr model '{name}'. "
                "Supported FD reprs are Inception pool3, MAE, and SigLIP."
            )
        if head_name in ("mae", "siglip") and pool_type != "cls":
            raise ValueError(
                f"Projector version {args.fd_projector_version} head '{head_name}' is used via its "
                f"primary token in the FD projector path, so pool_type must be 'cls' for repr '{name}'."
            )
        fd_projector_heads[name] = head_name
    fd_projector = build_fd_projector(args, set(fd_projector_heads.values()))

    judges = []
    for name, stats_path, weight, pool_type, ts in zip(
        args.fd_repr_models, args.fd_repr_stats_paths,
        args.fd_repr_weights, args.fd_repr_pool_types, args.fd_target_sizes,
    ):
        short = model_short_name(name)
        inception_layer = inception_feature_layer_from_name(name)
        logger.info("[FD] repr '%s' uses frozen projector head", short)
        projector_head = fd_projector_heads[name]
        repr_model = fd_projector
        feat_dim = fd_projector.head_dims[projector_head]
        mu_ref, sigma_ref = load_mu_and_sigma_reference(stats_path, pool_type=pool_type)
        queue = FeatureQueue(size=args.queue_size, feat_dim=feat_dim,
                             online_accum=args.fd_online_accum,
                             ema_beta=args.fd_ema_beta).cuda()
        sigma_ref_sqrt = None
        if args.fd_eigvalsh:
            sigma_ref_sqrt = precompute_sigma_ref_sqrt(sigma_ref)
        judge = {
            "name": short, "model": repr_model,
            "feat_dim": feat_dim,
            "pool_type": pool_type,
            "inception_layer": inception_layer,
            "projector_head": fd_projector_heads.get(name),
            "mu_ref": mu_ref, "sigma_ref": sigma_ref,
            "sigma_ref_sqrt": sigma_ref_sqrt,
            "queue": queue, "weight": weight,
        }
        judges.append(judge)
        eig_mode = "eigvalsh" if args.fd_eigvalsh else "eigvals"
        stats_mode = f"ema(beta={args.fd_ema_beta})" if args.fd_ema_beta > 0 else ("online_accum" if args.fd_online_accum else "snapshot")
        logger.info(f"[FD] Repr '{short}' ({name}): feat_dim={feat_dim}, "
                     f"weight={weight}, pool={pool_type}, stats={stats_path}, "
                     f"eig_mode={eig_mode}, stats_mode={stats_mode}, "
                     f"whiten={args.fd_whiten}, whiten_eps={args.fd_whiten_eps}")

    if args.fd_sequential_backward:
        logger.info(
            "[FD] sequential backward enabled: FD repr losses will be "
            "computed and backpropagated one at a time to reduce peak memory"
        )

    real_batch_fn = (
        build_real_image_batch_fn(args)
        if (args.fd_random_timestep_training or args.jit_loss_weight > 0)
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
        fill_all_queues_projector(judges, model_wo_ddp, args, tokenizer=tokenizer)
        run_sanity_check(judges, args.queue_size, args=args)
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

        loss, loss_dict = fd_train_step()

        grad_norm = (torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                     if args.grad_clip > 0.0 else get_grad_norm(model.parameters()))

        if torch.isfinite(grad_norm):
            optimizer.step()
            ema_model.step(model)
        else:
            logger.warning(f"[step {step}] NaN/Inf grad_norm — skipping optimizer & EMA update")
        optimizer.zero_grad(set_to_none=True)

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
        "--fd_sequential_backward",
        action="store_true",
        help=(
            "compute FD representation losses one at a time and backward each "
            "before moving to the next repr model; reduces peak memory for "
            "multi-repr FD training. Currently supports plain FD generator "
            "loss only."
        ),
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
    parser.add_argument("--queue_size", type=int, default=50000)
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
    parser.add_argument("--fd_projector_checkpoint", type=str, default=PROJECTOR_CHECKPOINT_DEFAULT,
                        help="checkpoint for the frozen multi-head FD projector")
    parser.add_argument("--fd_projector_version", type=str, default="II", choices=["I", "II", "III"])
    parser.add_argument("--fd_projector_backbone", type=str, default="vit_s",
                        help="projector backbone preset: vit_s or vit_b")
    parser.add_argument("--fd_projector_img_size", type=int, default=256)
    parser.add_argument("--fd_projector_patch_size", type=int, default=None)
    parser.add_argument("--fd_projector_embed_dim", type=int, default=None)
    parser.add_argument("--fd_projector_depth", type=int, default=None)
    parser.add_argument("--fd_projector_num_heads", type=int, default=None)
    parser.add_argument("--fd_projector_mlp_ratio", type=float, default=4.0)
    parser.add_argument("--fd_projector_head_variant", type=str, default="default",
                        choices=PROJECTOR_HEAD_VARIANTS)
    parser.add_argument("--fd_projector_head_mlp_layers", type=int, default=1)
    parser.add_argument("--fd_projector_head_hidden_dim", type=int, default=None)
    parser.add_argument("--fd_projector_head_conv_layers", type=int, default=2)
    parser.add_argument("--fd_projector_head_conv_kernel_size", type=int, default=3)
    parser.add_argument("--fd_projector_dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--fd_projector_no_input_normalize", action="store_true")
    parser.add_argument("--fd_projector_grad_checkpointing", action="store_true")
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


def _cleanup_distributed():
    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    try:
        exit_code = train_and_evaluate(args)
    finally:
        _cleanup_distributed()
    sys.exit(exit_code)
