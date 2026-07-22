#!/usr/bin/env python3
import argparse
import contextlib
import datetime
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import typing_extensions

# PyTorch 2.4 imports this decorator, while the system image ships an older
# typing_extensions. It is metadata-only for our use, so a no-op fallback is
# sufficient and keeps the training entry self-contained.
if not hasattr(typing_extensions, "deprecated"):
    def _deprecated(*args, **kwargs):
        del args, kwargs

        def decorator(obj):
            return obj

        return decorator

    typing_extensions.deprecated = _deprecated

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from meanflow.meanflow_jit import MeanFlowJiTDenoiser  # noqa: E402
from utils.data_util import center_crop_arr  # noqa: E402
from utils.ema_util import EMAModel  # noqa: E402


logger = logging.getLogger("meanflow_jit")
_DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def init_distributed():
    if "RANK" not in os.environ:
        return 0, 0, 1
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def is_main_process():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def unwrap_model(model):
    if isinstance(model, DDP):
        model = model.module
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


def setup_logging(log_dir):
    if is_main_process():
        os.makedirs(log_dir, exist_ok=True)
        handlers = [
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(log_dir, "train.log")),
        ]
    else:
        handlers = [logging.NullHandler()]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )


def setup_wandb(args):
    if not args.enable_wandb or not is_main_process():
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb is not installed, but --enable_wandb was set") from exc

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name or args.exp_name,
        dir=args.run_dir,
        config=vars(args),
        resume="allow",
    )
    logger.info("wandb run: %s", run.url)
    return run


def build_loader(args, rank, world_size):
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, DistributedSampler

    train_dir = os.path.join(args.data_path, "train")
    if not os.path.isdir(train_dir):
        train_dir = args.data_path

    transform_steps = [
        transforms.Lambda(lambda img: center_crop_arr(img, args.img_size)),
    ]
    if not args.disable_hflip:
        transform_steps.append(transforms.RandomHorizontalFlip())
    transform_steps.append(transforms.ToTensor())
    transform = transforms.Compose(transform_steps)
    dataset = datasets.ImageFolder(train_dir, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    ) if world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )
    logger.info("train images from %s: %d", train_dir, len(dataset))
    return loader, sampler


def select_state_dict(checkpoint, key):
    if not isinstance(checkpoint, dict):
        return checkpoint
    if key != "auto":
        if key not in checkpoint:
            raise KeyError(f"Checkpoint key '{key}' not found. Available: {list(checkpoint.keys())[:20]}")
        return checkpoint[key]
    for candidate in ("model", "state_dict", "model_ema2", "model_ema1", "model_ema"):
        if candidate in checkpoint:
            return checkpoint[candidate]
    return checkpoint


def normalize_state_dict_keys(state_dict):
    out = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        out[key] = value
    return out


def _state_key_is_allowed(key, prefixes):
    return any(key == prefix.rstrip(".") or key.startswith(prefix) for prefix in prefixes)


def _validate_checkpoint_metadata(checkpoint, required, context):
    if not required:
        return
    checkpoint_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    if not isinstance(checkpoint_args, dict) and hasattr(checkpoint_args, "__dict__"):
        checkpoint_args = vars(checkpoint_args)
    if not isinstance(checkpoint_args, dict):
        checkpoint_args = {}
    mismatches = {
        key: {"expected": expected, "found": checkpoint_args.get(key)}
        for key, expected in required.items()
        if checkpoint_args.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(
            f"{context} metadata mismatch: {mismatches}. Refusing to mix "
            "incompatible model parameterizations."
        )


def load_checkpoint(
    model,
    path,
    key,
    allowed_missing_prefixes=None,
    allowed_unexpected_prefixes=None,
    required_metadata=None,
):
    logger.info("loading checkpoint: %s key=%s", path, key)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    _validate_checkpoint_metadata(
        checkpoint,
        required_metadata,
        context="initialization checkpoint",
    )
    state_dict = normalize_state_dict_keys(select_state_dict(checkpoint, key))
    msg = model.load_state_dict(state_dict, strict=False)
    logger.info("loaded checkpoint: missing=%d unexpected=%d", len(msg.missing_keys), len(msg.unexpected_keys))
    if msg.missing_keys:
        logger.info("missing keys sample: %s", msg.missing_keys[:20])
    if msg.unexpected_keys:
        logger.info("unexpected keys sample: %s", msg.unexpected_keys[:20])
    if allowed_missing_prefixes is not None:
        allowed_unexpected_prefixes = allowed_unexpected_prefixes or ()
        disallowed_missing = [
            name
            for name in msg.missing_keys
            if not _state_key_is_allowed(name, allowed_missing_prefixes)
        ]
        disallowed_unexpected = [
            name
            for name in msg.unexpected_keys
            if not _state_key_is_allowed(name, allowed_unexpected_prefixes)
        ]
        if disallowed_missing or disallowed_unexpected:
            raise RuntimeError(
                "checkpoint/model mismatch outside the explicit whitelist: "
                f"missing={disallowed_missing[:20]}, "
                f"unexpected={disallowed_unexpected[:20]}"
            )


def create_model(args, model_cls=MeanFlowJiTDenoiser, **model_kwargs):
    return model_cls(
        img_size=args.img_size,
        model_size="base",
        num_classes=args.num_classes,
        label_drop_prob=args.label_drop_prob,
        attn_dropout=args.attn_dropout,
        proj_dropout=args.proj_dropout,
        P_mean=args.P_mean,
        P_std=args.P_std,
        t_eps=args.t_eps,
        noise_scale=args.noise_scale,
        legacy_time_convention=(args.objective != "scm"),
        rope_2d=True,
        learned_pe=True,
        grad_checkpointing=args.grad_checkpointing,
        adaptive_double_norm=(args.objective == "scm"),
        dropout_all_blocks=(args.objective == "scm"),
        meanflow_delta=args.meanflow_delta,
        dudt_weight=args.dudt_weight,
        dudt_drop_prob=args.dudt_drop_prob,
        dudt_clip_norm=args.dudt_clip_norm,
        loss_clip=args.loss_clip,
        objective=args.objective,
        tangent_norm_c=args.tangent_norm_c,
        tangent_warmup_steps=args.tangent_warmup_steps,
        adaptive_weighting=(
            args.objective == "scm" and not args.disable_adaptive_weighting
        ),
        adaptive_weight_hidden=args.adaptive_weight_hidden,
        adaptive_weight_max=args.adaptive_weight_max,
        sigma_data=args.sigma_data,
        sigma_max=args.sigma_max,
        jvp_dtype=args.jvp_dtype,
        **model_kwargs,
    )


def create_optimizer(args, model, world_size):
    if args.lr is None:
        args.lr = (
            args.blr * args.batch_size * args.grad_accum_steps * world_size / 256
        )

    def no_weight_decay(name, param):
        return (
            param.ndim < 2
            or "bias" in name
            or "norm" in name
            or "embed" in name
            or "token" in name
            # It is zero-touched during pre-sCM adaptation solely for DDP.
            # Coupled Adam weight decay must not move it before sCM begins.
            or "loss_weight_net." in name
        )

    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    decay = [p for n, p in named if not no_weight_decay(n, p)]
    nodecay = [p for n, p in named if no_weight_decay(n, p)]
    logger.info(
        "optimizer fused Adam lr=%.6g wd=%.6g decay_tensors=%d nodecay_tensors=%d",
        args.lr,
        args.weight_decay,
        len(decay),
        len(nodecay),
    )
    return torch.optim.Adam(
        [{"params": nodecay, "weight_decay": 0.0}, {"params": decay, "weight_decay": args.weight_decay}],
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )


def create_ema_model(args, model, world_size):
    if args.disable_ema:
        return None
    ema_values = (
        [args.ema_decay]
        if args.ema_type == "const"
        else [args.ema_halflife_kimg]
        if args.ema_type == "edm"
        else [args.ema_sigma_rel]
    )
    ema_model = EMAModel(
        unwrap_model(model),
        ema_type=args.ema_type,
        values=ema_values,
        batch_size=args.batch_size * world_size * args.grad_accum_steps,
    )
    logger.info("EMA: type=%s labels=%s", args.ema_type, ema_model.labels)
    return ema_model


def adjust_lr(optimizer, step, args):
    if step < args.warmup_steps:
        lr = args.lr * float(step + 1) / max(1, args.warmup_steps)
    elif args.lr_sched == "constant":
        lr = args.lr
    elif args.lr_sched == "edm2":
        lr = args.lr / math.sqrt(max(float(step + 1) / args.lr_ref_steps, 1.0))
    else:
        progress = (step - args.warmup_steps) / max(1, args.total_steps - args.warmup_steps)
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def reduce_value(value, device, op="mean"):
    tensor = torch.tensor(float(value), device=device)
    if dist.is_available() and dist.is_initialized():
        if op == "mean":
            dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
        elif op == "max":
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        elif op == "min":
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
        else:
            raise ValueError(f"unsupported reduce op: {op}")
    return tensor.item()


def reduce_mean(value, device):
    return reduce_value(value, device, op="mean")


def reduce_max(value, device):
    return reduce_value(value, device, op="max")


def reduce_min(value, device):
    return reduce_value(value, device, op="min")


def without_training_only_state(state_dict):
    return {
        key: value
        for key, value in state_dict.items()
        if not key.startswith("loss_weight_net.")
    }


def phase_adapt_steps(args):
    """Return the optional pre-sCM adaptation length for two-phase entries."""
    if hasattr(args, "phase_adapt_steps"):
        return int(args.phase_adapt_steps)
    if hasattr(args, "g_adapt_steps"):
        return int(args.g_adapt_steps)
    return int(getattr(args, "x_adapt_steps", 0))


def save_checkpoint(args, step, model, optimizer, ema_model):
    if not is_main_process():
        return
    raw_model = unwrap_model(model)
    training_state = raw_model.state_dict()
    checkpoint = {
        # Keep the generation weights separate from the training-only
        # adaptive weighting network. Sampling still requires the sCM wrapper.
        "model": without_training_only_state(training_state),
        "training_model": training_state,
        "optimizer": optimizer.state_dict(),
        "step": step,
        "args": vars(args),
    }
    if ema_model is not None:
        checkpoint["ema"] = ema_model.state_dict()
        checkpoint["model_ema"] = without_training_only_state(
            ema_model.state_dict(label=ema_model.default_label)
        )
    os.makedirs(args.ckpt_dir, exist_ok=True)
    path = os.path.join(args.ckpt_dir, f"step_{step:07d}.pth")
    torch.save(checkpoint, path)
    latest = os.path.join(args.ckpt_dir, "latest.pth")
    try:
        os.remove(latest)
    except FileNotFoundError:
        pass
    os.symlink(os.path.abspath(path), latest)
    logger.info("saved checkpoint: %s", path)
    preserve_steps = set()
    adapt_steps = phase_adapt_steps(args)
    if adapt_steps > 0:
        preserve_steps.add(adapt_steps - 1)
    cleanup_checkpoints(
        args.ckpt_dir,
        keep_last=args.keep_last,
        preserve_steps=preserve_steps,
    )


def cleanup_checkpoints(ckpt_dir, keep_last, preserve_steps=()):
    if keep_last <= 0:
        return
    preserve_steps = set(preserve_steps)
    checkpoints = []
    for name in os.listdir(ckpt_dir):
        if not (name.startswith("step_") and name.endswith(".pth")):
            continue
        path = os.path.join(ckpt_dir, name)
        try:
            step = int(name[len("step_"):-len(".pth")])
        except ValueError:
            continue
        checkpoints.append((step, path))
    checkpoints.sort()
    removable = [item for item in checkpoints if item[0] not in preserve_steps]
    for _, path in removable[:-keep_last]:
        try:
            os.remove(path)
            logger.info("removed old checkpoint: %s", path)
        except FileNotFoundError:
            pass


def maybe_resume(args, model, optimizer, ema_model):
    if not args.resume_from:
        return 0
    logger.info("resuming meanflow run from: %s", args.resume_from)
    checkpoint = torch.load(args.resume_from, map_location="cpu", weights_only=False)
    _validate_checkpoint_metadata(
        checkpoint,
        getattr(args, "resume_required_metadata", None),
        context="resume checkpoint",
    )
    raw_model = unwrap_model(model)
    state_dict = checkpoint.get("training_model", checkpoint["model"])
    msg = raw_model.load_state_dict(state_dict, strict=False)
    if msg.missing_keys or msg.unexpected_keys:
        logger.info(
            "resume state mismatch: missing=%d unexpected=%d",
            len(msg.missing_keys),
            len(msg.unexpected_keys),
        )
        if getattr(args, "enforce_checkpoint_whitelist", False):
            allowed_missing = getattr(
                args, "checkpoint_allowed_missing_prefixes", []
            )
            allowed_unexpected = getattr(
                args, "checkpoint_allowed_unexpected_prefixes", []
            )
            bad_missing = [
                name
                for name in msg.missing_keys
                if not _state_key_is_allowed(name, allowed_missing)
            ]
            bad_unexpected = [
                name
                for name in msg.unexpected_keys
                if not _state_key_is_allowed(name, allowed_unexpected)
            ]
            if bad_missing or bad_unexpected:
                raise RuntimeError(
                    "resume checkpoint/model mismatch outside whitelist: "
                    f"missing={bad_missing[:20]}, "
                    f"unexpected={bad_unexpected[:20]}"
                )
    optimizer.load_state_dict(checkpoint["optimizer"])
    if ema_model is not None and checkpoint.get("ema") is not None:
        ema_model.load_state_dict(checkpoint["ema"])
    return int(checkpoint.get("step", -1)) + 1


def train(
    args,
    model_factory=None,
    lr_adjuster=None,
    phase_transition=None,
):
    rank, local_rank, world_size = init_distributed()
    if args.grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be positive")
    if args.save_every <= 0:
        raise ValueError("save_every must be positive")
    if args.dense_save_every < 0:
        raise ValueError("dense_save_every must be non-negative")
    if args.dense_save_start < 0:
        raise ValueError("dense_save_start must be non-negative")
    if (
        args.dense_save_every > 0
        and args.dense_save_end >= 0
        and args.dense_save_end < args.dense_save_start
    ):
        raise ValueError(
            "dense_save_end must be >= dense_save_start, or negative"
        )
    if args.lr_sched == "edm2" and args.lr_ref_steps <= 0:
        raise ValueError("lr_ref_steps must be positive for the EDM2 schedule")
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    total_steps_override = getattr(args, "total_steps_override", None)
    args.total_steps = (
        int(total_steps_override)
        if total_steps_override is not None
        else args.epochs * args.steps_per_epoch
    )
    if args.total_steps <= 0:
        raise ValueError("total training steps must be positive")
    args.warmup_steps = args.warmup_epochs * args.steps_per_epoch
    args.run_dir = os.path.join(args.output_dir, args.project, args.exp_name)
    args.ckpt_dir = os.path.join(args.run_dir, "checkpoints")
    setup_logging(args.run_dir)
    wandb_run = setup_wandb(args)

    if is_main_process():
        os.makedirs(args.run_dir, exist_ok=True)
        with open(os.path.join(args.run_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2, sort_keys=True)
    logger.info("rank=%d local_rank=%d world_size=%d", rank, local_rank, world_size)
    logger.info("run_dir=%s", args.run_dir)
    logger.info(
        "batch: micro/device=%d accumulation=%d global=%d",
        args.batch_size,
        args.grad_accum_steps,
        args.batch_size * args.grad_accum_steps * world_size,
    )

    loader, sampler = build_loader(args, rank, world_size)
    model_factory = model_factory or create_model
    model = model_factory(args).to(device)
    enforce_checkpoint_whitelist = getattr(
        args, "enforce_checkpoint_whitelist", False
    )
    allowed_missing_prefixes = (
        getattr(args, "checkpoint_allowed_missing_prefixes", [])
        if enforce_checkpoint_whitelist
        else None
    )
    allowed_unexpected_prefixes = (
        getattr(args, "checkpoint_allowed_unexpected_prefixes", [])
        if enforce_checkpoint_whitelist
        else None
    )
    load_checkpoint(
        model,
        args.load_from,
        args.checkpoint_key,
        allowed_missing_prefixes=allowed_missing_prefixes,
        allowed_unexpected_prefixes=allowed_unexpected_prefixes,
        required_metadata=getattr(
            args, "checkpoint_required_metadata", None
        ),
    )
    initialize_reference_teacher = getattr(
        model, "initialize_reference_teacher", None
    )
    if initialize_reference_teacher is not None:
        # Some distillation variants keep an unregistered, frozen copy of the
        # initialization network.  Initialize it only after the requested
        # checkpoint key has been loaded, and before compile/DDP wrapping.
        initialize_reference_teacher()
    if args.compile:
        model = torch.compile(model)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    ema_model = create_ema_model(args, model, world_size)

    optimizer = create_optimizer(args, model, world_size)
    start_step = maybe_resume(args, model, optimizer, ema_model)
    scaler = torch.cuda.amp.GradScaler(enabled=args.dtype == "fp16")
    amp_dtype = _DTYPE[args.dtype]

    data_epoch = 0
    if sampler is not None:
        sampler.set_epoch(data_epoch)
    data_iter = iter(loader)
    start_time = time.time()
    last_log_time = start_time
    last_log_step = -1
    collapse_bad_logs = 0
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(start_step, args.total_steps):
        if phase_transition is not None:
            optimizer, ema_model = phase_transition(
                step, args, model, optimizer, ema_model, world_size
            )
        lr_fn = lr_adjuster or adjust_lr
        lr = lr_fn(optimizer, step, args)
        optimizer.zero_grad(set_to_none=True)
        local_loss_sum = None
        local_metrics_sum = {}

        for micro_step in range(args.grad_accum_steps):
            try:
                images, labels = next(data_iter)
            except StopIteration:
                data_epoch += 1
                if sampler is not None:
                    sampler.set_epoch(data_epoch)
                data_iter = iter(loader)
                images, labels = next(data_iter)

            images = images.to(device, non_blocking=True).mul(2.0).sub(1.0)
            labels = labels.to(device, non_blocking=True)
            sync_context = (
                model.no_sync()
                if isinstance(model, DDP) and micro_step + 1 < args.grad_accum_steps
                else contextlib.nullcontext()
            )
            with sync_context:
                with torch.cuda.amp.autocast(enabled=args.dtype != "fp32", dtype=amp_dtype):
                    micro_loss, micro_loss_dict = model(
                        images, labels, global_step=step
                    )
                scaler.scale(micro_loss / args.grad_accum_steps).backward()

            detached_loss = micro_loss.detach().float()
            local_loss_sum = (
                detached_loss
                if local_loss_sum is None
                else local_loss_sum + detached_loss
            )
            for name, value in micro_loss_dict.items():
                scalar_value = value.detach().float()
                if name.endswith("_max"):
                    local_metrics_sum[name] = (
                        scalar_value
                        if name not in local_metrics_sum
                        else torch.maximum(local_metrics_sum[name], scalar_value)
                    )
                elif name.endswith("_min"):
                    local_metrics_sum[name] = (
                        scalar_value
                        if name not in local_metrics_sum
                        else torch.minimum(local_metrics_sum[name], scalar_value)
                    )
                else:
                    local_metrics_sum[name] = (
                        scalar_value
                        if name not in local_metrics_sum
                        else local_metrics_sum[name] + scalar_value
                    )

        loss_is_finite = torch.isfinite(local_loss_sum).to(dtype=torch.int32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(loss_is_finite, op=dist.ReduceOp.MIN)
        if not loss_is_finite.item():
            raise FloatingPointError(
                f"non-finite loss at optimizer step {step}; refusing to update parameters"
            )

        grad_norm = None
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip, error_if_nonfinite=True
            )
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        optimizer_was_run = scaler.get_scale() >= previous_scale
        if optimizer_was_run and ema_model is not None:
            ema_model.step(unwrap_model(model))

        if step % args.print_freq == 0 or step + 1 == args.total_steps:
            now = time.time()
            steps_since_log = step - last_log_step
            seconds_since_log = max(now - last_log_time, 1e-8)
            global_batch = args.batch_size * args.grad_accum_steps * world_size
            images_per_second = steps_since_log * global_batch / seconds_since_log
            peak_allocated_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            peak_reserved_gib = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
            loss_value = reduce_mean(
                (local_loss_sum / args.grad_accum_steps).item(), device
            )
            metric_values = {}
            for name, value_sum in local_metrics_sum.items():
                scalar_tensor = (
                    value_sum
                    if name.endswith("_max") or name.endswith("_min")
                    else value_sum / args.grad_accum_steps
                )
                if name.endswith("_max"):
                    reducer = reduce_max
                elif name.endswith("_min"):
                    reducer = reduce_min
                else:
                    reducer = reduce_mean
                metric_values[name] = reducer(scalar_tensor.item(), device)
            # A model can emit count-weighted statistics as
            # ``GROUP__QUANTITY_sum`` plus ``GROUP__count``.  Reducing both by
            # the same operation and taking their ratio gives the exact mean
            # over uneven per-rank/per-microbatch time-bin populations.
            weighted_sums = {
                name: value
                for name, value in metric_values.items()
                if "__" in name and name.endswith("_sum")
            }
            for sum_name, value_sum in weighted_sums.items():
                group, quantity_sum = sum_name.split("__", maxsplit=1)
                count_name = f"{group}__count"
                count = metric_values.get(count_name)
                if count is None:
                    continue
                quantity = quantity_sum[:-len("_sum")]
                metric_values[f"{group}__{quantity}"] = (
                    value_sum / count if count > 0.0 else 0.0
                )
                # The derived mean plus the shared count are sufficient for
                # logs/W&B; avoid emitting dozens of implementation-only sums.
                metric_values.pop(sum_name, None)
            if grad_norm is not None:
                metric_values["grad_norm"] = reduce_mean(
                    grad_norm.detach().float().item(), device
                )

            collapse_threshold = float(
                getattr(args, "patch_collapse_abort_corr", 0.0)
            )
            collapse_patience = int(
                getattr(args, "patch_collapse_abort_patience", 1)
            )
            corr_x = metric_values.get("patch_shift_corr_x16")
            corr_y = metric_values.get("patch_shift_corr_y16")
            template_fraction = metric_values.get(
                "patch_template_explained_frac"
            )
            sample_correlation = metric_values.get("sample_output_corr")
            template_threshold = float(
                getattr(args, "patch_collapse_abort_template_frac", -1.0)
            )
            sample_corr_threshold = float(
                getattr(args, "patch_collapse_abort_sample_corr", -1.0)
            )
            collapse_detected = (
                collapse_threshold > 0.0
                and corr_x is not None
                and corr_y is not None
                and template_fraction is not None
                and sample_correlation is not None
                and corr_x >= collapse_threshold
                and corr_y >= collapse_threshold
                and template_fraction >= template_threshold
                and sample_correlation >= sample_corr_threshold
            )
            collapse_bad_logs = collapse_bad_logs + 1 if collapse_detected else 0
            metric_values["patch_collapse_bad_log_count"] = float(
                collapse_bad_logs
            )
            collapse_should_abort = (
                collapse_threshold > 0.0
                and collapse_bad_logs >= collapse_patience
            )
            elapsed = str(datetime.timedelta(seconds=int(time.time() - start_time)))
            metric_text = " ".join(
                f"{name}={value:.6f}" for name, value in metric_values.items()
            )
            logger.info(
                "step=%d/%d lr=%.6g loss=%.6f img/s=%.1f "
                "peak_mem=%.2f/%.2fGiB %s elapsed=%s",
                step, args.total_steps, lr, loss_value, images_per_second,
                peak_allocated_gib, peak_reserved_gib, metric_text, elapsed,
            )
            if wandb_run is not None:
                wandb_metrics = {
                    "train/loss": loss_value,
                    "train/lr": lr,
                    "perf/images_per_second": images_per_second,
                    "perf/peak_memory_allocated_gib": peak_allocated_gib,
                    "perf/peak_memory_reserved_gib": peak_reserved_gib,
                }
                wandb_metrics.update({f"train/{k}": v for k, v in metric_values.items()})
                wandb_run.log(wandb_metrics, step=step)
            if collapse_should_abort:
                raise RuntimeError(
                    "patch-collapse guard triggered: horizontal/vertical "
                    f"shift correlations stayed above {collapse_threshold} "
                    f"for {collapse_bad_logs} logs (x={corr_x:.6f}, "
                    f"y={corr_y:.6f}, template={template_fraction:.6f}, "
                    f"sample_corr={sample_correlation:.6f})"
                )
            last_log_time = now
            last_log_step = step
            torch.cuda.reset_peak_memory_stats(device)

        adapt_steps = phase_adapt_steps(args)
        is_phase_boundary = adapt_steps > 0 and step + 1 == adapt_steps
        dense_save_every = int(getattr(args, "dense_save_every", 0))
        dense_save_start = int(getattr(args, "dense_save_start", 0))
        dense_save_end = int(getattr(args, "dense_save_end", -1))
        is_dense_save = (
            dense_save_every > 0
            and step >= dense_save_start
            and (dense_save_end < 0 or step <= dense_save_end)
            and (step - dense_save_start) % dense_save_every == 0
        )
        if (
            (step + 1) % args.save_every == 0
            or step + 1 == args.total_steps
            or is_phase_boundary
            or is_dense_save
        ):
            save_checkpoint(args, step, model, optimizer, ema_model)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    if wandb_run is not None:
        wandb_run.finish()


def get_args_parser():
    parser = argparse.ArgumentParser("JiT-B MeanFlow / stabilized continuous consistency training")
    parser.add_argument("--data_path", required=True)
    parser.add_argument(
        "--load_from",
        default=str(REPO_ROOT / "checkpoints/baseline/jit-b-16/checkpoint-last.pth"),
    )
    parser.add_argument("--checkpoint_key", default="model", choices=["auto", "model", "model_ema1", "model_ema2", "model_ema"])
    parser.add_argument("--output_dir", default="./work_dirs")
    parser.add_argument("--project", default="meanflow")
    parser.add_argument("--exp_name", default="jit_b_meanflow_1step")

    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps_per_epoch", type=int, default=1250)
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--print_freq", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument(
        "--dense_save_every",
        type=int,
        default=0,
        help=(
            "additional exact-step checkpoint cadence inside the interval "
            "selected by --dense_save_start/--dense_save_end; 0 disables"
        ),
    )
    parser.add_argument("--dense_save_start", type=int, default=0)
    parser.add_argument(
        "--dense_save_end",
        type=int,
        default=-1,
        help="inclusive final dense-save step; negative means no upper bound",
    )
    parser.add_argument("--keep_last", type=int, default=5)
    parser.add_argument("--resume_from", default=None)
    parser.add_argument("--disable_hflip", action="store_true")

    parser.add_argument("--P_mean", type=float, default=0.8)
    parser.add_argument("--P_std", type=float, default=0.8)
    parser.add_argument("--t_eps", type=float, default=5e-2)
    parser.add_argument("--noise_scale", type=float, default=1.0)
    parser.add_argument("--meanflow_delta", type=float, default=0.01)
    parser.add_argument("--dudt_weight", type=float, default=0.5)
    parser.add_argument("--dudt_drop_prob", type=float, default=0.75)
    parser.add_argument("--dudt_clip_norm", type=float, default=0.0)
    parser.add_argument("--loss_clip", type=float, default=0.0)
    parser.add_argument("--objective", default="meanflow", choices=["meanflow", "scm"])
    parser.add_argument("--sigma_data", type=float, default=0.5)
    parser.add_argument("--sigma_max", type=float, default=80.0)
    parser.add_argument("--tangent_norm_c", type=float, default=0.1)
    parser.add_argument("--tangent_warmup_steps", type=int, default=10000)
    parser.add_argument("--disable_adaptive_weighting", action="store_true")
    parser.add_argument("--adaptive_weight_hidden", type=int, default=64)
    parser.add_argument("--adaptive_weight_max", type=float, default=20.0)
    parser.add_argument("--jvp_dtype", default="amp", choices=["amp", "fp32"])

    parser.add_argument("--label_drop_prob", type=float, default=0.1)
    parser.add_argument("--attn_dropout", type=float, default=0.0)
    parser.add_argument("--proj_dropout", type=float, default=0.0)
    parser.add_argument("--grad_checkpointing", action="store_true")
    parser.add_argument("--compile", action="store_true")

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--blr", type=float, default=5e-5)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--lr_sched", default="cosine", choices=["cosine", "constant", "edm2"])
    parser.add_argument("--lr_ref_steps", type=float, default=35000.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--disable_ema", action="store_true")
    parser.add_argument("--ema_type", default="edm", choices=["const", "edm", "power"])
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--ema_halflife_kimg", type=float, default=500.0)
    parser.add_argument("--ema_sigma_rel", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enable_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="meanflow")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_name", default=None)
    return parser


if __name__ == "__main__":
    train(get_args_parser().parse_args())
