"""AMFD integration for the static FD branch.

Wires the vendored ``amfd_loss`` into this repository's training loop: the
argparse surface, per-judge amortizer construction, the alternating amortizer
update, the generator-side loss, and checkpoint state.

Structure and hyperparameter defaults follow the official AMFD release
(https://github.com/poppuppy/amfd, ``main_amfd.py``) so results here stay
comparable to upstream.  Notable consequences of that fidelity:

* Every static judge gets its own amortizer, as upstream does for its SIM
  stack.  Feature normalization is per encoder.
* The step samples twice: once under ``no_grad`` to fit the amortizers, once
  with grad for the generator.  This is upstream's ``update_amort`` followed by
  its generator block.
* AMFD features are **not** all-gathered.  The amortizer predicts conditional
  moments per sample, so there is no cross-rank statistic to pool; gradients
  are averaged over ranks at the parameter level instead.
* AMFD replaces the static FD loss rather than adding to it, and brings its own
  per-encoder normalization instead of ``fid / (fid.detach() + eps)``.
* The amortizer optimizer shards its state with ZeRO-1 on distributed launches,
  where upstream uses a replicated ``AdamW``.  This is a memory-layout
  deviation only; see :func:`_build_amort_optimizer` for why it leaves the
  update arithmetic alone.

The adversarial FD branch is untouched by everything in this module.
"""

import logging
import os

import torch

from amfd.amfd_loss import AmortizedFDLoss
from frechet_distance.judges import extract_judge_features
from frechet_distance.losses import load_mu_and_sigma_reference
from utils.distributed_util import all_reduce_grads, broadcast_module_params
from utils.grad_util import get_grad_norm

logger = logging.getLogger("FD_loss")

AMFD_CHECKPOINT_KEYS = ("amort_states", "amort_ema_states", "amort_optimizer", "amort_metadata")

_AMFD_METADATA_SCHEMA = 1

def add_amfd_args(parser):
    """Register the AMFD argparse surface.

    Every ``--amort_*`` default is copied from upstream ``main_amfd.py`` so a
    bare ``--amfd_static`` run matches upstream's argparse defaults.  Note that
    upstream's own ImageNet launcher overrides several of these (c2048/d16/a4,
    manual JVP, per-encoder generator normalization); the shipped scripts here
    do the same.

    ``--amfd_static`` is this repository's master switch and has no upstream
    counterpart: upstream has no static/adversarial split.
    """
    group = parser.add_argument_group("AMFD (amortized FD, static branch)")
    group.add_argument(
        "--amfd_static", action="store_true",
        help="replace the static FD loss with AMFD on every static judge; "
             "the adversarial FD branch is unaffected",
    )
    group.add_argument(
        "--amfd_log_fd_freq", type=int, default=50,
        help="log the plain FD every N steps as a diagnostic while AMFD supplies "
             "the gradient; 0 disables. Repository-specific: upstream computes no "
             "FD during training. Costs one eigendecomposition per judge and never "
             "affects gradients.",
    )

    group.add_argument("--amort_lr", type=float, default=1e-4)
    group.add_argument("--amort_beta1", type=float, default=0.9)
    group.add_argument("--amort_beta2", type=float, default=0.95)
    group.add_argument("--amort_weight_decay", type=float, default=0.0)
    group.add_argument("--amort_grad_clip", type=float, default=1.0)

    group.add_argument(
        "--no_amort_zero", action="store_false", dest="amort_zero",
        default=os.environ.get("AMFD_ZERO", "1") != "0",
        help="keep the amortizer optimizer state replicated instead of sharding "
             "it with ZeRO-1. Sharding is mathematically equivalent, so this is "
             "for isolating ZeRO's communication cost when benchmarking, not for "
             "changing results. Also settable as AMFD_ZERO=0.",
    )

    group.add_argument("--amort_model_channels", type=int, default=1024)
    group.add_argument("--amort_depth", type=int, default=8)
    group.add_argument("--amort_num_adaln_blocks", type=int, default=2)
    group.add_argument(
        "--amort_uncond", action="store_true", default=False,
        help="use a single unconditional amortizer label instead of class labels (AMFD-U)",
    )

    group.add_argument(
        "--amort_updates_per_gen_update", type=int, default=1,
        help="number of amortizer updates per generator update",
    )
    group.add_argument(
        "--amort_ema_decay", type=float, default=0.0,
        help="EMA decay for the amortizer used by the generator loss; 0 disables",
    )

    group.add_argument(
        "--amort_norm_stats_paths", type=str, nargs="+", default=None,
        help="per-dim feature normalization .npz path(s), one per repr model. "
             "Uses mu and sqrt(diag(sigma)) for the selected pool type. "
             "Defaults to each judge's own FD reference stats.",
    )
    group.add_argument(
        "--amort_norm_mu", type=float, nargs="+", default=None,
        help="scalar feature normalization mean(s); one value or one per repr model",
    )
    group.add_argument(
        "--amort_norm_sigma", type=float, nargs="+", default=None,
        help="scalar feature normalization std(s); one value or one per repr model",
    )

    group.add_argument(
        "--amort_t", type=float, default=0.25,
        help="fixed interpolation time; default 0.25",
    )
    group.add_argument(
        "--amort_prediction_target", type=str, default="v", choices=("v", "x"),
        help="amortizer parameterization: 'v' velocity target, or 'x' converted to velocity MSE",
    )
    group.add_argument(
        "--amort_diff_batch_mul", type=int, default=4,
        help="number of independent t/noise samples per feature sample",
    )
    group.add_argument(
        "--amort_jvp_impl", type=str, default="torch_func", choices=("torch_func", "manual"),
        help="JVP implementation for the amortizer",
    )
    group.add_argument(
        "--no_amort_jacobi_gen_loss", action="store_false", dest="amort_jacobi_gen_loss",
        default=True,
        help="disable P_g(A_g-A_r)P_g u in the generator covariance loss",
    )
    group.add_argument(
        "--no_amort_share_real_fake_mlp", action="store_false",
        dest="amort_share_real_fake_mlp", default=True,
        help="disable the shared real/fake amortizer network",
    )
    group.add_argument(
        "--no_amort_train_real_branch", action="store_false",
        dest="amort_train_real_branch", default=True,
        help="disable the real-domain estimator loss during amortizer updates",
    )
    group.add_argument(
        "--amort_init_checkpoint", type=str, default=None,
        help="initialize the amortizers from a checkpoint containing amort_states",
    )
    group.add_argument(
        "--amort_normalize_gen_loss_per_encoder", action="store_true",
        help="divide each encoder's amortizer generator loss by a detached "
             "per-encoder magnitude before weighting",
    )
    group.add_argument(
        "--amort_gen_loss_norm_eps", type=float, default=0.01,
        help="epsilon for --amort_normalize_gen_loss_per_encoder",
    )
    group.add_argument(
        "--amort_gen_loss_norm_power", type=float, default=1.0,
        help="exponent p for the generator normalization denominator: "
             "(sum(gen_grad_proxy^2) ** p) + eps",
    )
    group.add_argument(
        "--amort_sequential_repr_backward", action="store_true",
        help="backward each encoder's generator loss immediately instead of summing "
             "first; same gradient, lower peak memory for multi-encoder training",
    )
    group.add_argument(
        "--amort_sequential_amort_backward", action="store_true",
        help="backward each encoder's amortizer loss immediately instead of summing "
             "first; same gradient, lower peak memory for multi-encoder training",
    )
    return group


def amfd_enabled(args) -> bool:
    return bool(getattr(args, "amfd_static", False))


def resolve_amfd_args(args):
    """Validate the AMFD normalization arguments.

    Mirrors upstream's checks.  Upstream *requires* either
    ``--amort_norm_stats_paths`` or both ``--amort_norm_mu`` and
    ``--amort_norm_sigma``; here all three may be omitted, in which case each
    amortizer derives its normalizer from that judge's own FD reference stats
    (``--fd_repr_stats_paths``), which is the same file upstream would be
    pointed at.
    """
    import os

    if not amfd_enabled(args):
        return

    num = len(args.fd_repr_models)

    if args.amort_norm_stats_paths is not None:
        if len(args.amort_norm_stats_paths) != num:
            raise ValueError(
                f"--amort_norm_stats_paths must contain {num} paths "
                f"(one per --fd_repr_models entry); got {len(args.amort_norm_stats_paths)}"
            )
        missing = [p for p in args.amort_norm_stats_paths if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(
                "--amort_norm_stats_paths contains missing files: " + ", ".join(missing)
            )
    elif args.amort_norm_mu is not None or args.amort_norm_sigma is not None:
        if args.amort_norm_mu is None or args.amort_norm_sigma is None:
            raise ValueError(
                "Scalar amortizer normalization requires both --amort_norm_mu "
                "and --amort_norm_sigma"
            )
        args.amort_norm_mu = [float(v) for v in args.amort_norm_mu]
        args.amort_norm_sigma = [float(v) for v in args.amort_norm_sigma]
        if len(args.amort_norm_mu) == 1:
            args.amort_norm_mu = args.amort_norm_mu * num
        elif len(args.amort_norm_mu) != num:
            raise ValueError(f"--amort_norm_mu must contain either 1 value or {num} values")
        if len(args.amort_norm_sigma) == 1:
            args.amort_norm_sigma = args.amort_norm_sigma * num
        elif len(args.amort_norm_sigma) != num:
            raise ValueError(f"--amort_norm_sigma must contain either 1 value or {num} values")

    if args.amort_init_checkpoint is not None and not os.path.isfile(args.amort_init_checkpoint):
        raise FileNotFoundError(f"--amort_init_checkpoint not found: {args.amort_init_checkpoint}")


def resolve_amfd_judges(judges, args):
    """Return the judges that get an amortizer.

    Every static judge, matching upstream where all ``--fd_repr_models``
    entries get one.  The adversarial judges are a separate list in this
    repository and are never included.
    """
    if not amfd_enabled(args):
        return []
    return list(judges)


def _build_feature_normalizer(args, judge, idx):
    """Return ``(feature_mean, feature_std, label)`` for one judge.

    Follows upstream's ``build_amort_feature_normalizer``: per-dim mean from
    ``mu`` and per-dim std from ``sqrt(diag(sigma))``.
    """
    feat_dim = int(judge["feat_dim"])
    pool_type = judge.get("pool_type", "cls")

    if args.amort_norm_stats_paths is not None:
        stats_path = args.amort_norm_stats_paths[idx]
        mu_ref, sigma_ref = load_mu_and_sigma_reference(stats_path, pool_type=pool_type)
        source = f"per_dim(stats={stats_path})"
    elif args.amort_norm_mu is not None:
        mean_value = args.amort_norm_mu[idx]
        std_value = args.amort_norm_sigma[idx]
        feature_mean = torch.full((feat_dim,), float(mean_value), dtype=torch.float32)
        feature_std = torch.full((feat_dim,), float(std_value), dtype=torch.float32)
        return feature_mean, feature_std, f"scalar(mu={mean_value}, sigma={std_value})"
    else:
        mu_ref = judge["mu_ref"]
        sigma_ref = judge["sigma_ref"]
        source = "per_dim(fd_repr_stats)"

    if mu_ref.ndim != 1 or mu_ref.numel() != feat_dim:
        raise ValueError(
            f"AMFD normalizer for '{judge['name']}': mean dim={mu_ref.numel()} "
            f"!= feat_dim={feat_dim}"
        )
    expected_shape = (feat_dim, feat_dim)
    if sigma_ref.ndim != 2 or tuple(sigma_ref.shape) != expected_shape:
        raise ValueError(
            f"AMFD normalizer for '{judge['name']}': sigma shape="
            f"{tuple(sigma_ref.shape)} != {expected_shape}"
        )
    var = sigma_ref.diagonal().float()
    if torch.any(~torch.isfinite(var)):
        raise ValueError(
            f"AMFD normalizer for '{judge['name']}': non-finite variance "
            f"for pool_type={pool_type!r}"
        )
    feature_mean = mu_ref.float().cpu()
    feature_std = var.clamp_min(1e-6).sqrt().cpu()
    return feature_mean, feature_std, source


def _zero_optimizer_class():
    """``ZeroRedundancyOptimizer``, or None on builds that lack it."""
    try:
        from torch.distributed.optim import ZeroRedundancyOptimizer
    except ImportError:
        return None
    return ZeroRedundancyOptimizer


def _is_zero_optimizer(optimizer):
    """True when *optimizer* keeps only this rank's shard of the state."""
    zero_cls = _zero_optimizer_class()
    return zero_cls is not None and isinstance(optimizer, zero_cls)


def _build_amort_optimizer(params, args):
    """Build the amortizer optimizer, sharding its state when distributed.

    AdamW carries two fp32 moments per parameter.  For the shipped c2048d16a4
    stack (1114M amortizer parameters over three encoders) that is 8.30 GiB
    resident on every rank, the largest single block in the step, and plain data
    parallelism replicates rather than splits it.  ``ZeroRedundancyOptimizer``
    partitions the moments over the process group, so each rank holds
    ``1 / world_size`` of them.

    This changes memory layout, not arithmetic.  :func:`update_amortizers`
    all-reduces amortizer gradients before stepping, so every rank enters
    ``step()`` with identical gradients and an identical clip scale, updates its
    own shard exactly as the replicated optimizer would have, and ZeRO then
    broadcasts the updated parameters.  Every hyperparameter is passed through
    unchanged.

    A single-process run has no group to shard over and falls back to plain
    AdamW; so does a build without ``torch.distributed.optim``.
    """
    params = list(params)
    defaults = dict(
        lr=args.amort_lr,
        betas=(args.amort_beta1, args.amort_beta2),
        weight_decay=args.amort_weight_decay,
    )

    zero_cls = _zero_optimizer_class()
    sharding_disabled = not getattr(args, "amort_zero", True)
    if sharding_disabled or not torch.distributed.is_initialized() or zero_cls is None:
        reason = (
            "disabled by --no_amort_zero/AMFD_ZERO=0" if sharding_disabled
            else "single process" if not torch.distributed.is_initialized()
            else "torch.distributed.optim unavailable"
        )
        logger.info("[AMFD] amortizer optimizer: AdamW (replicated, %s)", reason)
        return torch.optim.AdamW(params, **defaults)

    optimizer = zero_cls(params, optimizer_class=torch.optim.AdamW, **defaults)
    world_size = torch.distributed.get_world_size()
    moment_bytes = sum(p.numel() for p in params) * 8  # exp_avg + exp_avg_sq, fp32
    logger.info(
        "[AMFD] amortizer optimizer: ZeRO-1 AdamW over %d rank(s) -- "
        "moments %.2f GiB replicated -> %.2f GiB per rank",
        world_size,
        moment_bytes / 2 ** 30,
        moment_bytes / world_size / 2 ** 30,
    )
    return optimizer


def build_amfd_amortizers(judges, args):
    """Attach an amortizer to every AMFD judge and build their optimizer.

    Returns ``(amort_judges, amort_modules, optimizer_amort)``.  When AMFD is
    off this is ``([], None, None)`` and nothing is allocated.

    Each judge gains ``judge["amort"]`` and, when ``--amort_ema_decay > 0``,
    ``judge["amort_ema"]``.  Construction order and the optimizer follow
    upstream ``main_amfd.py``.
    """
    import copy

    from torch import nn

    amort_judges = resolve_amfd_judges(judges, args)
    if not amort_judges:
        return [], None, None

    use_amort_ema = args.amort_ema_decay > 0.0
    effective_num_classes = 1 if args.amort_uncond else int(args.num_classes)
    amort_modules = nn.ModuleList()

    for idx, judge in enumerate(amort_judges):
        feature_mean, feature_std, norm_label = _build_feature_normalizer(args, judge, idx)

        amort = AmortizedFDLoss(
            feat_dim=int(judge["feat_dim"]),
            model_channels=args.amort_model_channels,
            depth=args.amort_depth,
            num_classes=effective_num_classes,
            num_adaln_blocks=args.amort_num_adaln_blocks,
            feature_mean=feature_mean,
            feature_std=feature_std,
            grad_checkpointing=args.grad_checkpointing,
            t=args.amort_t,
            diff_batch_mul=args.amort_diff_batch_mul,
            train_real_branch=args.amort_train_real_branch,
            jvp_impl=args.amort_jvp_impl,
            jacobi_generator_loss=args.amort_jacobi_gen_loss,
            share_real_fake_mlp=args.amort_share_real_fake_mlp,
            prediction_target=args.amort_prediction_target,
            normalize_generator_loss=args.amort_normalize_gen_loss_per_encoder,
            generator_loss_norm_eps=args.amort_gen_loss_norm_eps,
            generator_loss_norm_power=args.amort_gen_loss_norm_power,
        ).cuda()

        if torch.distributed.is_initialized():
            broadcast_module_params(amort, src=0)
        judge["amort"] = amort
        if use_amort_ema:
            judge["amort_ema"] = copy.deepcopy(amort).cuda().eval().requires_grad_(False)
        amort_modules.append(amort)

        param_count = sum(p.numel() for p in amort.parameters())
        logger.info(
            f"[AMFD] Repr '{judge['name']}': feat_dim={judge['feat_dim']}, "
            f"weight={judge['weight']}, pool={judge.get('pool_type', 'cls')}, "
            f"norm={norm_label}, amort_arch=global_mlp, params={param_count / 1e6:.1f}M, "
            f"share_real_fake_mlp={args.amort_share_real_fake_mlp}, "
            f"ema_decay={args.amort_ema_decay}"
        )

    optimizer_amort = _build_amort_optimizer(amort_modules.parameters(), args)

    _load_pretrained_amort_if_requested(amort_judges, args, use_amort_ema)
    _freeze_amort_real_branch_if_requested(amort_judges, args, log=True)

    return amort_judges, amort_modules, optimizer_amort


def _load_pretrained_amort_if_requested(judges, args, use_amort_ema):
    """Initialize amortizers from ``--amort_init_checkpoint``. Upstream logic."""
    ckpt_path = args.amort_init_checkpoint
    if not ckpt_path:
        return False

    is_distributed = torch.distributed.is_initialized()
    rank = torch.distributed.get_rank() if is_distributed else 0
    if rank == 0:
        payload = torch.load(ckpt_path, map_location="cpu")
        if not isinstance(payload, dict) or not isinstance(payload.get("amort_states"), dict):
            raise ValueError(
                f"--amort_init_checkpoint must point to a checkpoint with amort_states: {ckpt_path}"
            )
        amort_states = payload["amort_states"]
        for judge in judges:
            name = judge["name"]
            if name not in amort_states:
                raise ValueError(f"{ckpt_path} does not contain amort state for '{name}'")
            judge["amort"].load_state_dict(amort_states[name], strict=True)
            if use_amort_ema:
                judge["amort_ema"].load_state_dict(judge["amort"].state_dict(), strict=True)
            logger.info("[AMFD] Initialized '%s' amortizer state from %s", name, ckpt_path)

    if is_distributed:
        for judge in judges:
            broadcast_module_params(judge["amort"], src=0)
            if use_amort_ema:
                broadcast_module_params(judge["amort_ema"], src=0)
    return True


def _freeze_amort_real_branch_if_requested(judges, args, log: bool = True):
    """Upstream ``freeze_amort_real_branch_if_requested``."""
    if args.amort_train_real_branch:
        return False
    froze_any = False
    for judge in judges:
        froze_any = judge["amort"].freeze_real_branch() or froze_any
    if froze_any and log:
        logger.info("[AMFD] real-domain estimator is frozen")
    return froze_any


def _set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad_(flag)


@torch.no_grad()
def _update_module_ema(ema_module, online_module, decay):
    """Upstream ``update_module_ema``."""
    for ema_param, online_param in zip(ema_module.parameters(), online_module.parameters()):
        ema_param.lerp_(online_param.detach(), 1.0 - decay)
    for ema_buffer, online_buffer in zip(ema_module.buffers(), online_module.buffers()):
        ema_buffer.copy_(online_buffer.detach())


def _accum_log(log_sums: dict, key: str, value, scale: float = 1.0):
    """Upstream ``_accum_log``."""
    if isinstance(value, torch.Tensor):
        value = value.detach()
    else:
        value = float(value)
    log_sums[key] = log_sums.get(key, 0.0) + value * scale


def _extract_judge_features_real_fake(judge, real_images, fake_images):
    """Extract real/fake features in one forward pass. Upstream helper.

    One concatenated forward keeps any batch-dependent behaviour in the encoder
    identical across the two halves.
    """
    if real_images.shape != fake_images.shape:
        raise ValueError(
            f"real/fake image shapes must match, got "
            f"{tuple(real_images.shape)} and {tuple(fake_images.shape)}"
        )
    batch = real_images.shape[0]
    h = extract_judge_features(judge, torch.cat([real_images, fake_images], dim=0))
    return h[:batch], h[batch:]


def _amfd_sampling_args(sampling_args):
    """Sampling args for AMFD's own no-grad sample.

    The training loop mutates its ``sampling_args`` in place (``return_velocity``
    and ``t_start`` for random-timestep training).  AMFD samples from pure noise
    like upstream's ``generate_sampled``, so those two keys are dropped while
    ``t_min``/``t_max``/``cfg``/``num_steps``/``t_start_min`` carry over.
    """
    clean = dict(sampling_args)
    clean.pop("return_velocity", None)
    clean.pop("t_start", None)
    return clean


@torch.no_grad()
def _generate_fake_for_amort(model_wo_ddp, args, y, sampling_args, tokenizer=None):
    """Sample a fake batch in [0, 1] with no autograd graph."""
    input_shape = (args.input_channels, args.input_size, args.input_size)
    z = torch.randn(y.shape[0], *input_shape, device="cuda") * args.noise_scale
    sampled = model_wo_ddp.sample_images_with_grad(z, y, sampling_args=sampling_args)
    if tokenizer is not None:
        sampled = tokenizer.decode(
            tokenizer.denormalize_z(sampled),
            decode_bsz=args.vae_decode_bsz,
        )
    return (sampled * 0.5 + 0.5).clamp(0.0, 1.0)


def update_amortizers(
    model_wo_ddp,
    amort_judges,
    amort_modules,
    optimizer_amort,
    real_batch_fn,
    sampling_args,
    args,
    tokenizer=None,
):
    """Fit the amortizers to the current real/fake conditional moments.

    Upstream's ``update_amort``: for each update, draw a real batch, sample a
    fake batch under ``no_grad`` with the *same* labels, fit every amortizer,
    then step.  Non-finite grad norms skip the step, as upstream does.

    Returns ``(amort_loss, grad_norm, loss_dict, labels)``.  ``labels`` are the
    real-batch labels from the last update and must be reused for the generator
    step so the conditional moments line up.
    """
    if not amort_judges:
        return None, None, {}, None

    updates = max(1, args.amort_updates_per_gen_update)
    use_amort_ema = args.amort_ema_decay > 0.0
    clean_sampling_args = _amfd_sampling_args(sampling_args)

    amort_loss_meter = torch.zeros((), device="cuda")
    grad_norm_meter = torch.zeros((), device="cuda")
    loss_dict = {}
    labels_for_gen = None

    for update_idx in range(updates):
        _set_requires_grad(amort_modules, True)
        _freeze_amort_real_branch_if_requested(amort_judges, args, log=False)
        optimizer_amort.zero_grad(set_to_none=True)

        real, y_real = real_batch_fn()
        real = real.cuda(non_blocking=True).clamp(0.0, 1.0)
        y = y_real.cuda(non_blocking=True).long()

        fake = _generate_fake_for_amort(
            model_wo_ddp, args, y, clean_sampling_args, tokenizer=tokenizer,
        )
        labels_for_gen = y.detach()
        amort_labels = torch.zeros_like(y) if args.amort_uncond else y

        amort_loss_total = torch.zeros((), device="cuda")
        for judge in amort_judges:
            with torch.no_grad():
                h_real, h_fake = _extract_judge_features_real_fake(judge, real, fake)

            loss, logs = judge["amort"].amort_loss(
                h_real=h_real,
                h_fake=h_fake,
                labels=amort_labels,
            )
            weighted = judge["weight"] * loss
            if args.amort_sequential_amort_backward:
                weighted.backward()
                amort_loss_total = amort_loss_total + weighted.detach()
            else:
                amort_loss_total = amort_loss_total + weighted

            log_scale = 1.0 / updates
            for k, v in logs.items():
                _accum_log(loss_dict, f"{judge['name']}/{k}", v, log_scale)
            if args.amort_sequential_amort_backward:
                del h_real, h_fake, loss, weighted, logs

        if not args.amort_sequential_amort_backward:
            amort_loss_total.backward()
            del h_real, h_fake, loss, weighted, logs
        del real, fake

        if torch.distributed.is_initialized():
            all_reduce_grads(amort_modules)

        grad_norm = (
            torch.nn.utils.clip_grad_norm_(amort_modules.parameters(), args.amort_grad_clip)
            if args.amort_grad_clip > 0
            else get_grad_norm(amort_modules.parameters())
        )

        if torch.isfinite(grad_norm):
            optimizer_amort.step()
            if use_amort_ema:
                for judge in amort_judges:
                    _update_module_ema(
                        judge["amort_ema"], judge["amort"], args.amort_ema_decay,
                    )
        else:
            logger.warning(
                f"[step {args.current_step}] NaN/Inf AMFD grad_norm — skipping AMFD "
                f"update {update_idx + 1}/{updates}"
            )
        optimizer_amort.zero_grad(set_to_none=True)

        amort_loss_meter = amort_loss_meter + amort_loss_total.detach()
        grad_norm_meter = grad_norm_meter + grad_norm.detach()

    amort_loss_meter = amort_loss_meter / updates
    grad_norm_meter = grad_norm_meter / updates

    # The generator step must not accumulate into the amortizers.
    _set_requires_grad(amort_modules, False)

    loss_dict["amfd/amort_total"] = amort_loss_meter.detach()
    loss_dict["amfd/grad_norm"] = grad_norm_meter.detach()
    return amort_loss_meter, grad_norm_meter, loss_dict, labels_for_gen


def amfd_generator_loss(judge, features, labels, args):
    """AMFD generator loss for one judge.

    ``features`` must carry the autograd graph back to the generator and must
    **not** be all-gathered: the amortizer predicts per-sample conditional
    moments, so there is no cross-rank statistic to pool.

    Returns ``(weighted_loss, logs)``.  The loss already carries AMFD's own
    per-encoder normalization, so the caller must not apply the static FD
    ``fid / (fid.detach() + eps)`` scaling on top.
    """
    amort = judge["amort_ema"] if args.amort_ema_decay > 0.0 else judge["amort"]
    amort_labels = torch.zeros_like(labels) if args.amort_uncond else labels
    loss, logs = amort.generator_loss(h_fake=features, labels=amort_labels)
    return judge["weight"] * loss, logs


def _amort_optimizer_state_dict(optimizer_amort):
    """Global amortizer optimizer state, or None if this rank does not hold it.

    Under ZeRO each rank owns only its own shard of the moments, so the shards
    have to be gathered before anything is written.  ``consolidate_state_dict``
    is a collective: every rank must enter it, and only the ``to=`` rank may
    then call ``state_dict()`` -- elsewhere that raises.  Non-target ranks
    therefore return None, which costs nothing because ``save_checkpoint``
    returns early on every rank but the main one.

    The consolidated dict uses global parameter indexing, so it is
    interchangeable with a plain AdamW state dict over the same parameters.

    Note that the returned dict holds live references to the moment tensors, so
    it must be serialized or copied before the optimizer steps again.  The
    checkpoint path already does: ``save_checkpoint`` either snapshots to CPU
    for the async saver or hands the dict straight to ``torch.save``.
    """
    if not _is_zero_optimizer(optimizer_amort):
        return optimizer_amort.state_dict()

    optimizer_amort.consolidate_state_dict(to=0)
    if torch.distributed.get_rank() != 0:
        return None
    return optimizer_amort.state_dict()


def save_amfd_state(amort_judges, optimizer_amort, args):
    """Collect AMFD state for the checkpoint. Upstream key layout.

    Must be called on every rank, not under a rank-0 guard: the ZeRO shard
    gather inside :func:`_amort_optimizer_state_dict` is a collective.
    """
    if not amort_judges:
        return {}
    state = {
        "amort_states": {j["name"]: j["amort"].state_dict() for j in amort_judges},
        "amort_metadata": {
            "schema": _AMFD_METADATA_SCHEMA,
            "ema_decay": args.amort_ema_decay,
            "share_real_fake_mlp": args.amort_share_real_fake_mlp,
            "uncond": bool(args.amort_uncond),
            "jvp_impl": args.amort_jvp_impl,
            "jacobi_gen_loss": args.amort_jacobi_gen_loss,
            "prediction_target": args.amort_prediction_target,
            "t": args.amort_t,
        },
    }
    optimizer_state = _amort_optimizer_state_dict(optimizer_amort)
    if optimizer_state is not None:
        state["amort_optimizer"] = optimizer_state
    if args.amort_ema_decay > 0.0:
        state["amort_ema_states"] = {
            j["name"]: j["amort_ema"].state_dict() for j in amort_judges
        }
    return state


def load_amfd_state(amort_judges, optimizer_amort, extra, args):
    """Restore AMFD state from a resume checkpoint.

    Upstream refuses to resume across an ``--amort_uncond`` flip because the
    amortizer's label space changes; the same check applies here.  Returns True
    when state was restored.
    """
    if not amort_judges or not extra or "amort_states" not in extra:
        if amort_judges and extra and "amort_states" not in extra:
            logger.warning(
                "[AMFD] Resume checkpoint has no amortizer state — amortizers start fresh"
            )
        return False

    metadata = extra.get("amort_metadata") or {}
    if metadata.get("schema") != _AMFD_METADATA_SCHEMA:
        raise ValueError(
            f"Resume checkpoint does not use amort_metadata schema="
            f"{_AMFD_METADATA_SCHEMA}"
        )
    saved_uncond = bool(metadata["uncond"])
    if saved_uncond != bool(args.amort_uncond):
        raise ValueError(
            f"Resume checkpoint amort_uncond={saved_uncond!r} does not match "
            f"--amort_uncond={bool(args.amort_uncond)!r}"
        )

    states = extra["amort_states"]
    for judge in amort_judges:
        name = judge["name"]
        if name not in states:
            raise ValueError(f"Resume checkpoint is missing amortizer state for {name!r}")
        judge["amort"].load_state_dict(states[name], strict=True)
    logger.info("[AMFD] Restored amortizer module states")

    use_amort_ema = args.amort_ema_decay > 0.0
    if use_amort_ema:
        ema_states = extra.get("amort_ema_states")
        if not ema_states:
            for judge in amort_judges:
                judge["amort_ema"].load_state_dict(judge["amort"].state_dict(), strict=True)
            logger.warning(
                "[AMFD] Resume checkpoint has no amortizer EMA state — "
                "seeded the EMA from the restored online weights"
            )
        else:
            for judge in amort_judges:
                name = judge["name"]
                if name not in ema_states:
                    raise ValueError(
                        f"Resume checkpoint is missing amortizer EMA state for {name!r}"
                    )
                judge["amort_ema"].load_state_dict(ema_states[name], strict=True)
            logger.info("[AMFD] Restored amortizer EMA module states")

    _load_amort_optimizer_state(optimizer_amort, extra)
    return True


def _load_amort_optimizer_state(optimizer_amort, extra):
    """Restore the amortizer optimizer state on this rank.

    The checkpoint always holds the *global* state -- consolidated on rank 0 at
    save time -- and ``ckpt_resume`` loads the file on every rank, so this is a
    plain per-rank call in both the ZeRO and the replicated case.  ZeRO picks
    its own shard out of the global dict and drops the rest, which is also what
    makes resuming onto a different world size work: the partition is recomputed
    from the new group size.

    A checkpoint saved before ZeRO landed, or written by a single-process run,
    carries a replicated AdamW dict over the same parameters in the same order;
    the two layouts share global indexing, so either loads into either.

    ZeRO edits the dict it is handed, replacing the entries this rank does not
    own with None.  That is safe here: each rank loaded its own copy from disk,
    nothing else reads ``amort_optimizer`` afterwards, and the checkpoint writer
    rebuilds its payload from :func:`save_amfd_state` rather than from *extra*.
    """
    saved = extra.get("amort_optimizer")
    if saved is None:
        logger.warning(
            "[AMFD] Resume checkpoint has no amortizer optimizer state — "
            "the amortizer moments start fresh"
        )
        return False
    optimizer_amort.load_state_dict(saved)
    logger.info("[AMFD] Restored amortizer optimizer state")
    return True


def log_amfd_config(args, amort_judges):
    """Log the resolved AMFD configuration once at startup."""
    if not amort_judges:
        return
    variant = "AMFD-U (unconditional amortizer)" if args.amort_uncond else "AMFD-C (conditional)"
    logger.info(
        f"[AMFD] enabled on the static FD branch: {variant}, "
        f"judges={[j['name'] for j in amort_judges]}"
    )
    logger.info(
        f"[AMFD] amortizer: c{args.amort_model_channels}d{args.amort_depth}"
        f"a{args.amort_num_adaln_blocks}, t={args.amort_t}, "
        f"diff_batch_mul={args.amort_diff_batch_mul}, jvp={args.amort_jvp_impl}, "
        f"share_real_fake_mlp={args.amort_share_real_fake_mlp}, "
        f"prediction_target={args.amort_prediction_target}, "
        f"jacobi_gen_loss={args.amort_jacobi_gen_loss}, "
        f"train_real_branch={args.amort_train_real_branch}"
    )
    logger.info(
        f"[AMFD] optim: lr={args.amort_lr}, betas=({args.amort_beta1}, {args.amort_beta2}), "
        f"wd={args.amort_weight_decay}, grad_clip={args.amort_grad_clip}, "
        f"updates_per_gen_update={args.amort_updates_per_gen_update}, "
        f"ema_decay={args.amort_ema_decay}"
    )
    logger.info(
        f"[AMFD] gen loss: normalize_per_encoder="
        f"{args.amort_normalize_gen_loss_per_encoder}, "
        f"norm_eps={args.amort_gen_loss_norm_eps}, "
        f"norm_power={args.amort_gen_loss_norm_power}"
    )
    logger.info(
        "[AMFD] the static FD loss is replaced by AMFD; the adversarial FD "
        "branch is unaffected"
    )
