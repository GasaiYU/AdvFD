"""
EMA (Exponential Moving Average) utilities for model weight averaging.

three schedule modes via ema_type:
  - "const": fixed decay rate (e.g. 0.9999)
  - "edm":   step-dependent decay via halflife ramp-up (Karras et al. 2024)
  - "power": EDM2 power-function EMA parameterized by relative std

usage:
    ema = EMAModel(model, ema_type="const", values=[0.9999, 0.9996])
    ema = EMAModel(model, ema_type="edm", values=[500, 1000], batch_size=1024)

    # training: call after optimizer.step()
    ema.step(model)

    # evaluation: context manager swaps in EMA weights, auto-restores on exit
    with ema.swap(model):                       # default (first) copy
        evaluate(model)
    with ema.swap(model, label="0.9996"):       # specific copy
        evaluate(model)
    with ema.swap(model, label="online"):        # no-op (online model, no swap)
        evaluate(model)

    # checkpoint
    torch.save({"ema": ema.state_dict()}, path)
    ema.load_state_dict(checkpoint["ema"])
"""

from __future__ import annotations

import logging
from functools import lru_cache
from contextlib import contextmanager, nullcontext

import torch

from utils.runtime_util import normalize_param_name

logger = logging.getLogger("FD_loss")


def _fsdp_full_param_context(model: torch.nn.Module):
    """Materialize full parameters while EMA reads/writes FSDP-wrapped modules."""
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    except ImportError:
        return nullcontext()

    fsdp_modules = [m for m in model.modules() if isinstance(m, FSDP)]
    if not fsdp_modules:
        return nullcontext()
    return FSDP.summon_full_params(fsdp_modules[0], recurse=True, writeback=True)


def _has_fsdp(model: torch.nn.Module) -> bool:
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    except ImportError:
        return False
    return any(isinstance(m, FSDP) for m in model.modules())


def const_schedule(step: int, batch_size: int, value: float) -> float:
    """constant decay — returns value as-is."""
    return value


def edm_schedule(step: int, batch_size: int, halflife_kimg: float) -> float:
    """edm-style decay (Karras et al. 2024): halflife ramps up during first 5% of training."""
    halflife_nimg = halflife_kimg * 1000
    rampup_ratio = 0.05
    halflife_nimg = min(halflife_nimg, step * batch_size * rampup_ratio)
    return 0.5 ** (batch_size / max(halflife_nimg, 1e-8))


@lru_cache(maxsize=None)
def _sigma_rel_to_gamma(sigma_rel: float) -> float:
    max_sigma_rel = (1.0 / 12.0) ** 0.5
    if not (0.0 < sigma_rel <= max_sigma_rel):
        raise ValueError(
            f"power EMA sigma_rel must be in (0, {max_sigma_rel}], got {sigma_rel}"
        )

    def sigma_for_gamma(gamma):
        return ((gamma + 1.0) / ((gamma + 2.0) ** 2 * (gamma + 3.0))) ** 0.5

    lo, hi = 0.0, 1.0
    while sigma_for_gamma(hi) > sigma_rel:
        hi *= 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if sigma_for_gamma(mid) > sigma_rel:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def power_schedule(step: int, batch_size: int, sigma_rel: float) -> float:
    """EDM2 power-function EMA with the requested relative profile std."""
    del batch_size
    gamma = _sigma_rel_to_gamma(sigma_rel)
    return ((step - 1.0) / max(float(step), 1.0)) ** (gamma + 1.0)


SCHEDULES = {"const": const_schedule, "edm": edm_schedule, "power": power_schedule}


class EMAModel:
    """
    exponential moving average of model parameters.

    maintains one or more shadow copies identified by string labels.
    supports constant and step-dependent decay schedules.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        ema_type: str = "const",
        values: list[float] | None = None,
        batch_size: int = 1,
    ):
        """
        args:
            model:      model whose parameters will be tracked.
            ema_type:   "const", "edm", or EDM2 "power".
            values:     for "const": list of decay rates (default [0.9999]).
                        for "edm": list of halflife in kimg (default [500, 1000, 2000]).
                        for "power": relative std values (e.g. [0.05]).
            batch_size: global batch size, used by edm schedule.
        """
        if ema_type not in SCHEDULES:
            raise ValueError(f"unknown ema_type '{ema_type}'. use one of {list(SCHEDULES.keys())}.")

        self.schedule_fn = SCHEDULES[ema_type]
        self.batch_size = batch_size
        self.step_count: int = 0

        # build (label, value) pairs
        if ema_type == "const":
            vals = values or [0.9999, 0.9996]
            self.schedules = [(str(v), float(v)) for v in vals]
        elif ema_type == "edm":
            vals = values or [500, 1000, 2000]
            self.schedules = [(f"edm_{v}", float(v)) for v in vals]
        else:
            vals = values or [0.05]
            self.schedules = [(f"power_{v}", float(v)) for v in vals]
            for value in vals:
                _sigma_rel_to_gamma(float(value))

        self.labels = [label for label, _ in self.schedules]

        # shadow parameters: {label: {normalized_name: tensor}}
        self.shadows: dict[str, dict[str, torch.Tensor]] = {}
        for label, _ in self.schedules:
            self.shadows[label] = {
                normalize_param_name(n): (p.clone().detach() if p.requires_grad else p)
                for n, p in model.named_parameters()
            }

    @property
    def default_label(self) -> str:
        return self.labels[0]

    @torch.no_grad()
    def step(self, model: torch.nn.Module) -> None:
        """update all EMA copies from current model weights. call once per training step."""
        self.step_count += 1
        with _fsdp_full_param_context(model):
            named_params = [
                (normalize_param_name(name), param)
                for name, param in model.named_parameters()
            ]
            for label, value in self.schedules:
                decay = self.schedule_fn(self.step_count, self.batch_size, value)
                shadow = self.shadows[label]
                trainable_shadows = []
                trainable_params = []
                for name, param in named_params:
                    if param.requires_grad:
                        trainable_shadows.append(shadow[name])
                        trainable_params.append(param.data)
                    else:
                        shadow[name].copy_(param.data)
                if trainable_shadows:
                    torch._foreach_lerp_(
                        trainable_shadows, trainable_params, 1.0 - decay
                    )

    @contextmanager
    def swap(self, model: torch.nn.Module, label: str | None = None):
        """temporarily replace model weights with EMA weights; restores on exit.

        - ``swap(model)``                 -> swap in the default (first) EMA copy.
        - ``swap(model, label="0.9999")`` -> swap in the named EMA copy.
        - ``swap(model, label="online")`` -> no-op (online model, no swap).
        """
        if label == "online":
            yield
            return
        label = label or self.default_label
        if label not in self.shadows:
            raise ValueError(f"unknown ema label '{label}'. available: {self.labels}")
        if _has_fsdp(model):
            logger.warning(
                "EMA swap is disabled for FSDP-wrapped models; using online weights for this evaluation."
            )
            yield
            return

        with _fsdp_full_param_context(model):
            # save current weights (stay on device to avoid slow D2H copy)
            stored = {normalize_param_name(n): p.data.clone() for n, p in model.named_parameters()}
            shadow = self.shadows[label]
            for name, param in model.named_parameters():
                param.data.copy_(shadow[normalize_param_name(name)].to(param.device))
            try:
                yield
            finally:
                for name, param in model.named_parameters():
                    param.data.copy_(stored[normalize_param_name(name)])

    def to(self, device=None, dtype=None) -> "EMAModel":
        """move all shadow parameters to the given device and/or dtype."""
        for shadow in self.shadows.values():
            for name, param in shadow.items():
                shadow[name] = param.to(device=device, dtype=dtype)
        return self

    def __repr__(self) -> str:
        return f"EMAModel(labels={self.labels}, step_count={self.step_count}, schedule={self.schedule_fn.__name__})"

    # -- serialization ---------------------------------------------------------

    def state_dict(self, label: str | None = None) -> dict:
        """full checkpoint dict, or shadow params for a single label."""
        if label is not None:
            return dict(self.shadows[label])
        return {
            "step_count": self.step_count,
            "schedule": self.schedule_fn.__name__,
            "batch_size": self.batch_size,
            "schedules": self.schedules,
            "shadows": {label: dict(s) for label, s in self.shadows.items()},
        }

    def load_state_dict(self, state_dict: dict, label: str | None = None) -> None:
        """
        load shadow parameters. two formats:
          1. full:        {"shadows": {label: {name: tensor}}, "step_count": ...}
          2. single-copy: {name: tensor}  -> loaded into specified/default label
        """
        if not isinstance(state_dict, dict):
            raise ValueError(f"state_dict must be a dict, got {type(state_dict)}")

        # full format
        if "shadows" in state_dict:
            self.step_count = state_dict.get("step_count", 0)
            available_labels = list(state_dict["shadows"].keys())
            for lbl in self.labels:
                if lbl in state_dict["shadows"]:
                    _copy_params(self.shadows[lbl], state_dict["shadows"][lbl])
                elif available_labels:
                    fallback = available_labels[0]
                    logger.warning(
                        f"ema label '{lbl}' not found in checkpoint, "
                        f"copying from '{fallback}' instead"
                    )
                    _copy_params(self.shadows[lbl], state_dict["shadows"][fallback])
                else:
                    logger.warning(f"ema label '{lbl}' not found and no fallback available in checkpoint")
            return

        # single-copy format: {name: tensor}
        target = label or self.default_label
        _copy_params(self.shadows[target], state_dict)


def _copy_params(shadow: dict[str, torch.Tensor], params: dict) -> None:
    """copy params into shadow dict, matching by normalized name with suffix fallback."""
    normed = {normalize_param_name(k): v for k, v in params.items()}

    # suffix index: fallback when exact match fails (e.g. "backbone.X" vs "X")
    suffix_map: dict[str, torch.Tensor] = {}
    for k, v in normed.items():
        suffix = k.split(".", 1)[-1]
        suffix_map[suffix] = v

    matched, skipped = 0, []
    for name in shadow:
        src = normed.get(name)
        if src is None:
            suffix = name.split(".", 1)[-1]
            src = suffix_map.get(suffix) or suffix_map.get(name)
        if src is not None:
            shadow[name].data.copy_(src.to(shadow[name].device))
            matched += 1
        else:
            skipped.append(name)

    if skipped:
        logger.warning(f"ema load: {matched} matched, {len(skipped)} skipped (e.g. {skipped[0]})")
