"""AMFD (Amortized Frechet Distance) support for the static FD branch.

``amfd_loss`` and ``jvp_manual`` are vendored verbatim from the official AMFD
release; ``integration`` holds everything specific to this repository.
"""

from amfd.amfd_loss import AmortizedFDLoss
from amfd.integration import (
    AMFD_CHECKPOINT_KEYS,
    add_amfd_args,
    amfd_enabled,
    amfd_generator_loss,
    build_amfd_amortizers,
    load_amfd_state,
    log_amfd_config,
    resolve_amfd_args,
    resolve_amfd_judges,
    save_amfd_state,
    update_amortizers,
)

__all__ = [
    "AMFD_CHECKPOINT_KEYS",
    "AmortizedFDLoss",
    "add_amfd_args",
    "amfd_enabled",
    "amfd_generator_loss",
    "build_amfd_amortizers",
    "load_amfd_state",
    "log_amfd_config",
    "resolve_amfd_args",
    "resolve_amfd_judges",
    "save_amfd_state",
    "update_amortizers",
]
