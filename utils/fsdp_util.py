import logging
from functools import partial

import torch
import torch.nn as nn

logger = logging.getLogger("FD_loss")


_SHARDING_STRATEGIES = {
    "full_shard": "FULL_SHARD",
    "shard_grad_op": "SHARD_GRAD_OP",
    "no_shard": "NO_SHARD",
}


def apply_fsdp(args, model: nn.Module) -> nn.Module:
    """Wrap the trainable backbone in FSDP while keeping denoiser helper methods intact."""
    if not getattr(args, "fsdp", False):
        return model

    from torch.distributed.fsdp import BackwardPrefetch, FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy

    if not hasattr(model, "net"):
        raise ValueError(f"FSDP expects the generation model to expose a 'net' submodule, got {type(model)}")

    strategy_name = _SHARDING_STRATEGIES[args.fsdp_sharding_strategy]
    sharding_strategy = getattr(ShardingStrategy, strategy_name)
    auto_wrap_policy = None
    if args.fsdp_wrap_granularity == "transformer":
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        from models.jit import JiTBlock
        from models.mit import TransformerBlock

        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={JiTBlock, TransformerBlock},
        )

    model.net = FSDP(
        model.net,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=sharding_strategy,
        use_orig_params=True,
        device_id=torch.cuda.current_device(),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        forward_prefetch=args.fsdp_forward_prefetch,
        limit_all_gathers=args.fsdp_limit_all_gathers,
    )
    logger.info(
        "[FSDP] Wrapped model.net with strategy=%s, wrap_granularity=%s, use_orig_params=True",
        args.fsdp_sharding_strategy,
        args.fsdp_wrap_granularity,
    )
    return model


def is_fsdp_enabled(args) -> bool:
    return bool(getattr(args, "fsdp", False))
