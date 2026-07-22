import math

import torch
from torch import inf


def _distributed_sum_(tensor: torch.Tensor) -> torch.Tensor:
    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
    ):
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return tensor


def _batch_gradient_normpreserve(
    grad: torch.Tensor,
    alpha: float,
    beta: float,
    max_scale: float,
    eps: float,
) -> torch.Tensor:
    """Reweight gradients and restore their norm up to ``max_scale``."""
    if grad.ndim == 0:
        raise ValueError("NormPreserve requires a gradient with a batch dimension")
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError(f"alpha must be finite and positive, got {alpha}")
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError(f"beta must be finite and positive, got {beta}")
    if not math.isfinite(max_scale) or max_scale < 1.0:
        raise ValueError(f"max_scale must be finite and >= 1, got {max_scale}")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"eps must be finite and positive, got {eps}")

    if alpha == 1.0 and beta == 1.0:
        return grad

    original_dtype = grad.dtype
    compute_dtype = torch.float64 if grad.dtype == torch.float64 else torch.float32
    grad_float = grad.to(dtype=compute_dtype)

    common_sum = grad_float.sum(dim=0, keepdim=True)
    batch_count = grad_float.new_tensor(float(grad.shape[0]))
    _distributed_sum_(common_sum)
    _distributed_sum_(batch_count)
    common = common_sum / batch_count.clamp_min(1.0)

    transformed = beta * grad_float + (alpha - beta) * common

    norm_squares = torch.stack(
        (grad_float.square().sum(), transformed.square().sum())
    )
    _distributed_sum_(norm_squares)
    grad_norm, transformed_norm = norm_squares.clamp_min(0.0).sqrt().unbind()
    norm_ratio = grad_norm / transformed_norm.clamp_min(eps)
    scale = torch.clamp(norm_ratio, max=max_scale)
    return (transformed * scale).to(dtype=original_dtype)


class _BatchGradientNormPreserve(torch.autograd.Function):
    """Identity in forward; apply batch-gradient NormPreserve in backward."""

    @staticmethod
    def forward(ctx, tensor, alpha, beta, max_scale, eps):
        ctx.alpha = float(alpha)
        ctx.beta = float(beta)
        ctx.max_scale = float(max_scale)
        ctx.eps = float(eps)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = _batch_gradient_normpreserve(
            grad_output,
            alpha=ctx.alpha,
            beta=ctx.beta,
            max_scale=ctx.max_scale,
            eps=ctx.eps,
        )
        return grad_input, None, None, None, None


def apply_batch_gradient_normpreserve(
    tensor: torch.Tensor,
    alpha: float,
    beta: float,
    max_scale: float = 1.5,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Reweight common/residual parts with bounded input-norm restoration.

    The forward pass is unchanged. Given an incoming per-sample gradient ``G``,
    backward computes ``V = alpha * mean(G) + beta * (G - mean(G))`` and
    returns ``V * min(max_scale, ||G||_F / max(||V||_F, eps))``. Thus the
    transformed gradient is never amplified by more than ``max_scale``.

    Distributed statistics cover the current backward microbatch across all
    ranks, but do not span gradient-accumulation microbatches. Every rank must
    execute this operation the same number of times and in the same order. Its
    explicit collectives still communicate inside a DDP ``no_sync()`` context.
    """
    return _BatchGradientNormPreserve.apply(tensor, alpha, beta, max_scale, eps)


def get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    if not parameters:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    grads = [p.grad.detach() for p in parameters]
    if float(norm_type) == inf:
        return max(g.abs().max().to(device) for g in grads)
    return torch.norm(
        torch.stack([torch.norm(g, norm_type).to(device) for g in grads]),
        norm_type,
    )
