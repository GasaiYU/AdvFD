"""Hand-written JVP for the AMFD amortizer MLP.

Vendored unmodified from the official AMFD release
(https://github.com/poppuppy/amfd, ``models/jvp_manual.py``).
MIT licensed; see ``licenses/LICENSE.amfd``.

Used when ``--amfd_jvp_impl manual`` is set; the default ``torch_func`` path
goes through ``torch.func.jvp`` instead.
"""

import torch
import torch.nn.functional as F


def _time_vector(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    t = t.to(device=x.device, dtype=x.dtype)
    if t.ndim == 0:
        return t.expand(x.shape[0])
    if t.ndim == 1:
        if t.shape[0] != x.shape[0]:
            raise ValueError(f"Expected t shape [B], got {tuple(t.shape)}")
        return t
    if t.shape[0] != x.shape[0]:
        raise ValueError(f"Expected t batch {x.shape[0]}, got {tuple(t.shape)}")
    if any(size != 1 for size in t.shape[1:]):
        raise ValueError(f"Expected t [B] or broadcastable [B, 1...], got {tuple(t.shape)}")
    return t.reshape(x.shape[0])


def _linear_jvp(linear, x: torch.Tensor, dx: torch.Tensor):
    y = F.linear(x, linear.weight, linear.bias)
    dy = F.linear(dx, linear.weight, None)
    return y, dy


def _rms_norm_jvp(norm, x: torch.Tensor, dx: torch.Tensor):
    eps = float(norm.eps)
    reduce_dims = tuple(range(x.ndim - len(norm.normalized_shape), x.ndim))
    x_float = x.float()
    dx_float = dx.float()
    mean_square = x_float.square().mean(dim=reduce_dims, keepdim=True)
    rstd = torch.rsqrt(mean_square + eps)
    drstd = -rstd.pow(3) * (x_float * dx_float).mean(dim=reduce_dims, keepdim=True)
    y = (x_float * rstd).to(dtype=x.dtype)
    dy = (dx_float * rstd + x_float * drstd).to(dtype=dx.dtype)
    weight = getattr(norm, "weight", None)
    if weight is not None:
        view_shape = [1] * (x.ndim - weight.ndim) + list(weight.shape)
        weight = weight.view(*view_shape)
        y = y * weight
        dy = dy * weight
    return y, dy


def _silu_jvp(x: torch.Tensor, dx: torch.Tensor):
    sig = torch.sigmoid(x)
    y = F.silu(x)
    dy = dx * sig * (1.0 + x * (1.0 - sig))
    return y, dy


def _res_block_jvp(
    block,
    x: torch.Tensor,
    dx: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    gate: torch.Tensor,
):
    h_norm, dh_norm = _rms_norm_jvp(block.norm, x, dx)
    h = h_norm * (1.0 + scale) + shift
    dh = dh_norm * (1.0 + scale)
    h12, dh12 = _linear_jvp(block.w1, h, dh)
    h1, h2 = h12.chunk(2, dim=-1)
    dh1, dh2 = dh12.chunk(2, dim=-1)
    silu_h1, dsilu_h1 = _silu_jvp(h1, dh1)
    hidden = silu_h1 * h2
    dhidden = dsilu_h1 * h2 + silu_h1 * dh2
    out, dout = _linear_jvp(block.w2, hidden, dhidden)
    x = x + out * gate
    dx = dx + dout * gate
    return x, dx


def manual_mlp_jvp(
    net,
    v: torch.Tensor,
    t: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    compute_dtype = net.input_proj.weight.dtype
    compute_device = net.input_proj.weight.device
    v = v.to(device=compute_device, dtype=compute_dtype)
    s0 = torch.zeros_like(v)
    t = _time_vector(t, v)
    labels = labels.to(device=compute_device, dtype=torch.long)

    x, dx = _linear_jvp(net.input_proj, s0, v)
    y = F.silu(net.time_embed(t) + net.cond_embed(labels))

    scale, shift, gate = net.ada_ln_blocks[0](y).chunk(3, dim=-1)
    for i, block in enumerate(net.res_blocks):
        if i > 0 and i % net.ada_ln_switch_freq == 0:
            ada_ln_block = net.ada_ln_blocks[i // net.ada_ln_switch_freq]
            scale, shift, gate = ada_ln_block(y).chunk(3, dim=-1)
        x, dx = _res_block_jvp(block, x, dx, scale, shift, gate)

    scale, shift = net.final_layer.ada_ln_modulation(y).chunk(2, dim=-1)
    x_norm, dx_norm = _rms_norm_jvp(net.final_layer.norm_final, x, dx)
    x = x_norm * (1.0 + scale) + shift
    dx = dx_norm * (1.0 + scale)
    _, dx = _linear_jvp(net.final_layer.linear, x, dx)
    return dx


__all__ = ["manual_mlp_jvp"]
