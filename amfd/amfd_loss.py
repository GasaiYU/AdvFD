"""Amortized Frechet Distance (AMFD) loss.

Vendored unmodified from the official AMFD release
(https://github.com/poppuppy/amfd, ``models/amfd_loss.py``), which accompanies
"Amortized Moment Matching for Visual Generation" (arXiv:2607.26860).
MIT licensed; see ``licenses/LICENSE.amfd``.

Kept byte-identical to upstream so it can be re-synced by copy.  All wiring
into this repository lives in ``amfd/integration.py``.
"""

import math
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.func import jvp
from torch.utils.checkpoint import checkpoint

from .jvp_manual import manual_mlp_jvp


def timestep_embedding(t, dim, max_period=10000, time_factor: float = 1.0):
    half = dim // 2
    out_dtype = t.dtype
    t = t.float() * float(time_factor)
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
        / half
    )
    args = t[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding.to(dtype=out_dtype)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        t_freq = timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.RMSNorm(channels, eps=1e-6, elementwise_affine=True)
        hidden_dim = int(channels * 1.5)
        self.w1 = nn.Linear(channels, hidden_dim * 2, bias=True)
        self.w2 = nn.Linear(hidden_dim, channels, bias=True)

    def forward(self, x, scale, shift, gate):
        h = self.norm(x) * (1.0 + scale) + shift
        h1, h2 = self.w1(h).chunk(2, dim=-1)
        h = self.w2(F.silu(h1) * h2)
        return x + h * gate


class FinalLayer(nn.Module):
    def __init__(self, channels, out_channels):
        super().__init__()
        self.norm_final = nn.RMSNorm(channels, eps=1e-6, elementwise_affine=False)
        self.ada_ln_modulation = nn.Linear(channels, channels * 2, bias=True)
        self.linear = nn.Linear(channels, out_channels, bias=True)

    def forward(self, x, y):
        scale, shift = self.ada_ln_modulation(y).chunk(2, dim=-1)
        x = self.norm_final(x) * (1.0 + scale) + shift
        return self.linear(x)


class MlpEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        num_res_blocks: int,
        out_channels: Optional[int] = None,
        num_classes: int = 1000,
        num_adaln_blocks: int = 2,
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing
        self.num_classes = num_classes

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Embedding(num_classes, model_channels)
        self.input_proj = nn.Linear(in_channels, model_channels)
        self.res_blocks = nn.ModuleList(
            [ResBlock(model_channels) for _ in range(num_res_blocks)]
        )

        if num_adaln_blocks <= 0:
            raise ValueError("num_adaln_blocks must be positive")
        if num_res_blocks % num_adaln_blocks != 0:
            raise ValueError("num_res_blocks must be divisible by num_adaln_blocks")
        self.ada_ln_blocks = nn.ModuleList(
            [
                nn.Linear(model_channels, model_channels * 3, bias=True)
                for _ in range(num_adaln_blocks)
            ]
        )
        self.ada_ln_switch_freq = num_res_blocks // num_adaln_blocks
        self.final_layer = FinalLayer(model_channels, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(basic_init)
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)
        nn.init.normal_(self.cond_embed.weight, std=0.02)
        for block in self.ada_ln_blocks:
            nn.init.constant_(block.weight, 0)
            nn.init.constant_(block.bias, 0)
        nn.init.constant_(self.final_layer.ada_ln_modulation.weight, 0)
        nn.init.constant_(self.final_layer.ada_ln_modulation.bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        y = F.silu(self.time_embed(t) + self.cond_embed(c.long()))

        scale, shift, gate = self.ada_ln_blocks[0](y).chunk(3, dim=-1)
        for i, block in enumerate(self.res_blocks):
            if i > 0 and i % self.ada_ln_switch_freq == 0:
                ada_ln_block = self.ada_ln_blocks[i // self.ada_ln_switch_freq]
                scale, shift, gate = ada_ln_block(y).chunk(3, dim=-1)
            if self.grad_checkpointing and self.training:
                x = checkpoint(block, x, scale, shift, gate, use_reentrant=False)
            else:
                x = block(x, scale, shift, gate)
        return self.final_layer(x, y)

def _feature_weight(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    t = t.to(device=x.device, dtype=x.dtype)
    while t.ndim < x.ndim:
        t = t.unsqueeze(-1)
    return t


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


def A_apply(
    g: Callable,
    v: torch.Tensor,
    t: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    def fn(s):
        return g(s, t, labels)

    _, av = jvp(fn, (torch.zeros_like(v),), (v,))
    return av


class AmortizedFDLoss(nn.Module):
    """
    FM+JVP moment loss.

    The main class owns the real/fake MLPs directly. With separate MLPs, real
    and fake both use cov times t in (0, 1) and mean time t=2. With a shared
    MLP, fake cov is encoded as -t and fake mean as t=-2.
    """

    def __init__(
        self,
        feat_dim: int,
        model_channels: int = 1024,
        depth: int = 8,
        num_classes: int = 1000,
        num_adaln_blocks: int = 2,
        feature_mean: Optional[torch.Tensor] = None,
        feature_std: Optional[torch.Tensor] = None,
        grad_checkpointing: bool = False,
        t: float = 0.25,
        diff_batch_mul: int = 1,
        train_real_branch: bool = True,
        jacobi_generator_loss: bool = False,
        share_real_fake_mlp: bool = False,
        prediction_target: str = "v",
        normalize_generator_loss: bool = False,
        generator_loss_norm_eps: float = 0.01,
        generator_loss_norm_power: float = 1.0,
        jvp_impl: str = "torch_func",
    ):
        super().__init__()

        self.feat_dim = int(feat_dim)
        self.diff_batch_mul = int(diff_batch_mul)
        self.t = float(t)
        self.train_real_branch = bool(train_real_branch)
        self.jacobi_generator_loss = bool(jacobi_generator_loss)
        self.share_real_fake_mlp = bool(share_real_fake_mlp)
        self.prediction_target = prediction_target
        self.normalize_generator_loss = bool(normalize_generator_loss)
        self.generator_loss_norm_eps = float(generator_loss_norm_eps)
        self.generator_loss_norm_power = float(generator_loss_norm_power)
        self.jvp_impl = jvp_impl

        if feature_mean is None or feature_std is None:
            raise ValueError("feature_mean and feature_std are required for AMFD feature normalization")
        if feature_mean.shape[-1] != feat_dim:
            raise ValueError(f"feature_mean dim {feature_mean.shape[-1]} != feat_dim {feat_dim}")
        if feature_std.shape[-1] != feat_dim:
            raise ValueError(f"feature_std dim {feature_std.shape[-1]} != feat_dim {feat_dim}")
        self.register_buffer("feature_mean", feature_mean.float(), persistent=True)
        self.register_buffer("feature_std", feature_std.float(), persistent=True)

        net_kwargs = dict(
            in_channels=feat_dim,
            model_channels=model_channels,
            num_res_blocks=depth,
            out_channels=feat_dim,
            num_classes=num_classes,
            num_adaln_blocks=num_adaln_blocks,
            grad_checkpointing=grad_checkpointing,
        )
        if self.share_real_fake_mlp:
            self.shared_net = MlpEncoder(**net_kwargs)
        else:
            self.real_net = MlpEncoder(**net_kwargs)
            self.fake_net = MlpEncoder(**net_kwargs)

    def maybe_normalize(self, h: torch.Tensor) -> torch.Tensor:
        mean = self.feature_mean.to(device=h.device, dtype=h.dtype).view(1, -1)
        std = self.feature_std.to(device=h.device, dtype=h.dtype).view(1, -1)
        return (h - mean) / std

    def mean(self, domain: str, labels: torch.Tensor, dtype=None) -> torch.Tensor:
        labels = labels.long()
        dtype = torch.float32 if dtype is None else dtype
        q0 = torch.zeros(labels.shape[0], self.feat_dim, device=labels.device, dtype=dtype)
        if domain == "real":
            net = self.shared_net if self.share_real_fake_mlp else self.real_net
            t_value = 2.0
        elif domain == "fake":
            net = self.shared_net if self.share_real_fake_mlp else self.fake_net
            t_value = -2.0 if self.share_real_fake_mlp else 2.0
        else:
            raise ValueError(f"Unknown AMFD domain {domain!r}; expected 'real' or 'fake'")
        t_mean = torch.full((labels.shape[0],), t_value, device=labels.device, dtype=dtype)
        return net(q0, _time_vector(t_mean, q0), labels)

    def A_apply(
        self,
        domain: str,
        v: torch.Tensor,
        t: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        labels = labels.long()
        if domain == "real":
            net = self.shared_net if self.share_real_fake_mlp else self.real_net
            net_t = t
        elif domain == "fake":
            net = self.shared_net if self.share_real_fake_mlp else self.fake_net
            net_t = -t if self.share_real_fake_mlp else t
        else:
            raise ValueError(f"Unknown AMFD domain {domain!r}; expected 'real' or 'fake'")

        def g(s, time, cls):
            return net(s, _time_vector(time, s), cls.long())

        if self.jvp_impl == "torch_func":
            out = A_apply(g, v, net_t, labels)
        elif self.jvp_impl == "manual":
            out = manual_mlp_jvp(net, v, net_t, labels)
        else:
            raise RuntimeError(f"Unexpected jvp_impl {self.jvp_impl!r}")
        return out

    def freeze_real_branch(self) -> bool:
        if self.share_real_fake_mlp:
            return False
        self.real_net.requires_grad_(False)
        return True

    def amort_loss(
        self,
        h_real: torch.Tensor,
        h_fake: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        h_real = self.maybe_normalize(h_real.detach())
        h_fake = self.maybe_normalize(h_fake.detach())
        labels = labels.long()
        if labels.shape[0] != h_real.shape[0]:
            raise ValueError(f"labels batch {labels.shape[0]} != h_real batch {h_real.shape[0]}")
        if h_fake.shape[0] != h_real.shape[0]:
            raise ValueError(f"h_fake batch {h_fake.shape[0]} != h_real batch {h_real.shape[0]}")

        if self.diff_batch_mul > 1:
            h_real = h_real.repeat_interleave(self.diff_batch_mul, dim=0)
            h_fake = h_fake.repeat_interleave(self.diff_batch_mul, dim=0)
            labels = labels.repeat_interleave(self.diff_batch_mul, dim=0)
        t = torch.full((h_real.shape[0], 1), self.t, device=h_real.device, dtype=h_real.dtype)

        batch = h_real.shape[0]
        x1 = torch.cat([h_real, h_fake], dim=0)
        x0 = torch.cat([torch.randn_like(h_real), torch.randn_like(h_fake)], dim=0)
        labels_cat = torch.cat([labels, labels], dim=0)
        t_base = t.expand(batch) if t.ndim == 0 else t
        t_state = torch.cat([t_base, t_base], dim=0)
        t_w = _feature_weight(t_state, x1)

        def build_zt(mu):
            if mu.shape != x1.shape:
                raise ValueError(f"mean returned {tuple(mu.shape)}, expected {tuple(x1.shape)}")
            xt = t_w * x1 + (1.0 - t_w) * x0
            return xt, xt - t_w * mu

        if self.share_real_fake_mlp:
            q0 = torch.zeros(
                labels_cat.shape[0],
                self.feat_dim,
                device=labels_cat.device,
                dtype=x1.dtype,
            )
            t_mean = torch.cat(
                [
                    torch.full((batch,), 2.0, device=labels.device, dtype=x1.dtype),
                    torch.full((batch,), -2.0, device=labels.device, dtype=x1.dtype),
                ],
                dim=0,
            )
            mu = self.shared_net(q0, _time_vector(t_mean, q0), labels_cat)
            xt, zt = build_zt(mu)
            raw_a = self.A_apply("real", zt, torch.cat([t_base, -t_base], dim=0), labels_cat)
        elif self.train_real_branch:
            mu = torch.cat(
                [
                    self.mean("real", labels, dtype=x1.dtype),
                    self.mean("fake", labels, dtype=x1.dtype),
                ],
                dim=0,
            )
            xt, zt = build_zt(mu)
            raw_a = torch.cat(
                [
                    self.A_apply("real", zt[:batch], t, labels),
                    self.A_apply("fake", zt[batch:], t, labels),
                ],
                dim=0,
            )
        else:
            with torch.no_grad():
                mu_real = self.mean("real", labels, dtype=x1.dtype)
            mu = torch.cat([mu_real, self.mean("fake", labels, dtype=x1.dtype)], dim=0)
            xt, zt = build_zt(mu)
            with torch.no_grad():
                raw_real = self.A_apply("real", zt[:batch], t, labels)
            raw_a = torch.cat([raw_real, self.A_apply("fake", zt[batch:], t, labels)], dim=0)

        if self.prediction_target == "v":
            v_target = x1 - x0
            v_pred = mu - zt + t_w * raw_a
        elif self.prediction_target == "x":
            denom = (1.0 - t_w).clamp_min(0.05)
            v_target = (x1 - xt) / denom
            v_pred = (mu + t_w * raw_a - xt) / denom
        else:
            raise ValueError(f"Unknown prediction_target {self.prediction_target!r}")

        real_loss_fm = F.mse_loss(v_pred[:batch], v_target[:batch])
        real_loss_mu = F.mse_loss(mu[:batch], x1[:batch])
        fake_loss_fm = F.mse_loss(v_pred[batch:], v_target[batch:])
        fake_loss_mu = F.mse_loss(mu[batch:], x1[batch:])
        real_loss = real_loss_fm + real_loss_mu
        fake_loss = fake_loss_fm + fake_loss_mu
        loss = (real_loss if self.train_real_branch else real_loss.detach() * 0.0) + fake_loss

        logs = {
            "amfd/amort_loss": loss.detach(),
            "amfd/real_fm_loss": real_loss_fm.detach(),
            "amfd/real_mu_loss": real_loss_mu.detach(),
            "amfd/fake_fm_loss": fake_loss_fm.detach(),
            "amfd/fake_mu_loss": fake_loss_mu.detach(),
        }
        return loss, logs

    def generator_loss(
        self,
        h_fake: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        h = self.maybe_normalize(h_fake)
        labels = labels.long()
        if labels.shape[0] != h.shape[0]:
            raise ValueError(f"labels batch {labels.shape[0]} != h_fake batch {h.shape[0]}")

        if self.diff_batch_mul > 1:
            h = h.repeat_interleave(self.diff_batch_mul, dim=0)
            labels = labels.repeat_interleave(self.diff_batch_mul, dim=0)
        t = torch.full((h.shape[0], 1), self.t, device=h.device, dtype=h.dtype)
        t_w = _feature_weight(t, h)

        mu_r = self.mean("real", labels, dtype=h.dtype).detach()
        mu_g = self.mean("fake", labels, dtype=h.dtype).detach()
        u = h - mu_g
        dmu = mu_g - mu_r

        def velocity_A_apply(domain, v):
            raw = self.A_apply(domain, v, t, labels)
            if self.prediction_target == "v":
                return raw
            if self.prediction_target == "x":
                return (raw - v) / (1.0 - _feature_weight(t, v)).clamp_min(0.05)
            raise ValueError(f"Unknown prediction_target {self.prediction_target!r}")

        if self.jacobi_generator_loss:
            ag_u = velocity_A_apply("fake", u)
            pg_u = (1.0 + t_w) * u - t_w.square() * ag_u
            ag_pg_u = velocity_A_apply("fake", pg_u)
            ar_pg_u = velocity_A_apply("real", pg_u)
            delta_inner = ag_pg_u - ar_pg_u
            ag_delta_inner = velocity_A_apply("fake", delta_inner)
            delta_au = (1.0 + t_w) * delta_inner - t_w.square() * ag_delta_inner
            # Near Sigma=I, P_g(A_g-A_r)P_g u contributes
            # (1-t)^3 / (t^2+(1-t)^2)^4 times the covariance gradient.
            # This scale matches the FD covariance gradient coefficient under
            # that approximation.
            cov_weight = 0.5 * (t_w.square() + (1.0 - t_w).square()).pow(4) / (1.0 - t_w).pow(3).clamp_min(1e-8)
        else:
            delta_au = velocity_A_apply("fake", u) - velocity_A_apply("real", u)
            # Near Sigma=I, A_g-A_r contributes
            # (1-t) / (t^2 + (1-t)^2)^2 times DeltaSigma. Match the
            # covariance force to the same FD mean/covariance calibration as
            # the Jacobi branch while omitting only the Jacobi factors.
            bridge_var = t_w.square() + (1.0 - t_w).square()
            cov_weight = (
                0.5
                * bridge_var.square()
                / (1.0 - t_w).clamp_min(1e-8)
            )

        loss_mu_terms = dmu * h
        loss_cov_terms = cov_weight * u * delta_au
        loss_terms = loss_mu_terms + loss_cov_terms
        with torch.no_grad():
            gen_grad_proxy = dmu + (2.0 * cov_weight) * delta_au

        loss_mu = loss_mu_terms.sum(dim=-1).mean() / float(self.feat_dim)
        loss_cov = loss_cov_terms.sum(dim=-1).mean() / float(self.feat_dim)
        loss_raw = loss_mu + loss_cov

        logs = {
            "amfd/gen_loss": loss_raw.detach(),
            "amfd/gen_mu_loss": loss_mu.detach(),
            "amfd/gen_cov_loss": loss_cov.detach(),
            "amfd/delta_mu_norm": dmu.norm(dim=-1).mean().detach(),
            "amfd/delta_au_norm": delta_au.norm(dim=-1).mean().detach(),
        }

        if self.normalize_generator_loss:
            reduce_dims = tuple(range(1, loss_terms.ndim))
            norm_denom = (
                gen_grad_proxy.float()
                .square()
                .sum(dim=reduce_dims, keepdim=True)
                .pow(self.generator_loss_norm_power)
                .add(self.generator_loss_norm_eps)
            )
            loss = (loss_terms.float() / norm_denom).sum(dim=reduce_dims).mean()
            logs["amfd/gen_loss"] = loss.detach()
            logs["amfd/gen_loss_norm_denom"] = norm_denom.detach().mean()
        else:
            loss = loss_raw

        return loss, logs

class JVPMomentBranch(nn.Module):
    """Compatibility wrapper for older diagnostic scripts.

    Main AMFD training should use AmortizedFDLoss directly.
    """

    def __init__(
        self,
        feat_dim: int,
        model_channels: int = 1024,
        depth: int = 8,
        num_classes: int = 1000,
        num_adaln_blocks: int = 2,
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.net = MlpEncoder(
            in_channels=feat_dim,
            model_channels=model_channels,
            num_res_blocks=depth,
            out_channels=feat_dim,
            num_classes=num_classes,
            num_adaln_blocks=num_adaln_blocks,
            grad_checkpointing=grad_checkpointing,
        )

    def mean(self, labels: torch.Tensor, dtype=None) -> torch.Tensor:
        labels = labels.long()
        dtype = torch.float32 if dtype is None else dtype
        q0 = torch.zeros(labels.shape[0], self.feat_dim, device=labels.device, dtype=dtype)
        t_mean = torch.full((labels.shape[0],), 2.0, device=labels.device, dtype=dtype)
        return self.net(q0, t_mean, labels)

    def g(self, s: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.net(s, _time_vector(t, s), labels.long())

    def A_apply(self, v: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return A_apply(self.g, v, t, labels.long())

    def forward_estimator(
        self,
        x1: torch.Tensor,
        labels: torch.Tensor,
        t: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
        **_ignored,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        labels = labels.long()
        t_w = _feature_weight(t, x1)
        mu = self.mean(labels, dtype=x1.dtype)
        x0 = torch.randn_like(x1) if x0 is None else x0.to(device=x1.device, dtype=x1.dtype)
        xt = t_w * x1 + (1.0 - t_w) * x0
        v_target = x1 - x0
        zt = xt - t_w * mu
        v_pred = mu - zt + t_w * self.A_apply(zt, t, labels)
        loss_fm = F.mse_loss(v_pred, v_target)
        loss_mu = F.mse_loss(mu, x1)
        loss = loss_fm + loss_mu
        return loss, {
            "loss": loss.detach(),
            "loss_fm": loss_fm.detach(),
            "loss_mu": loss_mu.detach(),
        }


def set_requires_grad(module: nn.Module, flag: bool):
    for param in module.parameters():
        param.requires_grad_(flag)


MomentBranch = JVPMomentBranch


__all__ = [
    "A_apply",
    "AmortizedFDLoss",
    "JVPMomentBranch",
    "MomentBranch",
    "set_requires_grad",
]
