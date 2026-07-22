#!/usr/bin/env python3
"""Train a direct x-prediction JiT-B continuous-time consistency model.

The supplied JiT-B checkpoint already predicts a clean image.  This entry point
keeps that semantic meaning throughout training: the raw network output is

    D_theta(x_t / sigma_data, t) ~= x_0.

Training has two phases.  First, the legacy linear-flow checkpoint is adapted
to TrigFlow with the velocity-equivalent x-prediction diffusion objective

    ||D_theta - x_0||^2 / sin(t)^2.

At the phase boundary, the adaptation EMA is copied back into the student;
Adam and EMA are then reset, and direct x-prediction sCM training starts with
a fresh tangent warmup.  The sCM target is the exact change of variables of
the paper's F-space Algorithm 1:

    j = cos(t) sin(t) dD^-/dt
    k = (1-r) cos(t)^2 (D^- - x_0) + r j
    q = sigma_data sin(t) k / (||k|| + c sin(t))
    R = D_theta - D^- + q.

The explicit factor 1 / (sigma_data^2 sin(t)^2) is retained in the loss.  This
keeps the adaptive-weight network in the original F-space convention and makes
the scalar objective (up to the configured numerical sin floor) identical to
the F-parameterized objective for every t > 0.  At inference time the network
output is returned directly; it is never reinterpreted as TrigFlow F.

The physical TrigFlow angle remains ``t`` everywhere in the path, loss, and
sampler.  To preserve the released JiT checkpoint's legacy time semantics,
every backbone call uses the permanent bounded conditioning map

    t_JiT = 1 - 2 * t / pi.

The JVP differentiates the complete wrapper, so its time derivative includes
the exact chain-rule factor -2/pi automatically.

For the high-noise control experiments, the JVP input direction can use the
sampled sCT conditional velocity, a frozen adaptation teacher's PF velocity

    v_T(x_t, t) = (cos(t) x_t - D_T(x_t, t)) / sin(t),

or a smooth hybrid that changes from sCT to the teacher direction over a
configurable physical-time interval.  Only this input direction changes; the
student parameterization, tangent warmup, normalization, and scalar loss stay
identical across the three branches.

Unlike the F parameterization, a raw x-prediction network does not satisfy the
consistency boundary by construction.  A positive boundary-band loss is
therefore mandatory during both phases.  It supervises a narrow interval near
t=0 rather than a single point, preventing the network from hiding a sharp
transition immediately after zero.  Without boundary information, every
input-independent image is a zero-tangent solution; JiT can realize that
degeneracy as one fixed 16x16 decoder patch repeated over the image.
"""

import contextlib
import copy
import math
import sys
from pathlib import Path

import typing_extensions

# Keep this entry point usable with the older typing_extensions installed in
# the project image (PyTorch 2.4 imports this metadata-only decorator).
if not hasattr(typing_extensions, "deprecated"):
    def _deprecated(*args, **kwargs):
        del args, kwargs

        def decorator(obj):
            return obj

        return decorator

    typing_extensions.deprecated = _deprecated

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from meanflow.meanflow_jit import (  # noqa: E402
    MeanFlowJiTDenoiser,
    _jvp_sdpa_context,
    _no_autocast_context,
)
from meanflow import train_meanflow_jit_b as trainer  # noqa: E402
from utils.runtime_util import normalize_param_name  # noqa: E402


class SCMJiTDenoiser(MeanFlowJiTDenoiser):
    """TrigFlow sCM whose network output is always the clean prediction D."""

    def __init__(
        self,
        *args,
        x_adapt_steps=0,
        x_loss_sin_min=1e-3,
        boundary_loss_weight=1.0,
        boundary_band_max=0.02,
        deterministic_boundary=True,
        collapse_monitor_samples=8,
        network_time_mode="legacy_reversed",
        tangent_velocity_mode="sct",
        hybrid_teacher_start=1.50,
        hybrid_teacher_end=1.53,
        teacher_velocity_sin_min=1e-3,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.objective != "scm":
            raise ValueError("SCMJiTDenoiser only supports objective='scm'")
        self.x_adapt_steps = int(x_adapt_steps)
        self.x_loss_sin_min = float(x_loss_sin_min)
        self.boundary_loss_weight = float(boundary_loss_weight)
        self.boundary_band_max = float(boundary_band_max)
        self.deterministic_boundary = bool(deterministic_boundary)
        self.collapse_monitor_samples = int(collapse_monitor_samples)
        self.network_time_mode = str(network_time_mode)
        self.tangent_velocity_mode = str(tangent_velocity_mode)
        self.hybrid_teacher_start = float(hybrid_teacher_start)
        self.hybrid_teacher_end = float(hybrid_teacher_end)
        self.teacher_velocity_sin_min = float(teacher_velocity_sin_min)
        # The reference JiT is intentionally not a registered submodule: it is
        # immutable, excluded from Adam/EMA/checkpoints/DDP broadcasts, and is
        # initialized from the selected adaptation checkpoint only after that
        # checkpoint has been loaded by the trainer.
        object.__setattr__(self, "_reference_net", None)
        if self.x_adapt_steps < 0:
            raise ValueError("x_adapt_steps must be non-negative")
        if self.x_loss_sin_min <= 0.0:
            raise ValueError("x_loss_sin_min must be positive")
        if self.boundary_loss_weight < 0.0:
            raise ValueError("boundary_loss_weight must be non-negative")
        if not 0.0 < self.boundary_band_max < 0.5 * math.pi:
            raise ValueError(
                "boundary_band_max must be in (0, pi/2)"
            )
        if self.collapse_monitor_samples <= 0:
            raise ValueError("collapse_monitor_samples must be positive")
        if self.network_time_mode not in {"legacy_reversed", "literal"}:
            raise ValueError(
                "network_time_mode must be 'legacy_reversed' or 'literal', "
                f"got {self.network_time_mode!r}"
            )
        if self.tangent_velocity_mode not in {"sct", "hybrid", "scd"}:
            raise ValueError(
                "tangent_velocity_mode must be 'sct', 'hybrid', or 'scd', "
                f"got {self.tangent_velocity_mode!r}"
            )
        if not 0.0 < self.teacher_velocity_sin_min <= 1.0:
            raise ValueError("teacher_velocity_sin_min must be in (0, 1]")
        if not (
            0.0
            < self.hybrid_teacher_start
            < self.hybrid_teacher_end
            < 0.5 * math.pi
        ):
            raise ValueError(
                "hybrid teacher interval must satisfy 0 < start < end < pi/2"
            )

    def initialize_reference_teacher(self):
        """Freeze the loaded adaptation network for sCD velocity queries."""
        if self.tangent_velocity_mode == "sct":
            return
        reference_net = copy.deepcopy(self.net)
        reference_net.requires_grad_(False)
        reference_net.eval()
        if hasattr(reference_net, "grad_checkpointing"):
            reference_net.grad_checkpointing = False
        object.__setattr__(self, "_reference_net", reference_net)
        trainer.logger.info(
            "initialized frozen reference JiT from loaded %s weights; "
            "tangent velocity mode=%s",
            self.network_time_mode,
            self.tangent_velocity_mode,
        )

    def train(self, mode=True):
        result = super().train(mode)
        reference_net = getattr(self, "_reference_net", None)
        if reference_net is not None:
            reference_net.eval()
        return result

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        reference_net = getattr(self, "_reference_net", None)
        if reference_net is not None:
            reference_net._apply(fn, recurse=recurse)
        return result

    @staticmethod
    def _reshape_time(t, x):
        return t.view(-1, *([1] * (x.ndim - 1)))

    def _sample_trigflow_time(self, x):
        tau = torch.randn(x.size(0), device=x.device) * self.P_std + self.P_mean

        # Stable evaluation of atan(exp(tau) / sigma_data): exp() only sees
        # non-positive arguments, so log-normal tails cannot overflow.
        log_sigma_data = tau.new_tensor(self.sigma_data).log()
        log_ratio = tau - log_sigma_data
        acute_t = torch.atan(torch.exp(-log_ratio.abs()))
        return torch.where(
            log_ratio >= 0.0,
            0.5 * torch.pi - acute_t,
            acute_t,
        )

    def _sin_for_loss(self, t):
        return torch.sin(t).float().clamp_min(self.x_loss_sin_min)

    def _network_time(self, physical_t):
        """Map physical TrigFlow time to the JiT conditioning coordinate."""
        if self.network_time_mode == "literal":
            return physical_t
        return 1.0 - (2.0 / math.pi) * physical_t

    def _net_physical_time(self, x_in, physical_t, labels):
        """Evaluate JiT as a function of physical TrigFlow time."""
        return self.net(
            x_in,
            self._network_time(physical_t).flatten(),
            labels,
        )

    def _teacher_velocity_mix(self, t):
        """Return the smooth per-sample sCD fraction in [0, 1]."""
        if self.tangent_velocity_mode == "sct":
            return torch.zeros_like(t, dtype=torch.float32)
        if self.tangent_velocity_mode == "scd":
            return torch.ones_like(t, dtype=torch.float32)

        unit = (
            (t.detach().float() - self.hybrid_teacher_start)
            / (self.hybrid_teacher_end - self.hybrid_teacher_start)
        ).clamp(0.0, 1.0)
        # C1-continuous smoothstep avoids a hard change in the JVP direction.
        return unit.square() * (3.0 - 2.0 * unit)

    @torch.no_grad()
    def _select_tangent_velocity(self, x_t, t, labels, conditional_velocity):
        """Blend conditional sCT velocity with frozen-teacher PF velocity."""
        mix = self._teacher_velocity_mix(t)
        selected = mix > 0.0
        conditional_float = conditional_velocity.detach().float()
        teacher_velocity_rms = torch.zeros_like(mix)
        teacher_prediction_rms = torch.zeros_like(mix)
        teacher_velocity_delta_rms = torch.zeros_like(mix)
        sin_clamped = torch.zeros_like(mix, dtype=torch.bool)

        if not selected.any():
            return (
                conditional_float,
                mix,
                teacher_velocity_rms,
                teacher_prediction_rms,
                teacher_velocity_delta_rms,
                sin_clamped,
            )

        reference_net = getattr(self, "_reference_net", None)
        if reference_net is None:
            raise RuntimeError(
                "frozen reference JiT is not initialized; the trainer must "
                "call initialize_reference_teacher() after checkpoint loading"
            )

        selected_t = t[selected].detach().float()
        selected_x_t = x_t[selected].detach()
        with _jvp_sdpa_context(x_t):
            selected_prediction = reference_net(
                selected_x_t / self.sigma_data,
                self._network_time(selected_t).flatten(),
                labels[selected],
            )
        selected_prediction = selected_prediction.detach().float()
        selected_t_img = self._reshape_time(selected_t, selected_x_t)
        selected_sin = torch.sin(selected_t_img)
        selected_denom = selected_sin.clamp_min(
            self.teacher_velocity_sin_min
        )
        selected_velocity = (
            torch.cos(selected_t_img) * selected_x_t.float()
            - selected_prediction
        ) / selected_denom

        teacher_velocity_rms[selected] = (
            selected_velocity.flatten(1).square().mean(dim=1).sqrt()
        )
        teacher_prediction_rms[selected] = (
            selected_prediction.flatten(1).square().mean(dim=1).sqrt()
        )
        teacher_velocity_delta_rms[selected] = (
            (selected_velocity - conditional_float[selected])
            .flatten(1).square().mean(dim=1).sqrt()
        )
        sin_clamped[selected] = (
            torch.sin(selected_t) < self.teacher_velocity_sin_min
        )
        velocity = conditional_float.clone()
        selected_mix_img = self._reshape_time(mix[selected], selected_velocity)
        velocity[selected] = torch.lerp(
            conditional_float[selected],
            selected_velocity,
            selected_mix_img,
        )
        return (
            velocity,
            mix,
            teacher_velocity_rms,
            teacher_prediction_rms,
            teacher_velocity_delta_rms,
            sin_clamped,
        )

    @staticmethod
    def _target_metric_edge_label(value):
        if math.isclose(value, 0.5 * math.pi):
            return "pi2"
        return f"{value:.4f}".replace(".", "p")

    def _target_bin_metric_sums(self, t, values):
        """Emit count-weighted high-noise target diagnostics.

        The trainer converts ``GROUP__VALUE_sum / GROUP__count`` into an exact
        distributed mean, avoiding the empty-rank bias of ordinary bin means.
        """
        edges = (0.0, 1.45, 1.50, 1.53, 1.55, 1.56, 1.5645, 0.5 * math.pi)
        detached_t = t.detach().float()
        metrics = {}
        for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
            if index + 1 == len(edges) - 1:
                mask = (detached_t >= low) & (detached_t <= high)
            else:
                mask = (detached_t >= low) & (detached_t < high)
            mask_float = mask.float()
            group = (
                f"target_t_{self._target_metric_edge_label(low)}_"
                f"{self._target_metric_edge_label(high)}"
            )
            metrics[f"{group}__count"] = mask_float.sum()
            for name, per_sample in values.items():
                per_sample = per_sample.detach().float()
                if per_sample.shape != detached_t.shape:
                    raise RuntimeError(
                        f"target diagnostic {name!r} must have shape "
                        f"{tuple(detached_t.shape)}, got "
                        f"{tuple(per_sample.shape)}"
                    )
                metrics[f"{group}__{name}_sum"] = (
                    per_sample * mask_float
                ).sum()
        return metrics

    @contextlib.contextmanager
    def _boundary_dropout_context(self):
        """Disable only dropout for a deterministic boundary-band forward.

        JiT attention consults the owning attention module's ``training`` flag,
        whereas projection/MLP dropout consult the ``nn.Dropout`` modules, so
        both kinds of owners are covered here.  Gradient checkpointing is
        disabled for this one forward: otherwise its backward recomputation
        would happen after the dropout flags had been restored and would no
        longer represent the same function.
        """
        if not self.deterministic_boundary:
            yield
            return

        dropout_owners = []
        had_grad_checkpointing = hasattr(self.net, "grad_checkpointing")
        grad_checkpointing = getattr(self.net, "grad_checkpointing", False)
        if had_grad_checkpointing:
            self.net.grad_checkpointing = False
        for module in self.net.modules():
            owns_attention_dropout = isinstance(
                getattr(module, "attn_drop", None), torch.nn.Dropout
            )
            if isinstance(module, torch.nn.Dropout) or owns_attention_dropout:
                dropout_owners.append((module, module.training))
                module.training = False
        try:
            yield
        finally:
            for module, was_training in dropout_owners:
                module.training = was_training
            if had_grad_checkpointing:
                self.net.grad_checkpointing = grad_checkpointing

    def _boundary_loss(self, x, labels):
        if self.boundary_loss_weight == 0.0:
            zeros = x.new_zeros(x.size(0), dtype=torch.float32)
            return zeros.mean(), None, zeros, zeros

        # A narrow interval is deliberately used instead of only t=0.  This
        # rules out a solution that is correct at exactly zero but jumps to an
        # input-independent template for every sampled positive time.
        t_boundary = (
            torch.rand(x.size(0), device=x.device, dtype=torch.float32)
            * self.boundary_band_max
        )
        t_img = self._reshape_time(t_boundary, x)
        noise = torch.randn_like(x) * self.sigma_data
        x_boundary = torch.cos(t_img) * x + torch.sin(t_img) * noise
        with self._boundary_dropout_context():
            boundary_pred = self._net_physical_time(
                x_boundary / self.sigma_data,
                t_boundary,
                labels,
            )
        loss_per_sample = (
            (boundary_pred.float() - x.float())
            .square()
            .flatten(1)
            .mean(dim=1)
        )
        return loss_per_sample.mean(), boundary_pred, t_boundary, loss_per_sample

    @staticmethod
    def _time_bin_metrics(prefix, values, t):
        """Report endpoint MSE in four fixed TrigFlow-angle intervals."""
        metrics = {}
        normalized_t = t.detach().float() / (0.5 * torch.pi)
        values = values.detach().float()
        for index in range(4):
            low = index / 4.0
            high = (index + 1) / 4.0
            if index == 3:
                mask = (normalized_t >= low) & (normalized_t <= high)
            else:
                mask = (normalized_t >= low) & (normalized_t < high)
            count = mask.float().sum()
            mean = (values * mask.float()).sum() / count.clamp_min(1.0)
            metrics[f"{prefix}_tbin{index}_mse"] = mean
            metrics[f"{prefix}_tbin{index}_count"] = count
        return metrics

    def _collapse_metrics(self, prediction):
        """Cheap diagnostics for the repeated decoder-patch failure mode."""
        sample_count = min(prediction.size(0), self.collapse_monitor_samples)
        value = prediction[:sample_count].detach().float()
        centered = value - value.mean(dim=(1, 2, 3), keepdim=True)
        variance = centered.square().mean(dim=(1, 2, 3)).clamp_min(1e-12)

        patch_size = int(getattr(self.net, "patch_size", 16))
        if value.size(-2) < patch_size or value.size(-1) < patch_size:
            zero = value.new_zeros(())
            return {
                "patch_shift_corr_x16": zero,
                "patch_shift_corr_y16": zero,
                "patch_spatial_std": zero,
                "patch_template_explained_frac": zero,
                "sample_output_corr": zero,
                "decoder_bias_projection_frac": zero,
            }

        corr_x = (
            centered * torch.roll(centered, shifts=patch_size, dims=-1)
        ).mean(dim=(1, 2, 3)) / variance
        corr_y = (
            centered * torch.roll(centered, shifts=patch_size, dims=-2)
        ).mean(dim=(1, 2, 3)) / variance

        # Arrange each patch in the same (p, p, channel) order as JiT's final
        # linear projection.  A repeated decoder template has zero residual
        # after subtracting the mean patch.
        patches = value.unfold(2, patch_size, patch_size).unfold(
            3, patch_size, patch_size
        )
        patches = patches.permute(0, 2, 3, 4, 5, 1).reshape(
            sample_count, -1, patch_size * patch_size * value.size(1)
        )
        patch_residual = patches - patches.mean(dim=1, keepdim=True)
        patch_residual_energy = patch_residual.square().mean(dim=(1, 2))
        patch_spatial_std = patch_residual_energy.sqrt().mean()
        template_explained = (
            1.0 - patch_residual_energy / variance
        ).clamp(min=-1.0, max=1.0).mean()

        if sample_count > 1:
            flat = centered.flatten(1)
            flat = flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
            sample_corr = (flat * torch.roll(flat, shifts=1, dims=0)).sum(
                dim=1
            ).mean()
        else:
            sample_corr = value.new_zeros(())

        bias = getattr(getattr(self.net, "final_layer", None), "linear", None)
        bias = getattr(bias, "bias", None)
        if bias is None or bias.numel() != patches.size(-1):
            bias_projection = value.new_zeros(())
        else:
            bias = bias.detach().float().reshape(1, 1, -1)
            dot = (patches * bias).sum(dim=(1, 2))
            bias_energy = patches.size(1) * bias.square().sum()
            output_energy = patches.square().sum(dim=(1, 2))
            bias_projection = (
                dot.square()
                / (bias_energy * output_energy).clamp_min(1e-12)
            ).clamp(max=1.0).mean()

        return {
            "patch_shift_corr_x16": corr_x.mean(),
            "patch_shift_corr_y16": corr_y.mean(),
            "patch_spatial_std": patch_spatial_std,
            "patch_template_explained_frac": template_explained,
            "sample_output_corr": sample_corr,
            "decoder_bias_projection_frac": bias_projection,
        }

    def forward(self, x, y, return_x_pred=False, return_t=False, **kwargs):
        global_step = int(kwargs.get("global_step", 0))
        labels = self.drop_labels(y) if self.training else y

        if global_step < self.x_adapt_steps:
            return self._forward_x_adaptation(
                x=x,
                labels=labels,
                return_x_pred=return_x_pred,
                return_t=return_t,
            )

        # sCM is a new optimization phase.  Its tangent warmup starts at zero
        # rather than inheriting iterations from TrigFlow x-pred adaptation.
        return self._forward_scm_xpred(
            x=x,
            labels=labels,
            global_step=global_step - self.x_adapt_steps,
            return_x_pred=return_x_pred,
            return_t=return_t,
        )

    def _forward_x_adaptation(
        self,
        x,
        labels,
        return_x_pred,
        return_t,
    ):
        """Velocity-equivalent TrigFlow diffusion loss in x-prediction space."""
        t = self._sample_trigflow_time(x)
        t_img = self._reshape_time(t, x)
        cos_t = torch.cos(t_img)
        sin_t = torch.sin(t_img)

        noise = torch.randn_like(x) * self.sigma_data
        x_t = cos_t * x + sin_t * noise
        network_t = self._network_time(t)
        x_pred = self._net_physical_time(
            x_t / self.sigma_data,
            t,
            labels,
        )
        x_pred_detached = x_pred.detach().float()

        x_mse_per_sample = (
            (x_pred.float() - x.float()).square().flatten(1).mean(dim=1)
        )
        sin_for_loss = self._sin_for_loss(t)
        loss_per_sample = x_mse_per_sample / sin_for_loss.square()
        loss_x_adapt = loss_per_sample.mean()
        (
            boundary_loss,
            boundary_pred,
            boundary_t,
            boundary_mse_per_sample,
        ) = self._boundary_loss(x, labels)
        weighted_boundary_loss = self.boundary_loss_weight * boundary_loss
        loss = loss_x_adapt + weighted_boundary_loss

        x_pred_batch_mean = x_pred_detached.mean(dim=0)
        x_pred_batch_var = (
            x_pred_detached.square().mean()
            - x_pred_batch_mean.square().mean()
        ).clamp_min(0.0)

        # The adaptive time-weight network belongs to the following sCM phase.
        # Touch it with a zero coefficient so DDP sees all trainable parameters.
        if self.loss_weight_net is not None:
            loss = loss + sum(
                parameter.float().sum() * 0.0
                for parameter in self.loss_weight_net.parameters()
            )

        # These are output-space gradient proxies, not full parameter-gradient
        # norms.  Unlike raw loss magnitudes they include all explicit scalar
        # factors and make a vanishingly weak boundary anchor visible without
        # requiring two expensive auxiliary backward passes.
        batch_size = x.size(0)
        elements_per_sample = x[0].numel()
        adapt_output_grad_norm = (
            4.0
            * x_mse_per_sample.detach()
            / (
                batch_size**2
                * elements_per_sample
                * sin_for_loss.pow(4)
            )
        ).sum().sqrt()
        boundary_output_grad_norm = (
            4.0
            * self.boundary_loss_weight**2
            * boundary_mse_per_sample.detach()
            / (batch_size**2 * elements_per_sample)
        ).sum().sqrt()

        loss_dict = {
            "loss_x_adapt": loss_x_adapt.detach(),
            "loss_x_adapt_max": loss_per_sample.detach().max(),
            "x_adapt_x0_mse": x_mse_per_sample.detach().mean(),
            "x_adapt_boundary_mse": boundary_loss.detach(),
            "x_adapt_boundary_weighted": weighted_boundary_loss.detach(),
            "boundary_band_t_mean": boundary_t.detach().float().mean(),
            "boundary_band_t_max": boundary_t.detach().float().max(),
            "boundary_band_mse_max": boundary_mse_per_sample.detach().max(),
            "boundary_band_jit_t_mean": self._network_time(
                boundary_t
            ).detach().float().mean(),
            "boundary_band_jit_t_min": self._network_time(
                boundary_t
            ).detach().float().min(),
            "x_pred_mean": x_pred_detached.mean(),
            "x_pred_rms": x_pred_detached.square().mean().sqrt(),
            "x_pred_abs_gt_1_frac": (x_pred_detached.abs() > 1.0)
            .float()
            .mean(),
            "x_pred_batch_std": x_pred_batch_var.sqrt(),
            "x_loss_sin_clamp_frac": (
                torch.sin(t).detach().float() < self.x_loss_sin_min
            ).float().mean(),
            "sampled_sin_min": torch.sin(t).detach().float().min(),
            "x_adapt_jit_t_mean": network_t.detach().float().mean(),
            "x_adapt_jit_t_min": network_t.detach().float().min(),
            "x_adapt_jit_t_max": network_t.detach().float().max(),
            "adapt_output_grad_norm_proxy": adapt_output_grad_norm,
            "boundary_output_grad_norm_proxy": boundary_output_grad_norm,
            "boundary_to_main_output_grad_ratio": (
                boundary_output_grad_norm
                / adapt_output_grad_norm.clamp_min(1e-12)
            ),
            "t_mean": t_img.detach().float().mean(),
            "phase_x_adapt": loss.detach().new_ones(()),
        }
        loss_dict.update(
            self._time_bin_metrics("x_adapt", x_mse_per_sample, t)
        )
        loss_dict.update(self._collapse_metrics(x_pred_detached))
        return self._format_outputs(
            loss,
            loss_dict,
            x_pred,
            x_t,
            t_img,
            return_x_pred,
            return_t,
        )

    def _forward_scm_xpred(
        self,
        x,
        labels,
        global_step,
        return_x_pred,
        return_t,
    ):
        """Direct x-prediction change of variables of sCM Algorithm 1."""
        if self.tangent_norm_c <= 0.0:
            raise ValueError(
                f"tangent_norm_c must be positive, got {self.tangent_norm_c}"
            )
        if self.tangent_warmup_steps > 0:
            warmup = min(
                1.0,
                max(0.0, global_step / self.tangent_warmup_steps),
            )
        else:
            warmup = 1.0

        t = self._sample_trigflow_time(x)
        t_img = self._reshape_time(t, x)
        network_t = self._network_time(t)
        cos_t = torch.cos(t_img)
        sin_t = torch.sin(t_img)
        cos_sin = cos_t * sin_t

        noise = torch.randn_like(x) * self.sigma_data
        x_t = cos_t * x + sin_t * noise
        conditional_velocity = cos_t * noise - sin_t * x
        network_input = x_t / self.sigma_data
        (
            tangent_velocity,
            teacher_velocity_mix,
            teacher_velocity_rms,
            teacher_prediction_rms,
            teacher_velocity_delta_rms,
            teacher_velocity_sin_clamped,
        ) = self._select_tangent_velocity(
            x_t,
            t,
            labels,
            conditional_velocity,
        )

        # Replay the exact same dropout realization in the trainable primal and
        # the stop-gradient JVP.  D^- is stop-grad w.r.t. parameters, but its
        # derivatives w.r.t. network input and time must remain available.
        if x.is_cuda:
            dropout_rng_state = torch.cuda.get_rng_state(x.device)
        else:
            dropout_rng_state = torch.random.get_rng_state()
        student_precision_context = (
            _no_autocast_context(x)
            if self.jvp_dtype == "fp32"
            else contextlib.nullcontext()
        )
        student_input = (
            network_input.float()
            if self.jvp_dtype == "fp32"
            else network_input
        )
        student_t = t.float() if self.jvp_dtype == "fp32" else t
        # A full-FP32 JVP must use a full-FP32 trainable primal as well;
        # otherwise D^- and dD^-/dt describe slightly different functions.
        with student_precision_context, _jvp_sdpa_context(x):
            d_pred = self._net_physical_time(
                student_input,
                student_t,
                labels,
            )

        x_jvp = network_input.detach().float()
        t_jvp = t_img.detach().float()
        # j = cs * dD/dt.  sCT follows the sampled conditional path; sCD uses
        # the frozen adaptation teacher's PF velocity.  The hybrid mode blends
        # only the input tangent, leaving the physical-time tangent unchanged.
        dx_jvp = (
            cos_sin * tangent_velocity / self.sigma_data
        ).detach().float()
        dt_jvp = cos_sin.detach().float()

        def d_fn(x_in, t_in):
            # Differentiate the complete physical-time wrapper.  Autograd
            # therefore applies da/dt=-2/pi to the JiT time tangent.
            return self._net_physical_time(x_in, t_in, labels)

        if x.is_cuda:
            torch.cuda.set_rng_state(dropout_rng_state, x.device)
        else:
            torch.random.set_rng_state(dropout_rng_state)
        jvp_amp_context = (
            _no_autocast_context(x)
            if self.jvp_dtype == "fp32"
            else contextlib.nullcontext()
        )
        with torch.no_grad(), jvp_amp_context, _jvp_sdpa_context(x):
            jvp_primal, scaled_dD_dt = torch.func.jvp(
                d_fn,
                (x_jvp, t_jvp),
                (dx_jvp, dt_jvp),
            )

        d_teacher = d_pred.detach().float()
        scaled_dD_dt = scaled_dD_dt.detach().float()
        local_target = cos_t.float().square() * (d_teacher - x.float())
        local_flat = local_target.flatten(1)
        jvp_flat = scaled_dD_dt.flatten(1)
        local_target_rms = local_flat.square().mean(dim=1).sqrt()
        jvp_target_rms = jvp_flat.square().mean(dim=1).sqrt()
        local_jvp_cosine = (local_flat * jvp_flat).sum(dim=1) / (
            local_flat.norm(dim=1) * jvp_flat.norm(dim=1)
        ).clamp_min(1e-12)
        jvp_to_local_rms = jvp_target_rms / local_target_rms.clamp_min(1e-12)
        # k = (1-r)c^2(D^- - a_t) + r*cs*dD^-/dt, with a_t=x_0 for sCT.
        tangent_raw = (1.0 - warmup) * local_target + warmup * scaled_dD_dt

        tangent_raw_norm = tangent_raw.flatten(1).norm(dim=1)
        sin_scalar = torch.sin(t).detach().float()
        tangent_denom = (
            tangent_raw_norm + self.tangent_norm_c * sin_scalar
        ).clamp_min(torch.finfo(torch.float32).tiny)
        q_scale = self.sigma_data * sin_scalar / tangent_denom
        tangent = tangent_raw * q_scale.view(
            -1, *([1] * (d_pred.ndim - 1))
        )

        # The plus sign is required because D-D^-=-sigma*sin(t)*(F-F^-).
        residual = d_pred.float() - d_teacher + tangent.detach()
        x_loss_per_sample = residual.square().flatten(1).mean(dim=1)

        # Exact F-space scalar loss written in D units.  Keeping the explicit
        # scale also preserves the meaning and clamp range of w_phi(t).
        sin_for_loss = self._sin_for_loss(t)
        f_equiv_loss_per_sample = x_loss_per_sample / (
            self.sigma_data**2 * sin_for_loss.square()
        )
        loss_scm_mse = f_equiv_loss_per_sample.mean()

        if self.loss_weight_net is not None:
            with _no_autocast_context(x):
                raw_log_weight = self.loss_weight_net(t_img.detach())
                if raw_log_weight.shape != f_equiv_loss_per_sample.shape:
                    raise RuntimeError(
                        "loss_weight_net must return shape "
                        f"{tuple(f_equiv_loss_per_sample.shape)}, got "
                        f"{tuple(raw_log_weight.shape)}"
                    )
                adaptive_clamp_mask = (
                    raw_log_weight.abs() >= self.adaptive_weight_max
                )
                log_weight = raw_log_weight.clamp(
                    min=-self.adaptive_weight_max,
                    max=self.adaptive_weight_max,
                )
                effective_weight = log_weight.exp()
                weighted_per_sample = (
                    effective_weight * f_equiv_loss_per_sample - log_weight
                )
            loss_scm = weighted_per_sample.mean()
            mean_log_weight = log_weight.detach().mean()
            min_log_weight = log_weight.detach().min()
            max_log_weight = log_weight.detach().max()
            min_raw_log_weight = raw_log_weight.detach().min()
            max_raw_log_weight = raw_log_weight.detach().max()
            adaptive_clamp_frac = adaptive_clamp_mask.detach().float().mean()
            shifted_log_weight = (
                log_weight.detach()
                - 2.0
                * torch.log(
                    sin_for_loss.detach() * self.sigma_data
                )
            ).mean()
            per_sample_log_weight = log_weight.detach().float()
        else:
            loss_scm = loss_scm_mse
            mean_log_weight = loss_scm.new_zeros(())
            min_log_weight = loss_scm.new_zeros(())
            max_log_weight = loss_scm.new_zeros(())
            min_raw_log_weight = loss_scm.new_zeros(())
            max_raw_log_weight = loss_scm.new_zeros(())
            adaptive_clamp_frac = loss_scm.new_zeros(())
            shifted_log_weight = loss_scm.new_zeros(())
            effective_weight = torch.ones_like(f_equiv_loss_per_sample)
            per_sample_log_weight = torch.zeros_like(
                f_equiv_loss_per_sample
            )

        (
            boundary_loss,
            boundary_pred,
            boundary_t,
            boundary_mse_per_sample,
        ) = self._boundary_loss(x, labels)
        weighted_boundary_loss = self.boundary_loss_weight * boundary_loss
        loss = loss_scm + weighted_boundary_loss

        d_pred_batch_mean = d_teacher.mean(dim=0)
        d_pred_batch_var = (
            d_teacher.square().mean()
            - d_pred_batch_mean.square().mean()
        ).clamp_min(0.0)

        batch_size = x.size(0)
        elements_per_sample = x[0].numel()
        scm_output_grad_norm = (
            4.0
            * effective_weight.detach().square()
            * x_loss_per_sample.detach()
            / (
                batch_size**2
                * elements_per_sample
                * self.sigma_data**4
                * sin_for_loss.pow(4)
            )
        ).sum().sqrt()
        boundary_output_grad_norm = (
            4.0
            * self.boundary_loss_weight**2
            * boundary_mse_per_sample.detach()
            / (batch_size**2 * elements_per_sample)
        ).sum().sqrt()

        tangent_norm = tangent.flatten(1).norm(dim=1)
        jvp_primal_error = jvp_primal.detach().float() - d_teacher
        jvp_primal_relative_rms = (
            jvp_primal_error.square().mean().sqrt()
            / d_teacher.square().mean().sqrt().clamp_min(1e-12)
        )
        d_x0_mse_per_sample = (
            (d_teacher - x.detach().float()).square().flatten(1).mean(dim=1)
        )
        conditional_velocity_rms = (
            conditional_velocity.detach().float().flatten(1).square()
            .mean(dim=1).sqrt()
        )
        tangent_velocity_rms = (
            tangent_velocity.detach().float().flatten(1).square()
            .mean(dim=1).sqrt()
        )
        loss_dict = {
            "loss_scm": loss_scm.detach(),
            # Compatibility key: this is the F-equivalent scaled MSE.
            "loss_scm_mse": loss_scm_mse.detach(),
            "loss_scm_x_mse": x_loss_per_sample.detach().mean(),
            "loss_scm_boundary_mse": boundary_loss.detach(),
            "loss_scm_boundary_weighted": weighted_boundary_loss.detach(),
            "boundary_band_t_mean": boundary_t.detach().float().mean(),
            "boundary_band_t_max": boundary_t.detach().float().max(),
            "boundary_band_mse_max": boundary_mse_per_sample.detach().max(),
            "boundary_band_jit_t_mean": self._network_time(
                boundary_t
            ).detach().float().mean(),
            "boundary_band_jit_t_min": self._network_time(
                boundary_t
            ).detach().float().min(),
            "x_pred_mean": d_teacher.mean(),
            "x_pred_rms": d_teacher.square().mean().sqrt(),
            "x_pred_abs_gt_1_frac": (d_teacher.abs() > 1.0).float().mean(),
            "x_pred_batch_std": d_pred_batch_var.sqrt(),
            "tangent_k_norm": tangent_raw_norm.detach().mean(),
            "tangent_k_norm_max": tangent_raw_norm.detach().max(),
            "tangent_q_norm": tangent_norm.detach().mean(),
            "tangent_local_target_rms": local_target_rms.detach().mean(),
            "tangent_jvp_target_rms": jvp_target_rms.detach().mean(),
            "tangent_jvp_to_local_rms": jvp_to_local_rms.detach().mean(),
            "tangent_local_jvp_cosine": local_jvp_cosine.detach().mean(),
            "teacher_velocity_mix": teacher_velocity_mix.detach().mean(),
            "teacher_velocity_active_frac": (
                teacher_velocity_mix.detach() > 0.0
            ).float().mean(),
            "teacher_velocity_sin_clamp_frac": (
                teacher_velocity_sin_clamped.detach().float().mean()
            ),
            "conditional_velocity_rms": conditional_velocity_rms.mean(),
            "tangent_velocity_rms": tangent_velocity_rms.mean(),
            "teacher_velocity_rms": teacher_velocity_rms.mean(),
            "teacher_prediction_rms": teacher_prediction_rms.mean(),
            "teacher_velocity_delta_rms": (
                teacher_velocity_delta_rms.mean()
            ),
            "tangent_warmup": loss.new_tensor(warmup).detach(),
            "adaptive_log_weight": mean_log_weight,
            "adaptive_log_weight_min": min_log_weight,
            "adaptive_log_weight_max": max_log_weight,
            "adaptive_log_weight_raw_min": min_raw_log_weight,
            "adaptive_log_weight_raw_max": max_raw_log_weight,
            "adaptive_weight_clamp_frac": adaptive_clamp_frac,
            "adaptive_shifted_log_weight": shifted_log_weight,
            "jvp_primal_error_max": jvp_primal_error.abs().max(),
            "jvp_primal_relative_rms": jvp_primal_relative_rms,
            "x_loss_sin_clamp_frac": (
                sin_scalar < self.x_loss_sin_min
            ).float().mean(),
            "sampled_sin_min": sin_scalar.min(),
            "scm_jit_t_mean": network_t.detach().float().mean(),
            "scm_jit_t_min": network_t.detach().float().min(),
            "scm_jit_t_max": network_t.detach().float().max(),
            "scm_output_grad_norm_proxy": scm_output_grad_norm,
            "boundary_output_grad_norm_proxy": boundary_output_grad_norm,
            "boundary_to_main_output_grad_ratio": (
                boundary_output_grad_norm
                / scm_output_grad_norm.clamp_min(1e-12)
            ),
            "t_mean": t_img.detach().float().mean(),
        }
        loss_dict.update(
            self._time_bin_metrics("scm_x0", d_x0_mse_per_sample, t)
        )
        loss_dict.update(
            self._target_bin_metric_sums(
                t,
                {
                    "local_rms": local_target_rms,
                    "jvp_rms": jvp_target_rms,
                    "jvp_to_local_rms": jvp_to_local_rms,
                    "local_jvp_cosine": local_jvp_cosine,
                    "adaptive_log_weight": per_sample_log_weight,
                    "adaptive_weight": effective_weight.detach().float(),
                    "x0_mse": d_x0_mse_per_sample,
                    "teacher_mix": teacher_velocity_mix,
                    "conditional_velocity_rms": conditional_velocity_rms,
                    "tangent_velocity_rms": tangent_velocity_rms,
                    "teacher_velocity_rms": teacher_velocity_rms,
                    "teacher_velocity_delta_rms": (
                        teacher_velocity_delta_rms
                    ),
                },
            )
        )
        loss_dict.update(self._collapse_metrics(d_teacher))
        return self._format_outputs(
            loss,
            loss_dict,
            d_pred,
            x_t,
            t_img,
            return_x_pred,
            return_t,
        )

    def _scm_consistency_output(self, x_t, t, labels, cfg=1.0):
        """Return the direct clean prediction D; no F-space wrapper is used."""
        d_cond = self._net_physical_time(
            x_t / self.sigma_data,
            t,
            labels,
        )
        if cfg == 1.0:
            return d_cond
        null_labels = torch.full_like(labels, self.num_classes)
        d_uncond = self._net_physical_time(
            x_t / self.sigma_data,
            t,
            null_labels,
        )
        return d_uncond + cfg * (d_cond - d_uncond)

    def sample_images_with_grad(self, x, y, sampling_args=None):
        sampling_args = sampling_args or {}
        t_start = sampling_args.get("t_start")
        if t_start is None:
            scalar_t = torch.atan(x.new_tensor(self.sigma_max / self.sigma_data))
            t = scalar_t.expand(x.size(0)).view(-1, 1, 1, 1)
        else:
            t = t_start.to(device=x.device, dtype=x.dtype).view(-1, 1, 1, 1)
        output = self._scm_consistency_output(x, t, y, cfg=1.0)
        if sampling_args.get("return_velocity", False):
            sin_t = torch.sin(t.float())
            if torch.any(sin_t <= 0.0):
                raise ValueError(
                    "direct x-pred velocity is undefined at t=0; "
                    "return_velocity requires strictly positive t_start"
                )
            velocity = (
                torch.cos(t.float()) * x.float() - output.float()
            ) / sin_t
            return output, velocity.to(dtype=output.dtype)
        return output

    @torch.inference_mode()
    def generate(self, n_samples, labels, cfg=1.0, args=None, verbose=True, z_t=None):
        del verbose
        if z_t is None:
            x_t = torch.randn(
                n_samples,
                3,
                self.img_size,
                self.img_size,
                device=labels.device,
            ) * self.sigma_data
        else:
            x_t = z_t

        t_max = torch.atan(x_t.new_tensor(self.sigma_max / self.sigma_data))
        t = t_max.expand(n_samples).view(-1, 1, 1, 1)
        x = self._scm_consistency_output(x_t, t, labels, cfg=cfg)

        num_steps = int(getattr(args, "num_sampling_steps", 1))
        if num_steps == 1:
            return x
        if num_steps != 2:
            raise ValueError("x-pred sCM sampling currently supports 1 or 2 steps")

        t_mid_value = float(getattr(args, "scm_intermediate_t", 1.1))
        t_max_value = float(t_max)
        if not 0.0 < t_mid_value < t_max_value:
            raise ValueError(
                "scm_intermediate_t must satisfy 0 < t_mid < t_max="
                f"{t_max_value:.6f}, got {t_mid_value}"
            )
        t_mid = x.new_full((n_samples, 1, 1, 1), t_mid_value)
        noise = torch.randn_like(x) * self.sigma_data
        x_mid = torch.cos(t_mid) * x + torch.sin(t_mid) * noise
        return self._scm_consistency_output(x_mid, t_mid, labels, cfg=cfg)


def create_model(args):
    trainer.logger.info(
        "two-phase direct x-pred schedule: init=%s x-adapt=%d steps at "
        "lr=%s, sCM=%d steps, JiT-time=1-2*t_TrigFlow/pi, "
        "tangent-velocity=%s",
        args.init_parameterization,
        args.x_adapt_steps,
        args.x_adapt_lr,
        args.scm_steps,
        args.tangent_velocity_mode,
    )
    return trainer.create_model(
        args,
        model_cls=SCMJiTDenoiser,
        x_adapt_steps=args.x_adapt_steps,
        x_loss_sin_min=args.x_loss_sin_min,
        boundary_loss_weight=args.boundary_loss_weight,
        boundary_band_max=args.boundary_band_max,
        deterministic_boundary=args.deterministic_boundary,
        collapse_monitor_samples=args.collapse_monitor_samples,
        network_time_mode="legacy_reversed",
        tangent_velocity_mode=args.tangent_velocity_mode,
        hybrid_teacher_start=args.hybrid_teacher_start,
        hybrid_teacher_end=args.hybrid_teacher_end,
        teacher_velocity_sin_min=args.teacher_velocity_sin_min,
    )


def adjust_two_phase_lr(optimizer, step, args):
    if step < args.x_adapt_steps:
        base_lr = args.x_adapt_lr if args.x_adapt_lr is not None else args.lr
        if step < args.x_adapt_warmup_steps:
            lr = base_lr * float(step + 1) / max(1, args.x_adapt_warmup_steps)
        else:
            lr = base_lr
    else:
        scm_step = step - args.x_adapt_steps
        if scm_step < args.warmup_steps:
            lr = args.lr * float(scm_step + 1) / max(1, args.warmup_steps)
        elif args.lr_sched == "constant":
            lr = args.lr
        elif args.lr_sched == "edm2":
            lr = args.lr / math.sqrt(
                max(float(scm_step + 1) / args.lr_ref_steps, 1.0)
            )
        else:
            progress = (scm_step - args.warmup_steps) / max(
                1, args.scm_steps - args.warmup_steps
            )
            progress = min(max(progress, 0.0), 1.0)
            lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )

    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


@torch.no_grad()
def copy_ema_to_student(model, ema_model):
    """Permanently initialize the student from the default adaptation EMA."""
    if ema_model is None:
        raise RuntimeError(
            "x-pred adaptation-to-sCM transition requires EMA; "
            "do not use --disable_ema with --x_adapt_steps > 0"
        )

    raw_model = trainer.unwrap_model(model)
    shadow = ema_model.state_dict(label=ema_model.default_label)
    missing = []
    shape_mismatches = []
    copied = 0
    for name, parameter in raw_model.named_parameters():
        normalized_name = normalize_param_name(name)
        source = shadow.get(normalized_name)
        if source is None:
            missing.append(normalized_name)
            continue
        if source.shape != parameter.shape:
            shape_mismatches.append(
                (normalized_name, tuple(source.shape), tuple(parameter.shape))
            )
            continue
        parameter.copy_(source.to(device=parameter.device, dtype=parameter.dtype))
        copied += 1

    if missing or shape_mismatches:
        raise RuntimeError(
            "cannot initialize sCM student from adaptation EMA: "
            f"copied={copied}, missing={missing[:5]}, "
            f"shape_mismatches={shape_mismatches[:5]}"
        )
    trainer.logger.info(
        "copied adaptation EMA '%s' into sCM student (%d tensors)",
        ema_model.default_label,
        copied,
    )


def maybe_transition_to_scm(
    step,
    args,
    model,
    optimizer,
    ema_model,
    world_size,
):
    if args.x_adapt_steps <= 0 or step != args.x_adapt_steps:
        return optimizer, ema_model

    trainer.logger.info(
        "TrigFlow x-pred adaptation complete at step=%d; copying adaptation "
        "EMA to student, then resetting Adam and EMA for direct x-pred sCM",
        step,
    )
    copy_ema_to_student(model, ema_model)
    return (
        trainer.create_optimizer(args, model, world_size),
        trainer.create_ema_model(args, model, world_size),
    )


def get_args_parser():
    parser = trainer.get_args_parser()
    parser.prog = "train_scm_jit_b.py"
    parser.description = (
        "JiT-B TrigFlow x-pred adaptation followed by direct x-pred sCM"
    )
    parser.set_defaults(
        objective="scm",
        checkpoint_key="model_ema2",
        exp_name="jit_b_scm_xpred_reversed_time_boundary_band",
        P_mean=-1.0,
        P_std=1.6,
        sigma_data=0.5,
        sigma_max=80.0,
        tangent_norm_c=0.1,
        tangent_warmup_steps=10000,
        adaptive_weight_max=20.0,
        beta1=0.9,
        beta2=0.99,
        adam_eps=1e-11,
        lr_sched="edm2",
        lr_ref_steps=35000.0,
        ema_type="power",
        ema_sigma_rel=0.05,
    )
    parser.add_argument(
        "--init_parameterization",
        default="x_pred",
        choices=["x_pred", "trigflow_x_pred_reversed_time"],
        help=(
            "semantic meaning of the checkpoint head; x_pred denotes the "
            "legacy linear-flow checkpoint and requires adaptation; the "
            "already-adapted option must use permanent reversed JiT time"
        ),
    )
    parser.add_argument(
        "--x_adapt_steps",
        "--f_adapt_steps",
        dest="x_adapt_steps",
        type=int,
        default=10000,
        help=(
            "TrigFlow x-pred diffusion-adaptation steps before sCM; "
            "--f_adapt_steps is a deprecated compatibility alias"
        ),
    )
    parser.add_argument(
        "--x_adapt_lr",
        "--f_adapt_lr",
        dest="x_adapt_lr",
        type=float,
        default=1e-4,
        help=(
            "learning rate for x-pred adaptation; --f_adapt_lr is a "
            "deprecated compatibility alias"
        ),
    )
    parser.add_argument(
        "--x_adapt_warmup_steps",
        "--f_adapt_warmup_steps",
        dest="x_adapt_warmup_steps",
        type=int,
        default=1250,
        help=(
            "linear LR warmup in x-pred adaptation; "
            "--f_adapt_warmup_steps is a deprecated compatibility alias"
        ),
    )
    parser.add_argument(
        "--x_loss_sin_min",
        type=float,
        default=1e-3,
        help=(
            "numerical floor for sin(t) in explicit x-space loss scaling; "
            "1e-3 caps the largest inverse-square factor at 1e6"
        ),
    )
    parser.add_argument(
        "--tangent_velocity_mode",
        choices=["sct", "hybrid", "scd"],
        default="sct",
        help=(
            "input-path velocity used by the stop-gradient JVP: sampled "
            "conditional sCT velocity, frozen-teacher PF velocity, or a "
            "smooth high-noise hybrid"
        ),
    )
    parser.add_argument(
        "--hybrid_teacher_start",
        type=float,
        default=1.50,
        help="physical TrigFlow angle where smooth teacher blending begins",
    )
    parser.add_argument(
        "--hybrid_teacher_end",
        type=float,
        default=1.53,
        help="physical angle where hybrid reaches full teacher PF velocity",
    )
    parser.add_argument(
        "--teacher_velocity_sin_min",
        type=float,
        default=1e-3,
        help=(
            "sin(t) floor in v_T=(cos(t)x_t-D_T)/sin(t); irrelevant to the "
            "default high-noise hybrid but protects the all-sCD control"
        ),
    )
    parser.add_argument(
        "--boundary_loss_weight",
        type=float,
        default=1.0,
        help=(
            "positive MSE weight anchoring D to x0 in the boundary band; "
            "required to exclude input-independent direct-x-pred solutions"
        ),
    )
    parser.add_argument(
        "--boundary_band_max",
        type=float,
        default=0.02,
        help="sample boundary anchors uniformly over t in [0, this angle]",
    )
    boundary_dropout_group = parser.add_mutually_exclusive_group()
    boundary_dropout_group.add_argument(
        "--deterministic_boundary",
        dest="deterministic_boundary",
        action="store_true",
        help="disable JiT dropout only during the boundary-band forward",
    )
    boundary_dropout_group.add_argument(
        "--stochastic_boundary",
        dest="deterministic_boundary",
        action="store_false",
        help="leave JiT dropout active during the boundary-band forward",
    )
    parser.set_defaults(deterministic_boundary=True)
    parser.add_argument(
        "--collapse_monitor_samples",
        type=int,
        default=8,
        help="number of predictions used for inexpensive patch-collapse metrics",
    )
    parser.add_argument(
        "--patch_collapse_abort_corr",
        type=float,
        default=0.98,
        help=(
            "abort after repeated logs when both horizontal and vertical "
            "one-patch shift correlations exceed this value; <=0 disables"
        ),
    )
    parser.add_argument(
        "--patch_collapse_abort_patience",
        type=int,
        default=5,
        help="number of consecutive logging intervals required before aborting",
    )
    parser.add_argument(
        "--patch_collapse_abort_template_frac",
        type=float,
        default=0.95,
        help="minimum repeated-patch explained fraction used by the abort guard",
    )
    parser.add_argument(
        "--patch_collapse_abort_sample_corr",
        type=float,
        default=0.95,
        help="minimum cross-sample output correlation used by the abort guard",
    )
    parser.add_argument(
        "--scm_steps",
        type=int,
        default=None,
        help="sCM optimizer steps; defaults to epochs * steps_per_epoch",
    )
    return parser


def validate_and_finalize_args(parser, args):
    if args.objective != "scm":
        parser.error("train_scm_jit_b.py only supports --objective scm")
    if args.x_adapt_steps < 0:
        parser.error("--x_adapt_steps must be non-negative")
    if args.x_adapt_warmup_steps < 0:
        parser.error("--x_adapt_warmup_steps must be non-negative")
    if args.x_adapt_lr is not None and args.x_adapt_lr <= 0.0:
        parser.error("--x_adapt_lr must be positive")
    if args.x_loss_sin_min <= 0.0:
        parser.error("--x_loss_sin_min must be positive")
    if not 0.0 < args.teacher_velocity_sin_min <= 1.0:
        parser.error("--teacher_velocity_sin_min must be in (0, 1]")
    if not (
        0.0
        < args.hybrid_teacher_start
        < args.hybrid_teacher_end
        < 0.5 * math.pi
    ):
        parser.error(
            "hybrid interval must satisfy 0 < --hybrid_teacher_start < "
            "--hybrid_teacher_end < pi/2"
        )
    if args.tangent_velocity_mode != "sct" and args.compile:
        parser.error(
            "--compile is not supported with the unregistered frozen "
            "reference JiT used by hybrid/sCD"
        )
    if args.boundary_loss_weight <= 0.0:
        parser.error(
            "--boundary_loss_weight must be positive for direct x-pred sCM; "
            "zero admits an input-independent zero-tangent solution"
        )
    if not 0.0 < args.boundary_band_max < 0.5 * math.pi:
        parser.error("--boundary_band_max must be in (0, pi/2)")
    if args.collapse_monitor_samples <= 0:
        parser.error("--collapse_monitor_samples must be positive")
    if args.patch_collapse_abort_corr > 1.0:
        parser.error(
            "--patch_collapse_abort_corr must be <= 1; values <= 0 disable it"
        )
    if args.patch_collapse_abort_patience <= 0:
        parser.error("--patch_collapse_abort_patience must be positive")
    if not -1.0 <= args.patch_collapse_abort_template_frac <= 1.0:
        parser.error(
            "--patch_collapse_abort_template_frac must be in [-1, 1]"
        )
    if not -1.0 <= args.patch_collapse_abort_sample_corr <= 1.0:
        parser.error(
            "--patch_collapse_abort_sample_corr must be in [-1, 1]"
        )
    if args.x_adapt_steps > 0 and args.disable_ema:
        parser.error(
            "--disable_ema is incompatible with x-pred adaptation because "
            "the adaptation EMA initializes the sCM phase"
        )
    minimum_adaptive_log_weight = math.log(3 * args.img_size * args.img_size)
    if (
        not args.disable_adaptive_weighting
        and args.adaptive_weight_max < minimum_adaptive_log_weight
    ):
        parser.error(
            "--adaptive_weight_max is too small for per-pixel direct x-pred "
            f"sCM at {args.img_size}px: need at least log(3HW)="
            f"{minimum_adaptive_log_weight:.3f}, got "
            f"{args.adaptive_weight_max}"
        )
    if args.init_parameterization == "x_pred" and args.x_adapt_steps == 0:
        parser.error(
            "a legacy x_pred checkpoint requires --x_adapt_steps > 0; use "
            "--init_parameterization trigflow_x_pred_reversed_time for an "
            "already-adapted permanent-reversed-time checkpoint"
        )
    if (
        args.init_parameterization == "trigflow_x_pred_reversed_time"
        and args.x_adapt_steps != 0
    ):
        parser.error(
            "a trigflow_x_pred_reversed_time checkpoint should use "
            "--x_adapt_steps 0"
        )

    args.scm_steps = (
        args.epochs * args.steps_per_epoch
        if args.scm_steps is None
        else int(args.scm_steps)
    )
    if args.scm_steps <= 0:
        parser.error("--scm_steps must be positive")
    args.total_steps_override = args.x_adapt_steps + args.scm_steps
    args.training_entry = "train_scm_jit_b_xpred_reversed_time"
    args.raw_output_parameterization = "TrigFlow_x_pred_D"
    args.boundary_condition = f"supervised_band_[0,{args.boundary_band_max}]"
    args.phase_transition_initialization = "adaptation_ema_to_student"
    args.tangent_velocity_reference = args.tangent_velocity_mode
    args.reference_teacher_initialization = (
        f"frozen_copy_of_{args.checkpoint_key}_from_{args.load_from}"
        if args.tangent_velocity_mode != "sct"
        else "none"
    )
    args.enforce_checkpoint_whitelist = True
    args.checkpoint_allowed_missing_prefixes = ["loss_weight_net."]
    args.checkpoint_allowed_unexpected_prefixes = []
    args.checkpoint_time_convention = "legacy_raw_t1_data"
    args.training_time_convention = "trigflow_angle_t0_data"
    args.jit_time_conditioning = "permanent_1_minus_2t_over_pi"
    args.network_time_mode = "legacy_reversed"
    args.time_embedding_bridge = "none"
    time_metadata = {
        "jit_time_conditioning": args.jit_time_conditioning,
        "network_time_mode": args.network_time_mode,
    }
    args.checkpoint_required_metadata = (
        time_metadata
        if args.init_parameterization == "trigflow_x_pred_reversed_time"
        else {}
    )
    args.resume_required_metadata = time_metadata
    return args


if __name__ == "__main__":
    argument_parser = get_args_parser()
    arguments = validate_and_finalize_args(
        argument_parser,
        argument_parser.parse_args(),
    )
    trainer.train(
        arguments,
        model_factory=create_model,
        lr_adjuster=adjust_two_phase_lr,
        phase_transition=maybe_transition_to_scm,
    )
