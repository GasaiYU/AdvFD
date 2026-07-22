import contextlib

import torch
import torch.nn as nn

from models.denoiser_jit import JiTDenoiser


class AdaptiveTimeWeight(nn.Module):
    """Small learned log-variance model used by sCM adaptive weighting."""

    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, t):
        return self.net(t.reshape(-1, 1).float()).flatten()


def _jvp_sdpa_context(x):
    attention = getattr(torch.nn, "attention", None)
    sdpa_kernel = getattr(attention, "sdpa_kernel", None)
    sdp_backend = getattr(attention, "SDPBackend", None)
    if sdpa_kernel is not None and sdp_backend is not None:
        # The CPU flash SDPA kernel in PyTorch 2.4 has no forward-AD rule.
        # Selecting MATH is therefore required for CPU diagnostics as well as
        # for deterministic CUDA JVPs.
        return sdpa_kernel(sdp_backend.MATH)

    if not x.is_cuda:
        return contextlib.nullcontext()

    sdp_kernel = getattr(torch.backends.cuda, "sdp_kernel", None)
    if sdp_kernel is not None:
        return sdp_kernel(
            enable_flash=False,
            enable_math=True,
            enable_mem_efficient=False,
        )

    return contextlib.nullcontext()


def _no_autocast_context(x):
    if x.is_cuda:
        return torch.cuda.amp.autocast(enabled=False)
    return contextlib.nullcontext()


class MeanFlowJiTDenoiser(JiTDenoiser):
    """JiT denoiser with legacy MeanFlow and stabilized continuous-CM losses.

    The legacy MeanFlow branch predicts the data endpoint and converts it to
    average velocity.  The sCM branch follows TrigFlow and interprets the same
    JiT output as the paper's velocity-like ``F_theta`` parameterization.
    """

    def __init__(
        self,
        *args,
        meanflow_delta=0.01,
        dudt_weight=0.5,
        dudt_drop_prob=0.75,
        dudt_clip_norm=0.0,
        loss_clip=0.0,
        objective="meanflow",
        tangent_norm_c=0.1,
        tangent_warmup_steps=10000,
        adaptive_weighting=True,
        adaptive_weight_hidden=64,
        adaptive_weight_max=20.0,
        sigma_data=0.5,
        sigma_max=80.0,
        jvp_dtype="amp",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.meanflow_delta = float(meanflow_delta)
        self.dudt_weight = float(dudt_weight)
        self.dudt_drop_prob = float(dudt_drop_prob)
        self.dudt_clip_norm = float(dudt_clip_norm)
        self.loss_clip = float(loss_clip)
        if objective not in ("meanflow", "scm"):
            raise ValueError(f"unsupported objective: {objective}")
        self.objective = objective
        self.tangent_norm_c = float(tangent_norm_c)
        self.tangent_warmup_steps = int(tangent_warmup_steps)
        self.adaptive_weighting = bool(adaptive_weighting)
        self.adaptive_weight_max = float(adaptive_weight_max)
        self.sigma_data = float(sigma_data)
        self.sigma_max = float(sigma_max)
        if jvp_dtype not in ("amp", "fp32"):
            raise ValueError(f"unsupported JVP dtype: {jvp_dtype}")
        self.jvp_dtype = jvp_dtype
        if self.sigma_data <= 0.0:
            raise ValueError(f"sigma_data must be positive, got {self.sigma_data}")
        if self.sigma_max <= 0.0:
            raise ValueError(f"sigma_max must be positive, got {self.sigma_max}")
        self.loss_weight_net = (
            AdaptiveTimeWeight(hidden_dim=int(adaptive_weight_hidden))
            if self.adaptive_weighting
            else None
        )

    def forward(self, x, y, return_x_pred=False, return_t=False, **kwargs):
        labels = self.drop_labels(y) if self.training else y

        if self.objective == "scm":
            return self._forward_scm(
                x=x,
                labels=labels,
                global_step=int(kwargs.get("global_step", 0)),
                return_x_pred=return_x_pred,
                return_t=return_t,
            )

        delta = float(kwargs.get("meanflow_delta", self.meanflow_delta))
        dudt_weight = float(kwargs.get("dudt_weight", self.dudt_weight))
        dudt_drop_prob = float(kwargs.get("dudt_drop_prob", self.dudt_drop_prob))
        dudt_clip_norm = float(kwargs.get("dudt_clip_norm", self.dudt_clip_norm))
        loss_clip = float(kwargs.get("loss_clip", self.loss_clip))

        if delta <= 0.0:
            raise ValueError(f"meanflow_delta must be positive, got {delta}")
        min_t = max(float(self.t_eps), delta)
        max_t = 1.0 - delta
        if min_t >= max_t:
            raise ValueError(
                f"Invalid timestep range [{min_t}, {max_t}]. "
                "Use a smaller meanflow_delta or t_eps."
            )

        t = self.sample_t(x.size(0), device=x.device).clamp(min=min_t, max=max_t)
        t_img = t.view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        # Standard convention: t=0 is data, t=1 is noise.
        z = (1.0 - t_img) * x + t_img * e
        v = e - x

        x_pred = self.net(z, self._backbone_t(t_img).flatten(), labels)

        return self._forward_meanflow(
            x_pred=x_pred,
            z=z,
            t_img=t_img,
            v=v,
            labels=labels,
            dudt_weight=dudt_weight,
            dudt_drop_prob=dudt_drop_prob,
            dudt_clip_norm=dudt_clip_norm,
            loss_clip=loss_clip,
            return_x_pred=return_x_pred,
            return_t=return_t,
        )

    def _forward_scm(
        self,
        x,
        labels,
        global_step,
        return_x_pred,
        return_t,
    ):
        """Algorithm 1 of sCM for consistency training under TrigFlow."""
        if self.tangent_norm_c <= 0.0:
            raise ValueError(
                f"tangent_norm_c must be positive, got {self.tangent_norm_c}"
            )
        if self.tangent_warmup_steps > 0:
            warmup = min(1.0, max(0.0, global_step / self.tangent_warmup_steps))
        else:
            warmup = 1.0

        tau = (
            torch.randn(x.size(0), device=x.device) * self.P_std + self.P_mean
        )
        # Algebraically identical to atan(exp(tau) / sigma_data), but exp()
        # only sees non-positive inputs.  This keeps the log-normal tails from
        # overflowing without clipping or changing the timestep distribution.
        log_sigma = tau.new_tensor(self.sigma_data).log()
        log_ratio = tau - log_sigma
        acute_t = torch.atan(torch.exp(-log_ratio.abs()))
        t = torch.where(log_ratio >= 0.0, 0.5 * torch.pi - acute_t, acute_t)
        t_img = t.view(-1, *([1] * (x.ndim - 1)))
        cos_t = torch.cos(t_img)
        sin_t = torch.sin(t_img)
        noise = torch.randn_like(x) * self.sigma_data
        x_t = cos_t * x + sin_t * noise
        dx_dt = cos_t * noise - sin_t * x

        network_input = x_t / self.sigma_data

        # Save the dropout RNG state so the trainable forward and the
        # stop-gradient JVP use the same stochastic network realization.
        if x.is_cuda:
            dropout_rng_state = torch.cuda.get_rng_state(x.device)
        else:
            dropout_rng_state = torch.random.get_rng_state()
        # Use the same SDPA implementation for the trainable primal and the
        # stop-gradient JVP.  Restoring RNG state is only sufficient to replay
        # attention dropout when both forwards use the same backend.
        with _jvp_sdpa_context(x):
            f_pred = self.net(network_input, t, labels)

        x_jvp = network_input.detach().float()
        t_jvp = t_img.detach().float()
        # sCM Sec. 5.1 moves cos(t) * sin(t) * sigma_data inside the
        # JVP.  The resulting tangents are bounded and avoid large FP16/BF16
        # intermediates near either endpoint of the TrigFlow trajectory.
        cos_sin = cos_t * sin_t
        dx_jvp = (cos_sin * dx_dt).detach().float()
        dt_jvp = (cos_sin * self.sigma_data).detach().float()

        def f_fn(x_in, t_in):
            return self.net(x_in, t_in.flatten(), labels)

        if x.is_cuda:
            torch.cuda.set_rng_state(dropout_rng_state, x.device)
        else:
            torch.random.set_rng_state(dropout_rng_state)
        # Keep forward-mode AD out of the reverse graph.  AMP is safe here
        # because neither the JVP primal nor tangent participates in backward;
        # ``fp32`` remains available as a conservative debugging fallback.
        jvp_amp_context = (
            _no_autocast_context(x)
            if self.jvp_dtype == "fp32"
            else contextlib.nullcontext()
        )
        with torch.no_grad(), jvp_amp_context, _jvp_sdpa_context(x):
            jvp_primal, scaled_df_dt = torch.func.jvp(
                f_fn,
                (x_jvp, t_jvp),
                (dx_jvp, dt_jvp),
            )

        f_teacher = f_pred.detach()
        scaled_df_dt = scaled_df_dt.detach()
        # Eq. (6) / Algorithm 1 JVP rearrangement.  This is cos(t) times
        # the tangent of the consistency endpoint, expressed in F-space.
        tangent_raw = (
            -cos_t.square() * (self.sigma_data * f_teacher - dx_dt)
            - warmup * (cos_sin * x_t + scaled_df_dt)
        )
        tangent_raw_norm = tangent_raw.float().flatten(1).norm(dim=1)
        tangent_scale = 1.0 / (tangent_raw_norm + self.tangent_norm_c)
        tangent = tangent_raw * tangent_scale.view(
            -1, *([1] * (f_pred.ndim - 1))
        ).to(tangent_raw.dtype)

        residual = f_pred.float() - f_teacher.detach().float() - tangent.detach().float()
        loss_per_sample = (
            residual.pow(2).flatten(1).mean(dim=1)
        )
        loss_scm_mse = loss_per_sample.mean()

        if self.loss_weight_net is not None:
            with _no_autocast_context(x):
                log_weight = self.loss_weight_net(t_img.detach())
                log_weight = log_weight.clamp(
                    min=-self.adaptive_weight_max,
                    max=self.adaptive_weight_max,
                )
                weighted_per_sample = log_weight.exp() * loss_per_sample - log_weight
            loss = weighted_per_sample.mean()
            mean_log_weight = log_weight.detach().mean()
        else:
            loss = loss_scm_mse
            mean_log_weight = loss.new_zeros(())

        tangent_norm = tangent.float().flatten(1).norm(dim=1)
        x_pred = cos_t * x_t - sin_t * self.sigma_data * f_pred
        loss_dict = {
            "loss_scm": loss.detach(),
            "loss_scm_mse": loss_scm_mse.detach(),
            "tangent_norm_raw": tangent_raw_norm.detach().mean(),
            "tangent_norm_raw_max": tangent_raw_norm.detach().max(),
            "tangent_norm": tangent_norm.detach().mean(),
            "tangent_warmup": loss.new_tensor(warmup).detach(),
            "adaptive_log_weight": mean_log_weight,
            "jvp_primal_error_max": (
                jvp_primal.detach().float() - f_teacher.float()
            ).abs().max(),
            "t_mean": t_img.detach().float().mean(),
        }
        return self._format_outputs(
            loss, loss_dict, x_pred, x_t, t_img, return_x_pred, return_t
        )

    def _forward_meanflow(
        self,
        x_pred,
        z,
        t_img,
        v,
        labels,
        dudt_weight,
        dudt_drop_prob,
        dudt_clip_norm,
        loss_clip,
        return_x_pred,
        return_t,
    ):
        u = (z - x_pred) / t_img.clamp_min(self.t_eps)

        if dudt_drop_prob > 0.0:
            keep_prob = max(0.0, 1.0 - dudt_drop_prob)
            keep_flat = torch.rand(z.size(0), device=z.device) < keep_prob
        else:
            keep_flat = torch.ones(z.size(0), device=z.device, dtype=torch.bool)
        keep = keep_flat.view(-1, *([1] * (z.ndim - 1)))

        def u_fn(z_in, t_in, labels_in):
            x_pred_in = self.net(z_in, self._backbone_t(t_in).flatten(), labels_in)
            return (z_in - x_pred_in) / t_in.clamp_min(self.t_eps)

        # Total derivative along the true data->noise path:
        # d/dt u(z_t,t) = J_u(z_t,t) @ (v, 1).  Detaching the correction keeps
        # the target stop-grad while allowing the current u branch to train.  It
        # is computed only for kept samples; dropped samples use u_tgt = v.
        tdudt = torch.zeros_like(v)
        if keep_flat.any():
            z_jvp = z.detach()[keep_flat].float()
            t_jvp = t_img.detach()[keep_flat].float()
            v_jvp = v.detach()[keep_flat].float()
            labels_jvp = labels[keep_flat]
            with torch.no_grad(), _no_autocast_context(z), _jvp_sdpa_context(z):
                _, du_dt = torch.func.jvp(
                    lambda z_in, t_in: u_fn(z_in, t_in, labels_jvp),
                    (z_jvp, t_jvp),
                    (v_jvp, torch.ones_like(t_jvp)),
                )
            tdudt[keep_flat] = (t_jvp.clamp_min(self.t_eps) * du_dt).to(tdudt.dtype)
        tdudt_rms_raw = tdudt.float().pow(2).flatten(1).mean(dim=1).sqrt()
        if dudt_clip_norm > 0.0:
            tdudt_norm = tdudt.flatten(1).norm(dim=1).view(-1, *([1] * (z.ndim - 1)))
            tdudt_scale = (dudt_clip_norm / tdudt_norm.clamp_min(1e-6)).clamp(max=1.0)
            tdudt = tdudt * tdudt_scale
        tdudt_rms = tdudt.float().pow(2).flatten(1).mean(dim=1).sqrt()
        u_tgt = v - dudt_weight * tdudt

        loss_mf_per_sample = ((u - u_tgt) ** 2).mean(dim=(1, 2, 3))
        loss_mf_raw = loss_mf_per_sample.mean()
        loss_mf_raw_max = loss_mf_per_sample.max()
        if loss_clip > 0.0:
            loss_mf_per_sample = loss_mf_per_sample.clamp(max=loss_clip)
        loss_mf = loss_mf_per_sample.mean()
        loss = loss_mf
        loss_dict = {
            "loss_mf": loss_mf.detach(),
            "loss_mf_raw": loss_mf_raw.detach(),
            "loss_mf_raw_max": loss_mf_raw_max.detach(),
            "tdudt_rms": tdudt_rms.detach().mean(),
            "tdudt_rms_raw": tdudt_rms_raw.detach().mean(),
            "tdudt_rms_max": tdudt_rms.detach().max(),
            "dudt_keep_frac": keep.detach().float().mean(),
            "t_mean": t_img.detach().float().mean(),
        }
        return self._format_outputs(
            loss, loss_dict, x_pred, z, t_img, return_x_pred, return_t
        )

    def _scm_consistency_output(self, x_t, t, labels, cfg=1.0):
        f_cond = self.net(x_t / self.sigma_data, t.flatten(), labels)
        if cfg != 1.0:
            null_labels = torch.full_like(labels, self.num_classes)
            f_uncond = self.net(x_t / self.sigma_data, t.flatten(), null_labels)
            f_value = f_uncond + cfg * (f_cond - f_uncond)
        else:
            f_value = f_cond
        return torch.cos(t) * x_t - torch.sin(t) * self.sigma_data * f_value

    def sample_images_with_grad(self, x, y, sampling_args=None):
        if self.objective != "scm":
            return super().sample_images_with_grad(x, y, sampling_args=sampling_args)
        sampling_args = sampling_args or {}
        t_start = sampling_args.get("t_start")
        if t_start is None:
            scalar_t = torch.atan(x.new_tensor(self.sigma_max / self.sigma_data))
            t = scalar_t.expand(x.size(0)).view(-1, 1, 1, 1)
        else:
            t = t_start.to(device=x.device, dtype=x.dtype).view(-1, 1, 1, 1)
        output = self._scm_consistency_output(x, t, y, cfg=1.0)
        if sampling_args.get("return_velocity", False):
            return output, None
        return output

    @torch.inference_mode()
    def generate(self, n_samples, labels, cfg=1.0, args=None, verbose=True, z_t=None):
        if self.objective != "scm":
            return super().generate(
                n_samples, labels, cfg=cfg, args=args, verbose=verbose, z_t=z_t
            )

        if z_t is None:
            x_t = torch.randn(
                n_samples, 3, self.img_size, self.img_size, device=labels.device
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
            raise ValueError("sCM sampling currently supports exactly 1 or 2 steps")

        t_mid_value = float(getattr(args, "scm_intermediate_t", 1.1))
        t_mid = x.new_full((n_samples, 1, 1, 1), t_mid_value)
        noise = torch.randn_like(x) * self.sigma_data
        x_mid = torch.cos(t_mid) * x + torch.sin(t_mid) * noise
        return self._scm_consistency_output(x_mid, t_mid, labels, cfg=cfg)

    @staticmethod
    def _format_outputs(loss, loss_dict, x_pred, z, t_img, return_x_pred, return_t):
        if return_x_pred and return_t:
            return loss, loss_dict, x_pred, z, t_img
        if return_x_pred:
            return loss, loss_dict, x_pred, z
        if return_t:
            return loss, loss_dict, t_img
        return loss, loss_dict
