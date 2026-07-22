#!/usr/bin/env python3
"""Train JiT-B sCM with a hard-boundary residual-x parameterization.

The raw JiT head predicts a normalized residual ``G_theta`` and the public
consistency endpoint is

    f_theta(x_t, t) = x_t + sigma_data * sin(t) * G_theta(x_t / sigma_data, t).

Consequently ``f_theta(x, 0) == x`` for every network parameter value.  No
boundary loss is needed, and an input-independent endpoint is not a regular
solution of the parameterized model.

This is an affine, nonsingular reparameterization of TrigFlow F-prediction:

    G = -F - tan(t / 2) * (x_t / sigma_data).

Training has two phases.  A legacy JiT x-prediction checkpoint is first
adapted to the exact TrigFlow velocity objective using

    sigma_data * F = -sigma_data * G - tan(t / 2) * x_t.

Adam and EMA are then reset before sCM training.  In G-space the exact sCM
targets are

    local = cos(t)^2 * (v_ref + sigma_data * G^- + tan(t/2) * x_t)
    full  = cos(t) * v_ref + sigma_data * cos(t)^2 * G^-
            + sigma_data * cos(t) * sin(t) * dG^-/dt
    R_G   = G - G^- + normalize((1-r) * local + r * full).

The G residual differs from the paper's F residual only by a sign, so the
adaptive loss needs no sin-dependent rescaling.
"""

import contextlib
import math
import sys
from pathlib import Path

import typing_extensions

# PyTorch 2.4 imports this metadata-only decorator, while the project image
# may contain an older typing_extensions package.
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


class SCMJiTResidualXDenoiser(MeanFlowJiTDenoiser):
    """TrigFlow sCM with raw G output and a hard clean-endpoint boundary."""

    def __init__(self, *args, g_adapt_steps=0, **kwargs):
        super().__init__(*args, **kwargs)
        if self.objective != "scm":
            raise ValueError(
                "SCMJiTResidualXDenoiser only supports objective='scm'"
            )
        self.g_adapt_steps = int(g_adapt_steps)
        if self.g_adapt_steps < 0:
            raise ValueError("g_adapt_steps must be non-negative")

    @staticmethod
    def _reshape_time(t, x):
        return t.view(-1, *([1] * (x.ndim - 1)))

    @staticmethod
    def _half_angle_sin_over_one_plus_cos(sin_t, cos_t):
        # Equal to tan(t/2).  The denominator lies in [1, 2] for TrigFlow's
        # t in [0, pi/2], so this form is finite and reuses the same trig values.
        return sin_t / (1.0 + cos_t)

    def _sample_trigflow_time(self, x):
        tau = torch.randn(x.size(0), device=x.device) * self.P_std + self.P_mean
        log_sigma_data = tau.new_tensor(self.sigma_data).log()
        log_ratio = tau - log_sigma_data
        acute_t = torch.atan(torch.exp(-log_ratio.abs()))
        return torch.where(
            log_ratio >= 0.0,
            0.5 * torch.pi - acute_t,
            acute_t,
        )

    @staticmethod
    def _batch_output_std(value):
        value = value.detach().float()
        batch_mean = value.mean(dim=0)
        batch_var = (
            value.square().mean() - batch_mean.square().mean()
        ).clamp_min(0.0)
        return batch_var.sqrt()

    def _endpoint_from_g(self, x_t, sin_t, g_value):
        return x_t + self.sigma_data * sin_t * g_value

    def forward(self, x, y, return_x_pred=False, return_t=False, **kwargs):
        global_step = int(kwargs.get("global_step", 0))
        labels = self.drop_labels(y) if self.training else y

        if global_step < self.g_adapt_steps:
            return self._forward_g_adaptation(
                x=x,
                labels=labels,
                return_x_pred=return_x_pred,
                return_t=return_t,
            )

        return self._forward_scm_residual_x(
            x=x,
            labels=labels,
            global_step=global_step - self.g_adapt_steps,
            return_x_pred=return_x_pred,
            return_t=return_t,
        )

    def _forward_g_adaptation(
        self,
        x,
        labels,
        return_x_pred,
        return_t,
    ):
        """Exact TrigFlow velocity objective expressed through raw G."""
        t = self._sample_trigflow_time(x)
        t_img = self._reshape_time(t, x)
        cos_t = torch.cos(t_img)
        sin_t = torch.sin(t_img)
        alpha = self._half_angle_sin_over_one_plus_cos(sin_t, cos_t)

        noise = torch.randn_like(x) * self.sigma_data
        x_t = cos_t * x + sin_t * noise
        velocity_target = cos_t * noise - sin_t * x

        g_pred = self.net(x_t / self.sigma_data, t, labels)
        # F = -G-alpha*u, hence sigma*F = -sigma*G-alpha*x_t.
        velocity_pred = (
            -self.sigma_data * g_pred.float()
            - alpha.float() * x_t.float()
        )
        loss_per_sample = (
            (velocity_pred - velocity_target.float())
            .square()
            .flatten(1)
            .mean(dim=1)
        )
        loss_g_adapt = loss_per_sample.mean()
        loss = loss_g_adapt

        # The adaptive time-weight network belongs only to the following sCM
        # phase.  A zero dependency keeps DDP from reporting unused parameters.
        if self.loss_weight_net is not None:
            loss = loss + sum(
                parameter.float().sum() * 0.0
                for parameter in self.loss_weight_net.parameters()
            )

        x_pred = self._endpoint_from_g(x_t, sin_t, g_pred)
        x_pred_float = x_pred.detach().float()
        g_target = (
            alpha.float() * x.float() / self.sigma_data
            - noise.float() / self.sigma_data
        )
        x0_mse = (
            (x_pred_float - x.float()).square().flatten(1).mean(dim=1)
        )
        loss_dict = {
            "loss_g_adapt": loss_g_adapt.detach(),
            "loss_g_adapt_max": loss_per_sample.detach().max(),
            "g_adapt_x0_mse": x0_mse.detach().mean(),
            "g_pred_rms": g_pred.detach().float().square().mean().sqrt(),
            "g_target_rms": g_target.detach().square().mean().sqrt(),
            "g_target_mse": (
                (g_pred.detach().float() - g_target.detach())
                .square()
                .flatten(1)
                .mean(dim=1)
                .mean()
            ),
            "velocity_pred_rms": velocity_pred.detach().square().mean().sqrt(),
            "velocity_target_rms": (
                velocity_target.detach().float().square().mean().sqrt()
            ),
            "x_pred_mean": x_pred_float.mean(),
            "x_pred_rms": x_pred_float.square().mean().sqrt(),
            "x_pred_abs_gt_1_frac": (
                (x_pred_float.abs() > 1.0).float().mean()
            ),
            "x_pred_batch_std": self._batch_output_std(x_pred_float),
            "alpha_mean": alpha.detach().float().mean(),
            "t_mean": t_img.detach().float().mean(),
            "phase_g_adapt": loss.detach().new_ones(()),
        }
        return self._format_outputs(
            loss,
            loss_dict,
            x_pred,
            x_t,
            t_img,
            return_x_pred,
            return_t,
        )

    def _forward_scm_residual_x(
        self,
        x,
        labels,
        global_step,
        return_x_pred,
        return_t,
    ):
        """Exact F-space sCM objective after the affine F-to-G transform."""
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
        cos_t = torch.cos(t_img)
        sin_t = torch.sin(t_img)
        cos_sin = cos_t * sin_t
        alpha = self._half_angle_sin_over_one_plus_cos(sin_t, cos_t)

        noise = torch.randn_like(x) * self.sigma_data
        x_t = cos_t * x + sin_t * noise
        velocity_ref = cos_t * noise - sin_t * x
        network_input = x_t / self.sigma_data

        # The trainable primal must retain reverse-mode parameter gradients.
        # The no-grad JVP below replays exactly the same dropout realization.
        if x.is_cuda:
            dropout_rng_state = torch.cuda.get_rng_state(x.device)
        else:
            dropout_rng_state = torch.random.get_rng_state()
        with _jvp_sdpa_context(x):
            g_pred = self.net(network_input, t, labels)

        x_jvp = network_input.detach().float()
        t_jvp = t_img.detach().float()
        # q_G = sigma*c*s*dG/dt.  Since du/dt=v_ref/sigma, moving the
        # complete scale inside JVP gives tangents (c*s*v_ref, sigma*c*s).
        dx_jvp = (cos_sin * velocity_ref).detach().float()
        dt_jvp = (self.sigma_data * cos_sin).detach().float()

        def g_fn(x_in, t_in):
            return self.net(x_in, t_in.flatten(), labels)

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
            jvp_primal, scaled_dg_dt = torch.func.jvp(
                g_fn,
                (x_jvp, t_jvp),
                (dx_jvp, dt_jvp),
            )

        g_teacher = g_pred.detach().float()
        scaled_dg_dt = scaled_dg_dt.detach().float()
        cos_float = cos_t.detach().float()
        alpha_float = alpha.detach().float()
        x_t_float = x_t.detach().float()
        velocity_float = velocity_ref.detach().float()

        local_target = cos_float.square() * (
            velocity_float
            + self.sigma_data * g_teacher
            + alpha_float * x_t_float
        )
        full_target = (
            cos_float * velocity_float
            + self.sigma_data * cos_float.square() * g_teacher
            + scaled_dg_dt
        )
        tangent_raw = (
            (1.0 - warmup) * local_target + warmup * full_target
        )

        tangent_raw_norm = tangent_raw.flatten(1).norm(dim=1)
        tangent_denom = (
            tangent_raw_norm + self.tangent_norm_c
        ).clamp_min(torch.finfo(torch.float32).tiny)
        tangent = tangent_raw / tangent_denom.view(
            -1, *([1] * (g_pred.ndim - 1))
        )

        # F-F^- = -(G-G^-), hence the F-space minus sign becomes plus here.
        residual = g_pred.float() - g_teacher + tangent.detach()
        loss_per_sample = residual.square().flatten(1).mean(dim=1)
        loss_scm_mse = loss_per_sample.mean()

        if self.loss_weight_net is not None:
            with _no_autocast_context(x):
                log_weight = self.loss_weight_net(t_img.detach())
                log_weight = log_weight.clamp(
                    min=-self.adaptive_weight_max,
                    max=self.adaptive_weight_max,
                )
                weighted_per_sample = (
                    log_weight.exp() * loss_per_sample - log_weight
                )
            loss = weighted_per_sample.mean()
            mean_log_weight = log_weight.detach().mean()
        else:
            loss = loss_scm_mse
            mean_log_weight = loss.new_zeros(())

        x_pred = self._endpoint_from_g(x_t, sin_t, g_pred)
        x_pred_float = x_pred.detach().float()
        tangent_norm = tangent.flatten(1).norm(dim=1)
        correction = self.sigma_data * sin_t.float() * g_pred.detach().float()
        loss_dict = {
            "loss_scm": loss.detach(),
            "loss_scm_mse": loss_scm_mse.detach(),
            "g_pred_rms": g_teacher.square().mean().sqrt(),
            "x_pred_mean": x_pred_float.mean(),
            "x_pred_rms": x_pred_float.square().mean().sqrt(),
            "x_pred_abs_gt_1_frac": (
                (x_pred_float.abs() > 1.0).float().mean()
            ),
            "x_pred_batch_std": self._batch_output_std(x_pred_float),
            "endpoint_correction_rms": correction.square().mean().sqrt(),
            "tangent_local_norm": (
                local_target.flatten(1).norm(dim=1).mean()
            ),
            "tangent_full_norm": (
                full_target.flatten(1).norm(dim=1).mean()
            ),
            "tangent_raw_norm": tangent_raw_norm.detach().mean(),
            "tangent_raw_norm_max": tangent_raw_norm.detach().max(),
            "tangent_norm": tangent_norm.detach().mean(),
            "tangent_warmup": loss.new_tensor(warmup).detach(),
            "adaptive_log_weight": mean_log_weight,
            "jvp_primal_error_max": (
                jvp_primal.detach().float() - g_teacher
            ).abs().max(),
            "alpha_mean": alpha_float.mean(),
            "t_mean": t_img.detach().float().mean(),
            "phase_scm": loss.detach().new_ones(()),
        }
        return self._format_outputs(
            loss,
            loss_dict,
            x_pred,
            x_t,
            t_img,
            return_x_pred,
            return_t,
        )

    def _scm_consistency_output(self, x_t, t, labels, cfg=1.0):
        """Decode raw G into the hard-boundary clean endpoint."""
        g_cond = self.net(x_t / self.sigma_data, t.flatten(), labels)
        if cfg == 1.0:
            g_value = g_cond
        else:
            null_labels = torch.full_like(labels, self.num_classes)
            g_uncond = self.net(
                x_t / self.sigma_data,
                t.flatten(),
                null_labels,
            )
            g_value = g_uncond + cfg * (g_cond - g_uncond)
        return self._endpoint_from_g(x_t, torch.sin(t), g_value)

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
            return output, None
        return output

    @torch.inference_mode()
    def generate(self, n_samples, labels, cfg=1.0, args=None, verbose=True, z_t=None):
        del verbose
        same_noise = bool(getattr(args, "same_noise", False))
        if z_t is None:
            noise_batch = 1 if same_noise else n_samples
            x_t = torch.randn(
                noise_batch,
                3,
                self.img_size,
                self.img_size,
                device=labels.device,
            ) * self.sigma_data
            if same_noise:
                x_t = x_t.repeat(n_samples, 1, 1, 1)
        else:
            x_t = z_t

        t_max = torch.atan(x_t.new_tensor(self.sigma_max / self.sigma_data))
        t = t_max.expand(n_samples).view(-1, 1, 1, 1)
        x = self._scm_consistency_output(x_t, t, labels, cfg=cfg)

        num_steps = int(getattr(args, "num_sampling_steps", 1))
        if num_steps == 1:
            return x
        if num_steps != 2:
            raise ValueError(
                "residual-x sCM sampling currently supports 1 or 2 steps"
            )

        t_mid_value = float(getattr(args, "scm_intermediate_t", 1.1))
        t_mid = x.new_full((n_samples, 1, 1, 1), t_mid_value)
        noise_batch = 1 if same_noise else n_samples
        noise = torch.randn(
            noise_batch,
            *x.shape[1:],
            device=x.device,
            dtype=x.dtype,
        ) * self.sigma_data
        if same_noise:
            noise = noise.repeat(n_samples, 1, 1, 1)
        x_mid = torch.cos(t_mid) * x + torch.sin(t_mid) * noise
        return self._scm_consistency_output(x_mid, t_mid, labels, cfg=cfg)


def create_model(args):
    trainer.logger.info(
        "two-phase hard-boundary residual-x schedule: init=%s G-adapt=%d "
        "steps at lr=%s, sCM=%d steps",
        args.init_parameterization,
        args.g_adapt_steps,
        args.g_adapt_lr,
        args.scm_steps,
    )
    return trainer.create_model(
        args,
        model_cls=SCMJiTResidualXDenoiser,
        g_adapt_steps=args.g_adapt_steps,
    )


def adjust_two_phase_lr(optimizer, step, args):
    if step < args.g_adapt_steps:
        base_lr = args.g_adapt_lr if args.g_adapt_lr is not None else args.lr
        if step < args.g_adapt_warmup_steps:
            lr = base_lr * float(step + 1) / max(1, args.g_adapt_warmup_steps)
        else:
            lr = base_lr
    else:
        scm_step = step - args.g_adapt_steps
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


def maybe_transition_to_scm(
    step,
    args,
    model,
    optimizer,
    ema_model,
    world_size,
):
    if args.g_adapt_steps <= 0 or step != args.g_adapt_steps:
        return optimizer, ema_model

    trainer.logger.info(
        "TrigFlow G adaptation complete at step=%d; resetting Adam and EMA "
        "for hard-boundary residual-x sCM",
        step,
    )
    return (
        trainer.create_optimizer(args, model, world_size),
        trainer.create_ema_model(args, model, world_size),
    )


def get_args_parser():
    parser = trainer.get_args_parser()
    parser.prog = "train_scm_jit_b_residual_x.py"
    parser.description = (
        "JiT-B TrigFlow G adaptation followed by hard-boundary residual-x sCM"
    )
    parser.set_defaults(
        objective="scm",
        checkpoint_key="model_ema2",
        exp_name="jit_b_scm_residual_x_hard_boundary",
        P_mean=-1.0,
        P_std=1.6,
        sigma_data=0.5,
        sigma_max=80.0,
        tangent_norm_c=0.1,
        tangent_warmup_steps=10000,
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
        choices=["x_pred", "trigflow_G"],
        help=(
            "semantic meaning of the checkpoint head; legacy x_pred requires "
            "G adaptation, while trigflow_G is already compatible"
        ),
    )
    parser.add_argument(
        "--g_adapt_steps",
        type=int,
        default=10000,
        help="exact TrigFlow velocity-adaptation steps for the raw G head",
    )
    parser.add_argument(
        "--g_adapt_lr",
        type=float,
        default=1e-4,
        help="learning rate for the supervised G-adaptation phase",
    )
    parser.add_argument(
        "--g_adapt_warmup_steps",
        type=int,
        default=1250,
        help="linear LR warmup within G adaptation",
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
        parser.error(
            "train_scm_jit_b_residual_x.py only supports --objective scm"
        )
    if args.g_adapt_steps < 0:
        parser.error("--g_adapt_steps must be non-negative")
    if args.g_adapt_warmup_steps < 0:
        parser.error("--g_adapt_warmup_steps must be non-negative")
    if args.g_adapt_lr is not None and args.g_adapt_lr <= 0.0:
        parser.error("--g_adapt_lr must be positive")
    if args.init_parameterization == "x_pred" and args.g_adapt_steps == 0:
        parser.error(
            "a legacy x_pred checkpoint requires --g_adapt_steps > 0; use "
            "--init_parameterization trigflow_G only for an adapted G checkpoint"
        )
    if (
        args.init_parameterization == "trigflow_G"
        and args.g_adapt_steps != 0
    ):
        parser.error("a trigflow_G checkpoint should use --g_adapt_steps 0")

    args.scm_steps = (
        args.epochs * args.steps_per_epoch
        if args.scm_steps is None
        else int(args.scm_steps)
    )
    if args.scm_steps <= 0:
        parser.error("--scm_steps must be positive")

    args.phase_adapt_steps = args.g_adapt_steps
    args.total_steps_override = args.g_adapt_steps + args.scm_steps
    args.training_entry = "train_scm_jit_b_residual_x"
    args.raw_output_parameterization = "TrigFlow_G"
    args.endpoint_parameterization = "x_t + sigma_data * sin(t) * G"
    args.boundary_condition = "hard_skip_sin"
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
