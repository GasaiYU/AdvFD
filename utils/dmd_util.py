import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger("FD_loss")


def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(requires_grad)


class DMDGuidance(nn.Module):
    """DMD one-step distribution matching guidance.

    This adapts the DMD2 ImageNet training structure to the FD-Loss denoisers:
    a frozen real score model, a trainable fake score model, a generator-side
    distribution matching pseudo-loss, and a fake-score training loss. The fake
    score loss can be the original DMD regression loss or an ASD-style
    score-space discriminator objective.
    """

    def __init__(self, base_model: nn.Module, args):
        super().__init__()
        self.args = args
        self.real_score_model = copy.deepcopy(base_model).cuda().eval()
        self.fake_score_model = copy.deepcopy(base_model).cuda().train()
        _set_requires_grad(self.real_score_model, False)
        _set_requires_grad(self.fake_score_model, True)

    def train(self, mode: bool = True):
        super().train(mode)
        self.real_score_model.eval()
        self.fake_score_model.train(mode)
        return self

    def load_real_score_state_dict(self, state_dict, strict: bool = False):
        msg = self.real_score_model.load_state_dict(state_dict, strict=strict)
        logger.info(f"[DMD] Loaded real score model: {msg}")
        return msg

    def state_dict_for_checkpoint(self):
        return {"fake_score_model": self.fake_score_model.state_dict()}

    def load_state_dict_from_checkpoint(self, state):
        if not state:
            return
        if "fake_score_model" in state:
            msg = self.fake_score_model.load_state_dict(state["fake_score_model"], strict=True)
            logger.info(f"[DMD] Restored fake score model: {msg}")

    def sample_t(self, batch_size: int, device, dtype, *, fake_score: bool = False):
        min_t = self.args.dmd_fake_min_t if fake_score else self.args.dmd_min_t
        max_t = self.args.dmd_fake_max_t if fake_score else self.args.dmd_max_t
        return self.sample_t_range(batch_size, device, dtype, min_t=min_t, max_t=max_t)

    def sample_t_range(self, batch_size: int, device, dtype, *, min_t: float, max_t: float):
        if getattr(self.args, "dmd_timestep_logit_normal", False):
            t = self.real_score_model.sample_t(batch_size, device=device).to(dtype=dtype)
        else:
            t = torch.empty(batch_size, device=device, dtype=dtype).uniform_(min_t, max_t)
        return t.clamp(min_t, max_t)

    def _renoise(self, x0, t):
        t_view = t.view(x0.shape[0], 1, 1, 1)
        noise = torch.randn_like(x0) * self.args.noise_scale
        noisy = (1.0 - t_view) * x0 + t_view * noise
        return noisy

    def _predict_x0(self, model: nn.Module, noisy, t, labels, guidance_scale: float):
        """Predict clean x0 from a noisy point at interpolation timestep t."""
        bsz = noisy.shape[0]
        t_vec = t.view(bsz)
        t_view = t.view(bsz, 1, 1, 1)

        if hasattr(model, "_forward_with_cfg"):
            u = model._forward_with_cfg(noisy, t_view, labels, cfg=guidance_scale)
            return noisy - t_view * u

        if hasattr(model, "u_fn"):
            omega = torch.full(
                (bsz,),
                guidance_scale,
                device=noisy.device,
                dtype=noisy.dtype,
            )
            t_min = torch.full((bsz,), self.args.interval_min, device=noisy.device, dtype=noisy.dtype)
            t_max = torch.full((bsz,), self.args.interval_max, device=noisy.device, dtype=noisy.dtype)
            u = model.u_fn(noisy, t_vec, t_view, omega, t_min, t_max, y=labels)[0]
            return noisy - t_view * u

        raise TypeError(f"DMDGuidance does not know how to predict x0 for {type(model)}")

    def _compute_original_distribution_matching_grad(self, x0, labels):
        batch_size = x0.shape[0]
        t = self.sample_t(batch_size, x0.device, x0.dtype, fake_score=False)
        noisy = self._renoise(x0, t)

        dm_grad_mode = getattr(self.args, "dmd_dm_grad_mode", "original")
        if dm_grad_mode == "uncond_real_minus_fake":
            pred_real_uncond_x0 = self._predict_x0(
                self.real_score_model,
                noisy,
                t,
                labels,
                guidance_scale=0.0,
            )
            pred_fake_x0 = self._predict_x0(
                self.fake_score_model,
                noisy,
                t,
                labels,
                guidance_scale=self.args.dmd_fake_guidance_scale,
            )

            p_real_uncond = x0 - pred_real_uncond_x0
            p_fake = x0 - pred_fake_x0
            weight_factor = p_real_uncond.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(
                self.args.dmd_grad_norm_eps
            )
            grad = (p_real_uncond - p_fake) / weight_factor
            return grad, {
                "dmd_t_mean": float(t.mean().detach()),
                "dmd_uncond_real_norm": float(p_real_uncond.norm().detach()),
                "dmd_fake_score_norm": float(p_fake.norm().detach()),
            }
        if dm_grad_mode != "original":
            raise ValueError(f"Unknown dmd_dm_grad_mode={dm_grad_mode}")

        pred_real_x0 = self._predict_x0(
            self.real_score_model,
            noisy,
            t,
            labels,
            guidance_scale=self.args.dmd_real_guidance_scale,
        )
        pred_fake_x0 = self._predict_x0(
            self.fake_score_model,
            noisy,
            t,
            labels,
            guidance_scale=self.args.dmd_fake_guidance_scale,
        )

        p_real = x0 - pred_real_x0
        p_fake = x0 - pred_fake_x0
        weight_factor = p_real.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(
            self.args.dmd_grad_norm_eps
        )
        grad = (p_real - p_fake) / weight_factor
        return grad, {
            "dmd_t_mean": float(t.mean().detach()),
        }

    def _compute_decoupled_distribution_matching_grad(self, x0, labels):
        batch_size = x0.shape[0]
        t_ca = self.sample_t_range(
            batch_size, x0.device, x0.dtype,
            min_t=self.args.dmd_ca_min_t,
            max_t=self.args.dmd_ca_max_t,
        )
        t_dm = self.sample_t_range(
            batch_size, x0.device, x0.dtype,
            min_t=self.args.dmd_dm_min_t,
            max_t=self.args.dmd_dm_max_t,
        )
        noisy_ca = self._renoise(x0, t_ca)
        noisy_dm = self._renoise(x0, t_dm)

        pred_real_cond_ca = self._predict_x0(
            self.real_score_model,
            noisy_ca,
            t_ca,
            labels,
            guidance_scale=1.0,
        )
        pred_real_cond_dm = self._predict_x0(
            self.real_score_model,
            noisy_dm,
            t_dm,
            labels,
            guidance_scale=1.0,
        )
        pred_fake_dm = self._predict_x0(
            self.fake_score_model,
            noisy_dm,
            t_dm,
            labels,
            guidance_scale=self.args.dmd_fake_guidance_scale,
        )

        p_real_cond_ca = x0 - pred_real_cond_ca
        p_real_cond_dm = x0 - pred_real_cond_dm
        p_fake_dm = x0 - pred_fake_dm

        if hasattr(self.real_score_model, "_forward_with_cfg"):
            pred_real_uncond_ca = self._predict_x0(
                self.real_score_model,
                noisy_ca,
                t_ca,
                labels,
                guidance_scale=0.0,
            )
            p_real_uncond_ca = x0 - pred_real_uncond_ca
            grad_ca_raw = (
                self.args.dmd_real_guidance_scale - 1.0
            ) * (p_real_cond_ca - p_real_uncond_ca)
        else:
            pred_real_cfg_ca = self._predict_x0(
                self.real_score_model,
                noisy_ca,
                t_ca,
                labels,
                guidance_scale=self.args.dmd_real_guidance_scale,
            )
            p_real_cfg_ca = x0 - pred_real_cfg_ca
            grad_ca_raw = p_real_cfg_ca - p_real_cond_ca
        grad_dm_raw = p_real_cond_dm - p_fake_dm
        ca_weight_factor = p_real_cond_ca.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(
            self.args.dmd_grad_norm_eps
        )
        dm_weight_factor = p_real_cond_dm.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(
            self.args.dmd_grad_norm_eps
        )
        grad_ca = grad_ca_raw / ca_weight_factor
        grad_dm = grad_dm_raw / dm_weight_factor
        grad = self.args.dmd_ca_weight * grad_ca + self.args.dmd_dm_weight * grad_dm
        return grad, {
            "dmd_ca_t_mean": float(t_ca.mean().detach()),
            "dmd_dm_t_mean": float(t_dm.mean().detach()),
            "dmd_ca_grad_norm": float(grad_ca.norm().detach()),
            "dmd_dm_grad_norm": float(grad_dm.norm().detach()),
        }

    def compute_distribution_matching_loss(self, x0, labels):
        original_x0 = x0

        with torch.no_grad():
            if getattr(self.args, "dmd_decoupled", False):
                grad, stats = self._compute_decoupled_distribution_matching_grad(x0, labels)
            else:
                grad, stats = self._compute_original_distribution_matching_grad(x0, labels)
            grad = torch.nan_to_num(grad)
            if self.args.dmd_grad_clip > 0:
                grad = grad.clamp(-self.args.dmd_grad_clip, self.args.dmd_grad_clip)

        target = (original_x0 - grad).detach()
        loss = 0.5 * F.mse_loss(original_x0, target, reduction="mean")
        stats.update({
            "dmd_loss": float(loss.detach()),
            "dmd_grad_norm": float(grad.norm().detach()),
        })
        return loss, stats

    def compute_fake_score_loss(self, x0, labels):
        x0 = x0.detach()
        batch_size = x0.shape[0]
        t = self.sample_t(batch_size, x0.device, x0.dtype, fake_score=True)
        t_view = t.view(batch_size, 1, 1, 1)
        noise = torch.randn_like(x0) * self.args.noise_scale
        noisy = (1.0 - t_view) * x0 + t_view * noise

        pred_fake_x0 = self._predict_x0(
            self.fake_score_model,
            noisy,
            t,
            labels,
            guidance_scale=self.args.dmd_fake_guidance_scale,
        )

        fake_prediction_type = getattr(self.args, "dmd_fake_prediction_type", "x0")
        if fake_prediction_type == "v":
            t_safe = t_view.clamp_min(self.args.dmd_grad_norm_eps)
            pred_fake = (noisy - pred_fake_x0) / t_safe
            target = (noisy - x0) / t_safe
        elif fake_prediction_type == "x0":
            pred_fake = pred_fake_x0
            target = x0
        else:
            raise ValueError(f"Unknown dmd_fake_prediction_type={fake_prediction_type}")

        loss_per_pixel = (pred_fake - target).pow(2)
        if self.args.dmd_fake_loss_snr_weight:
            snr = ((1.0 - t_view) / t_view.clamp_min(self.args.dmd_min_t)).pow(2)
            loss_per_pixel = loss_per_pixel * (snr + 1.0)

        score_loss = loss_per_pixel.mean()
        if getattr(self.args, "dmd_fake_loss_type", "dmd") == "asd":
            with torch.no_grad():
                pred_teacher_x0 = self._predict_x0(
                    self.real_score_model,
                    noisy,
                    t,
                    labels,
                    guidance_scale=1.0,
                )
                if fake_prediction_type == "v":
                    pred_teacher = (noisy - pred_teacher_x0) / t_safe
                else:
                    pred_teacher = pred_teacher_x0
            adv_loss = F.mse_loss(
                pred_fake.float(),
                pred_teacher.float(),
                reduction="mean",
            )
            loss = score_loss + self.args.dmd_asd_gamma * adv_loss
            return loss, {
                "dmd_fake_loss": float(loss.detach()),
                "dmd_fake_t_mean": float(t.mean().detach()),
                "dmd_asd_score_loss": float(score_loss.detach()),
                "dmd_asd_adv_loss": float(adv_loss.detach()),
                "dmd_asd_gamma": float(self.args.dmd_asd_gamma),
            }

        loss = score_loss
        return loss, {
            "dmd_fake_loss": float(loss.detach()),
            "dmd_fake_t_mean": float(t.mean().detach()),
        }
