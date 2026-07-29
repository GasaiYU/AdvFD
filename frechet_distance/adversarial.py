"""Adversarial real-whitened FD helpers."""

import logging
import math

import torch


logger = logging.getLogger("FD_loss")


def hard_l2_feature_cap(
    features: torch.Tensor,
    max_norm: float,
) -> torch.Tensor:
    """Project each feature vector onto a closed L2 ball.

    ``max_norm <= 0`` disables the transform. Norms and projections are
    evaluated in FP32 for low-precision features so the hard bound is not
    weakened by FP16/BF16 scale rounding. The transform remains
    differentiable with respect to ``features``.
    """
    if not math.isfinite(max_norm):
        raise ValueError("FD-Adv feature norm cap must be finite")
    if max_norm <= 0.0:
        return features
    if features.ndim == 0:
        raise ValueError("FD-Adv features must have a feature dimension")

    norm_features = (
        features.float()
        if features.dtype in (torch.float16, torch.bfloat16)
        else features
    )
    norms = torch.linalg.vector_norm(norm_features, ord=2, dim=-1, keepdim=True)
    bound = norms.new_tensor(float(max_norm))
    scale = bound / norms.clamp_min(bound)
    return norm_features * scale


def shared_residual_rms_projection(
    adv_real: torch.Tensor,
    ref_real: torch.Tensor,
    adv_fake: torch.Tensor,
    ref_fake: torch.Tensor,
    tau: float,
    eps: float = 1e-8,
):
    """Smoothly bound the RMS residual of a 50:50 real/fake mixture.

    A single differentiable scalar is shared by real and fake features:

    ``scale = 1 / sqrt(1 + residual_rms2 / tau**2 + eps)``.

    The returned residual mixture therefore has second moment at most
    ``tau**2``.  Inputs are expected to have already been gathered across all
    distributed ranks so every rank uses the same global scale.
    """
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError("FD-Adv residual RMS tau must be finite and > 0")
    if not math.isfinite(eps) or eps < 0.0:
        raise ValueError("FD-Adv residual RMS eps must be finite and >= 0")
    pairs = (
        ("real", adv_real, ref_real),
        ("fake", adv_fake, ref_fake),
    )
    for split_name, adv, ref in pairs:
        if adv.ndim < 2:
            raise ValueError(f"FD-Adv {split_name} features must be at least 2D")
        if adv.shape != ref.shape:
            raise ValueError(
                f"FD-Adv {split_name} adv/reference feature shapes differ: "
                f"{tuple(adv.shape)} vs {tuple(ref.shape)}"
            )
        if adv.shape[0] == 0:
            raise ValueError(f"FD-Adv {split_name} feature batch must be non-empty")

    # Evaluate the trust-region scalar in FP32 for BF16/FP16 feature models.
    # Do not detach: the critic and generator must see the derivative of the
    # scale as well as the derivative of the residual itself.
    compute_dtype = (
        torch.float32
        if adv_real.dtype in (torch.float16, torch.bfloat16)
        else adv_real.dtype
    )
    real_delta = adv_real.to(compute_dtype) - ref_real.to(compute_dtype)
    fake_delta = adv_fake.to(compute_dtype) - ref_fake.to(compute_dtype)
    real_rms2 = real_delta.square().sum(dim=-1).mean()
    fake_rms2 = fake_delta.square().sum(dim=-1).mean()
    residual_rms2 = 0.5 * (real_rms2 + fake_rms2)
    tau2 = residual_rms2.new_tensor(float(tau) ** 2)
    raw_ratio = residual_rms2 / tau2
    scale = torch.rsqrt(1.0 + raw_ratio + float(eps))

    bounded_real = ref_real.to(compute_dtype) + scale * real_delta
    bounded_fake = ref_fake.to(compute_dtype) + scale * fake_delta
    bounded_ratio = scale.square() * raw_ratio
    metrics = {
        "scale": scale.detach(),
        "raw_rms_to_tau": raw_ratio.clamp_min(0.0).sqrt().detach(),
        "bounded_rms_to_tau": bounded_ratio.clamp_min(0.0).sqrt().detach(),
    }
    return bounded_real, bounded_fake, metrics


def shared_feature_norm_offset_penalty(
    adv_real: torch.Tensor,
    ref_real: torch.Tensor,
    adv_fake: torch.Tensor,
    ref_fake: torch.Tensor,
    split: bool = False,
):
    """Penalize drift in global real/fake feature second moments.

    With ``split=False`` the penalty is ``(S_adv / S_ref - 1)**2``, where each
    second moment is computed on a 50:50 real/fake mixture. With ``split=True``
    it is the average of the independently normalized real and fake penalties.
    Reference inputs are detached so gradients reach only the trainable
    representation. Inputs are expected to have already been gathered across
    distributed ranks.
    """
    pairs = (
        ("real", adv_real, ref_real),
        ("fake", adv_fake, ref_fake),
    )
    for split_name, adv, ref in pairs:
        if adv.ndim < 2:
            raise ValueError(f"FD-Adv {split_name} features must be at least 2D")
        if adv.shape != ref.shape:
            raise ValueError(
                f"FD-Adv {split_name} adv/reference feature shapes differ: "
                f"{tuple(adv.shape)} vs {tuple(ref.shape)}"
            )
        if adv.shape[0] == 0:
            raise ValueError(f"FD-Adv {split_name} feature batch must be non-empty")

    compute_dtype = (
        torch.float32
        if adv_real.dtype in (torch.float16, torch.bfloat16)
        else adv_real.dtype
    )
    adv_real_value = adv_real.to(compute_dtype)
    adv_fake_value = adv_fake.to(compute_dtype)
    ref_real_value = ref_real.detach().to(compute_dtype)
    ref_fake_value = ref_fake.detach().to(compute_dtype)

    adv_real_second_moment = adv_real_value.square().sum(dim=-1).mean()
    adv_fake_second_moment = adv_fake_value.square().sum(dim=-1).mean()
    ref_real_second_moment = ref_real_value.square().sum(dim=-1).mean()
    ref_fake_second_moment = ref_fake_value.square().sum(dim=-1).mean()
    for split_name, value in (
        ("real", ref_real_second_moment),
        ("fake", ref_fake_second_moment),
    ):
        if not bool(torch.isfinite(value).item()):
            raise ValueError(
                f"FD-Adv {split_name} reference feature second moment must be finite"
            )
        if float(value) <= 0.0:
            raise ValueError(
                f"FD-Adv {split_name} reference feature second moment must be > 0"
            )
    for split_name, value in (
        ("real", adv_real_second_moment),
        ("fake", adv_fake_second_moment),
    ):
        if not bool(torch.isfinite(value.detach()).item()):
            raise ValueError(
                f"FD-Adv {split_name} trainable feature second moment must be finite"
            )

    real_ratio = adv_real_second_moment / ref_real_second_moment
    fake_ratio = adv_fake_second_moment / ref_fake_second_moment
    adv_second_moment = 0.5 * (
        adv_real_second_moment + adv_fake_second_moment
    )
    ref_second_moment = 0.5 * (
        ref_real_second_moment + ref_fake_second_moment
    )
    if not bool(torch.isfinite(ref_second_moment).item()):
        raise ValueError("FD-Adv reference feature second moment must be finite")
    if not bool(torch.isfinite(adv_second_moment.detach()).item()):
        raise ValueError("FD-Adv trainable feature second moment must be finite")
    second_moment_ratio = adv_second_moment / ref_second_moment
    shared_penalty = (second_moment_ratio - 1.0).square()
    split_penalty = 0.5 * (
        (real_ratio - 1.0).square() + (fake_ratio - 1.0).square()
    )
    penalty = split_penalty if split else shared_penalty
    metrics = {
        "second_moment_ratio": second_moment_ratio.detach(),
        "real_second_moment_ratio": real_ratio.detach(),
        "fake_second_moment_ratio": fake_ratio.detach(),
        "penalty": penalty.detach(),
        "shared_penalty": shared_penalty.detach(),
        "split_penalty": split_penalty.detach(),
        "adv_rms": adv_second_moment.clamp_min(0.0).sqrt().detach(),
        "ref_rms": ref_second_moment.clamp_min(0.0).sqrt().detach(),
    }
    return penalty, metrics


class FeatureStatsEMA(torch.nn.Module):
    """EMA moments with differentiable current-batch contribution."""

    def __init__(self, feat_dim: int, beta: float):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.beta = float(beta)
        self.register_buffer("mu_ema", torch.zeros(feat_dim, dtype=torch.float64))
        self.register_buffer("m2_ema", torch.zeros(feat_dim, feat_dim, dtype=torch.float64))
        self.register_buffer("initialized", torch.zeros(1, dtype=torch.long))

    def build_stats(self, feats: torch.Tensor):
        feats_d = feats.double()
        batch_mu = feats_d.mean(dim=0)
        batch_m2 = feats_d.T @ feats_d / feats_d.shape[0]
        if int(self.initialized.item()) == 0:
            mu = batch_mu
            m2 = batch_m2
        else:
            mu = self.beta * self.mu_ema.detach() + (1.0 - self.beta) * batch_mu
            m2 = self.beta * self.m2_ema.detach() + (1.0 - self.beta) * batch_m2
        cov = m2 - mu.unsqueeze(1) * mu.unsqueeze(0)
        return mu, cov

    def current_stats(self):
        if int(self.initialized.item()) == 0:
            raise RuntimeError("FeatureStatsEMA is not initialized")
        mu = self.mu_ema.detach()
        cov = self.m2_ema.detach() - mu.unsqueeze(1) * mu.unsqueeze(0)
        cov = 0.5 * (cov + cov.T)
        return mu, cov

    @torch.no_grad()
    def update(self, feats: torch.Tensor):
        feats_d = feats.detach().double()
        batch_mu = feats_d.mean(dim=0)
        batch_m2 = feats_d.T @ feats_d / feats_d.shape[0]
        if int(self.initialized.item()) == 0:
            self.mu_ema.copy_(batch_mu)
            self.m2_ema.copy_(batch_m2)
            self.initialized.fill_(1)
            return
        self.mu_ema.mul_(self.beta).add_(batch_mu, alpha=1.0 - self.beta)
        self.m2_ema.mul_(self.beta).add_(batch_m2, alpha=1.0 - self.beta)

    @torch.no_grad()
    def initialize_from_mean_cov(self, mu: torch.Tensor, cov: torch.Tensor):
        mu_d = mu.detach().double()
        cov_d = cov.detach().double()
        self.mu_ema.copy_(mu_d)
        self.m2_ema.copy_(cov_d + mu_d.unsqueeze(1) * mu_d.unsqueeze(0))
        self.initialized.fill_(1)

    @torch.no_grad()
    def initialize_from_mean_m2(self, mu: torch.Tensor, m2: torch.Tensor):
        self.mu_ema.copy_(mu.detach().double())
        self.m2_ema.copy_(m2.detach().double())
        self.initialized.fill_(1)


def build_real_whitening(
    real_mu: torch.Tensor,
    real_cov: torch.Tensor,
    eps: float = 1e-1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a detached real whitening transform in eigenspace."""
    real_mu = real_mu.double()
    real_cov = real_cov.double()
    real_cov = 0.5 * (real_cov + real_cov.T)
    dim = real_mu.shape[0]

    eye = torch.eye(dim, device=real_mu.device, dtype=real_mu.dtype)
    real_cov_reg = real_cov + eps * eye
    eigvals, eigvecs = torch.linalg.eigh(real_cov_reg)
    inv_sqrt = eigvals.clamp_min(eps).rsqrt()
    return real_mu.detach(), eigvecs.detach(), inv_sqrt.detach()


def real_whitened_frechet_distance_from_stats(
    real_mu: torch.Tensor,
    real_cov: torch.Tensor,
    fake_mu: torch.Tensor,
    fake_cov: torch.Tensor,
    eps: float = 1e-1,
    real_whitening: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """FD of fake stats after whitening by real stats.

    The diagonal loading is part of the Gaussian being whitened, not only a
    numerical trick for the inverse square root.  Applying the same loading to
    fake covariance makes the loss exactly zero when fake stats match real
    stats, while still capping amplification in low-variance real directions.
    """
    fake_mu = fake_mu.double()
    fake_cov = fake_cov.double()
    fake_cov = 0.5 * (fake_cov + fake_cov.T)
    dim = fake_mu.shape[0]

    if real_whitening is None:
        real_whitening = build_real_whitening(real_mu, real_cov, eps=eps)
    real_mu, real_eigvecs, real_inv_sqrt = real_whitening

    fake_mu_white = ((fake_mu - real_mu) @ real_eigvecs) * real_inv_sqrt
    eye = torch.eye(dim, device=fake_mu.device, dtype=fake_mu.dtype)
    fake_cov_reg = fake_cov + eps * eye
    fake_cov_eig = real_eigvecs.T @ fake_cov_reg @ real_eigvecs
    fake_cov_white = fake_cov_eig * real_inv_sqrt[:, None] * real_inv_sqrt[None, :]
    fake_cov_white = 0.5 * (fake_cov_white + fake_cov_white.T)

    mean_term = fake_mu_white.dot(fake_mu_white)
    eigvals = torch.linalg.eigvalsh(fake_cov_white).clamp_min(0.0)
    trace_term = (
        torch.diagonal(fake_cov_white).sum()
        + float(dim)
        - 2.0 * torch.sqrt(eigvals).sum()
    )
    return (mean_term + trace_term).float()


def save_fd_adv_states(judges):
    states = []
    for judge in judges:
        model = judge.get("adv_model")
        optimizer = judge.get("adv_optimizer")
        if model is None or optimizer is None:
            continue
        state = {
            "name": judge["name"],
            "init_mode": judge.get("adv_init_mode", "pretrained"),
            "feature_norm_cap": float(judge.get("adv_feature_norm_cap", 0.0)),
            "residual_rms_kappa": float(judge.get("adv_residual_rms_kappa", 0.0)),
            "residual_rms_tau": float(judge.get("adv_residual_rms_tau", 0.0)),
            "norm_offset_weight": float(judge.get("adv_norm_offset_weight", 0.0)),
            "norm_offset_split": bool(judge.get("adv_norm_offset_split", False)),
            "feature_transform": _fd_adv_feature_transform_name(judge),
            "optimizer": optimizer.state_dict(),
        }
        state["model"] = model.state_dict()
        real_stats = judge.get("adv_real_stats")
        fake_stats = judge.get("adv_fake_stats")
        neg_stats = judge.get("adv_neg_stats")
        if real_stats is not None:
            state["real_stats"] = real_stats.state_dict()
        if fake_stats is not None:
            state["fake_stats"] = fake_stats.state_dict()
        if neg_stats is not None:
            state["neg_stats"] = neg_stats.state_dict()
        states.append(state)
    return states


def load_fd_adv_states(judges, saved_states):
    if not saved_states:
        return False
    name_to_state = {s["name"]: s for s in saved_states}
    loaded = 0
    total = 0
    for judge in judges:
        model = judge.get("adv_model")
        optimizer = judge.get("adv_optimizer")
        if model is None or optimizer is None:
            continue
        total += 1
        state = name_to_state.get(judge["name"])
        if state is None:
            logger.warning(f"[FD-Adv] No saved adversarial state for '{judge['name']}'")
            continue
        expected_init_mode = judge.get("adv_init_mode", "pretrained")
        saved_init_mode = state.get("init_mode", "pretrained")
        if saved_init_mode != expected_init_mode:
            raise RuntimeError(
                f"[FD-Adv] Cannot restore '{judge['name']}' with "
                f"init_mode={saved_init_mode!r}; current run expects "
                f"init_mode={expected_init_mode!r}. Use a separate experiment "
                "directory or remove the incompatible resume checkpoint."
            )
        expected_feature_norm_cap = float(judge.get("adv_feature_norm_cap", 0.0))
        saved_feature_norm_cap = float(state.get("feature_norm_cap", 0.0))
        if not math.isclose(
            saved_feature_norm_cap,
            expected_feature_norm_cap,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                f"[FD-Adv] Cannot restore '{judge['name']}' with "
                f"feature_norm_cap={saved_feature_norm_cap}; current run expects "
                f"{expected_feature_norm_cap}. Use a separate experiment "
                "directory or remove the incompatible resume checkpoint."
            )
        expected_residual_rms_kappa = float(judge.get("adv_residual_rms_kappa", 0.0))
        saved_residual_rms_kappa = float(state.get("residual_rms_kappa", 0.0))
        if not math.isclose(
            saved_residual_rms_kappa,
            expected_residual_rms_kappa,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                f"[FD-Adv] Cannot restore '{judge['name']}' with "
                f"residual_rms_kappa={saved_residual_rms_kappa}; current run "
                f"expects {expected_residual_rms_kappa}. Use a separate "
                "experiment directory or remove the incompatible resume checkpoint."
            )
        expected_norm_offset_split = bool(
            judge.get("adv_norm_offset_split", False)
        )
        saved_norm_offset_split = bool(state.get("norm_offset_split", False))
        if saved_norm_offset_split != expected_norm_offset_split:
            raise RuntimeError(
                f"[FD-Adv] Cannot restore '{judge['name']}' with "
                f"norm_offset_split={saved_norm_offset_split}; current run "
                f"expects {expected_norm_offset_split}. Use a separate "
                "experiment directory or remove the incompatible resume checkpoint."
            )
        expected_residual_rms_tau = float(judge.get("adv_residual_rms_tau", 0.0))
        saved_residual_rms_tau = float(state.get("residual_rms_tau", 0.0))
        if not math.isclose(
            saved_residual_rms_tau,
            expected_residual_rms_tau,
            rel_tol=1e-8,
            abs_tol=1e-10,
        ):
            raise RuntimeError(
                f"[FD-Adv] Cannot restore '{judge['name']}' with "
                f"residual_rms_tau={saved_residual_rms_tau}; current run "
                f"expects {expected_residual_rms_tau}."
            )
        expected_norm_offset_weight = float(
            judge.get("adv_norm_offset_weight", 0.0)
        )
        saved_norm_offset_weight = float(state.get("norm_offset_weight", 0.0))
        if not math.isclose(
            saved_norm_offset_weight,
            expected_norm_offset_weight,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                f"[FD-Adv] Cannot restore '{judge['name']}' with "
                f"norm_offset_weight={saved_norm_offset_weight}; current run "
                f"expects {expected_norm_offset_weight}. Use a separate "
                "experiment directory or remove the incompatible resume checkpoint."
            )
        expected_feature_transform = _fd_adv_feature_transform_name(judge)
        saved_feature_transform = state.get("feature_transform", "none")
        if saved_feature_transform != expected_feature_transform:
            raise RuntimeError(
                f"[FD-Adv] Cannot restore '{judge['name']}' with "
                f"feature_transform={saved_feature_transform!r}; current run expects "
                f"{expected_feature_transform!r}."
            )
        if "model" in state:
            try:
                model.load_state_dict(state["model"])
            except RuntimeError as exc:
                logger.warning(
                    f"[FD-Adv] Could not restore adversarial model for "
                    f"'{judge['name']}': {exc}"
                )
                continue
        real_stats = judge.get("adv_real_stats")
        fake_stats = judge.get("adv_fake_stats")
        neg_stats = judge.get("adv_neg_stats")
        if real_stats is not None and "real_stats" in state:
            real_stats.load_state_dict(state["real_stats"])
        if fake_stats is not None and "fake_stats" in state:
            fake_stats.load_state_dict(state["fake_stats"])
        if neg_stats is not None and "neg_stats" in state:
            neg_stats.load_state_dict(state["neg_stats"])
        elif neg_stats is not None:
            logger.warning(
                f"[FD-Adv] No saved mixed negative stats for '{judge['name']}'"
            )
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        loaded += 1
        logger.info(f"[FD-Adv] Restored adversarial state for '{judge['name']}'")
    return total > 0 and loaded == total


def _fd_adv_feature_transform_name(judge):
    if float(judge.get("adv_residual_rms_kappa", 0.0)) > 0.0:
        return "smooth_shared_residual_rms_v1"
    if float(judge.get("adv_feature_norm_cap", 0.0)) > 0.0:
        return "hard_l2_v1"
    return "none"
