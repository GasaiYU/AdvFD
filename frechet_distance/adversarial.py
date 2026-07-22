"""Adversarial real-whitened FD helpers."""

import logging

import torch


logger = logging.getLogger("FD_loss")


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
