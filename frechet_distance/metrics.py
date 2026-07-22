import numpy as np
import torch


def compute_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Frechet distance between two Gaussians (matches OpenAI guided-diffusion evaluator)."""
    from scipy import linalg

    mu1 = np.atleast_1d(np.asarray(mu1, dtype=np.float64))
    mu2 = np.atleast_1d(np.asarray(mu2, dtype=np.float64))
    sigma1 = np.atleast_2d(np.asarray(sigma1, dtype=np.float64))
    sigma2 = np.atleast_2d(np.asarray(sigma2, dtype=np.float64))

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def compute_isc(logits, splits=10, rng_seed=2020):
    """Inception score (matches torch_fidelity formula). Returns (mean, std)."""
    if logits.dim() != 2:
        raise ValueError(f"Expected 2D logits tensor, got {logits.dim()}D")
    N = logits.shape[0]

    rng = np.random.RandomState(rng_seed)
    logits = logits[rng.permutation(N)].double()
    p = logits.softmax(dim=1)
    log_p = logits.log_softmax(dim=1)

    scores = []
    for i in range(splits):
        lo = i * N // splits
        hi = (i + 1) * N // splits
        p_chunk = p[lo:hi]
        log_p_chunk = log_p[lo:hi]
        q = p_chunk.mean(dim=0, keepdim=True)
        kl = (p_chunk * (log_p_chunk - q.log())).sum(1).mean().exp().item()
        scores.append(kl)
    return float(np.mean(scores)), float(np.std(scores))


# =============================================================================
# Precision & Recall  (Kynkäänniemi et al., NeurIPS 2019)
# =============================================================================

@torch.inference_mode()
def _knn_radius(features: torch.Tensor, k: int, batch_size: int = 5000) -> torch.Tensor:
    """k-th nearest-neighbor distance for each row, excluding self."""
    N = features.shape[0]
    radii = torch.empty(N, device=features.device)
    for lo in range(0, N, batch_size):
        hi = min(lo + batch_size, N)
        dists = torch.cdist(features[lo:hi], features)  # (chunk, N)
        idx = torch.arange(hi - lo, device=features.device)
        dists[idx, idx + lo] = float("inf")  # exclude self
        kth, _ = dists.topk(k, dim=1, largest=False)
        radii[lo:hi] = kth[:, -1]
    return radii


@torch.inference_mode()
def _fraction_in_manifold(
    query: torch.Tensor,
    reference: torch.Tensor,
    ref_radii: torch.Tensor,
    batch_size: int = 5000,
) -> float:
    """Fraction of *query* points inside the manifold estimated from *reference*."""
    N = query.shape[0]
    count = 0
    for lo in range(0, N, batch_size):
        hi = min(lo + batch_size, N)
        dists = torch.cdist(query[lo:hi], reference)
        in_ball = (dists <= ref_radii.unsqueeze(0)).any(dim=1)
        count += in_ball.sum().item()
    return count / N


@torch.inference_mode()
def compute_precision_recall(
    real_features: torch.Tensor,
    gen_features: torch.Tensor,
    k: int = 3,
    batch_size: int = 5000,
) -> tuple[float, float]:
    """Improved Precision and Recall (Kynkäänniemi et al., NeurIPS 2019).

    Both tensors should be float32 on the same device, shapes ``(N, D)`` and
    ``(M, D)``.  Returns ``(precision, recall)``.
    """
    real_radii = _knn_radius(real_features, k, batch_size)
    gen_radii = _knn_radius(gen_features, k, batch_size)
    precision = _fraction_in_manifold(gen_features, real_features, real_radii, batch_size)
    recall = _fraction_in_manifold(real_features, gen_features, gen_radii, batch_size)
    return precision, recall


# =============================================================================
# CMMD — Maximum Mean Discrepancy with Gaussian RBF kernel
# (Jayasumana et al., "Rethinking FID", CVPR 2024)
# =============================================================================

@torch.inference_mode()
def _median_bandwidth(
    X: torch.Tensor, Y: torch.Tensor, max_subsample: int = 2000,
) -> float:
    """Estimate RBF bandwidth via median heuristic on a random subsample."""
    n = min(max_subsample, X.shape[0], Y.shape[0])
    idx_x = torch.randperm(X.shape[0], device=X.device)[:n]
    idx_y = torch.randperm(Y.shape[0], device=Y.device)[:n]
    combined = torch.cat([X[idx_x], Y[idx_y]], dim=0)
    # Pairwise squared distances on subsample
    dists_sq = torch.cdist(combined, combined).pow(2)
    mask = torch.triu(torch.ones(dists_sq.shape[0], dists_sq.shape[0],
                                  device=dists_sq.device, dtype=torch.bool),
                       diagonal=1)
    sigma = dists_sq[mask].median().sqrt().item()
    return max(sigma, 1e-5)  # guard against zero


@torch.inference_mode()
def compute_mmd(
    real_features: torch.Tensor,
    gen_features: torch.Tensor,
    *,
    sigma: float | None = None,
    batch_size: int = 2000,
    scale: float = 1000.0,
) -> float:
    """Maximum Mean Discrepancy with Gaussian RBF kernel (biased estimator).

    Both tensors should be float32 on the same device, shapes ``(N, D)`` and
    ``(M, D)``.  If *sigma* is None, bandwidth is set via the median heuristic.
    Returns ``scale * MMD^2`` (default scale=1000 matches the CMMD paper).
    """
    X, Y = real_features, gen_features
    if sigma is None:
        sigma = _median_bandwidth(X, Y)
    gamma = 1.0 / (2.0 * sigma * sigma)

    # k_xx
    x_sqnorms = (X * X).sum(1)  # (N,)
    k_xx_sum = 0.0
    for lo in range(0, X.shape[0], batch_size):
        hi = min(lo + batch_size, X.shape[0])
        d2 = x_sqnorms[lo:hi, None] + x_sqnorms[None, :] - 2.0 * X[lo:hi] @ X.T
        k_xx_sum += torch.exp(-gamma * d2.clamp(min=0)).sum().item()
    k_xx = k_xx_sum / (X.shape[0] * X.shape[0])

    # k_yy
    y_sqnorms = (Y * Y).sum(1)  # (M,)
    k_yy_sum = 0.0
    for lo in range(0, Y.shape[0], batch_size):
        hi = min(lo + batch_size, Y.shape[0])
        d2 = y_sqnorms[lo:hi, None] + y_sqnorms[None, :] - 2.0 * Y[lo:hi] @ Y.T
        k_yy_sum += torch.exp(-gamma * d2.clamp(min=0)).sum().item()
    k_yy = k_yy_sum / (Y.shape[0] * Y.shape[0])

    # k_xy
    k_xy_sum = 0.0
    for lo in range(0, X.shape[0], batch_size):
        hi = min(lo + batch_size, X.shape[0])
        d2 = x_sqnorms[lo:hi, None] + y_sqnorms[None, :] - 2.0 * X[lo:hi] @ Y.T
        k_xy_sum += torch.exp(-gamma * d2.clamp(min=0)).sum().item()
    k_xy = k_xy_sum / (X.shape[0] * Y.shape[0])

    mmd2 = k_xx + k_yy - 2.0 * k_xy
    return scale * mmd2


# =============================================================================
# MIND — Monge Inception Distance
# (Berthet et al., "MIND", arXiv:2605.06797)
# =============================================================================

@torch.inference_mode()
def compute_mind(
    real_features: torch.Tensor,
    gen_features: torch.Tensor,
    *,
    num_projections: int = 1000,
    max_samples: int | None = None,
    rng_seed: int = 2021,
    projection_batch_size: int = 1000,
) -> float:
    """Monge Inception Distance via sliced 2-Wasserstein projections.

    Uses equal-size empirical samples by deterministic subsampling to
    ``min(max_samples, len(real), len(gen))``. Inputs should be feature tensors
    with shape ``(N, D)`` and ``(M, D)`` on the same device.
    """
    if real_features.dim() != 2 or gen_features.dim() != 2:
        raise ValueError("MIND expects 2D feature tensors")
    if real_features.shape[1] != gen_features.shape[1]:
        raise ValueError(
            f"Feature dimensions differ: {real_features.shape[1]} vs {gen_features.shape[1]}"
        )
    if num_projections <= 0:
        raise ValueError("num_projections must be positive")

    device = gen_features.device
    real = real_features.to(device=device, dtype=torch.float32)
    gen = gen_features.to(device=device, dtype=torch.float32)

    n = min(real.shape[0], gen.shape[0])
    if max_samples is not None:
        n = min(n, max_samples)
    if n < 2:
        raise ValueError(f"MIND needs at least 2 samples, got {n}")

    if real.shape[0] != n:
        g = torch.Generator(device=device)
        g.manual_seed(rng_seed)
        real = real[torch.randperm(real.shape[0], device=device, generator=g)[:n]]
    else:
        real = real[:n]

    if gen.shape[0] != n:
        g = torch.Generator(device=device)
        g.manual_seed(rng_seed + 1)
        gen = gen[torch.randperm(gen.shape[0], device=device, generator=g)[:n]]
    else:
        gen = gen[:n]

    d = real.shape[1]
    alpha = 3.0 * d
    total = torch.zeros((), device=device, dtype=torch.float32)
    done = 0

    while done < num_projections:
        m = min(projection_batch_size, num_projections - done)
        g = torch.Generator(device=device)
        g.manual_seed(rng_seed + 1000 + done)
        directions = torch.randn(m, d, device=device, generator=g, dtype=torch.float32)
        directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-12)

        real_proj = real @ directions.T
        gen_proj = gen @ directions.T
        real_sorted = real_proj.sort(dim=0).values
        gen_sorted = gen_proj.sort(dim=0).values
        total = total + (real_sorted - gen_sorted).square().mean(dim=0).sum()
        done += m

    return float((alpha * total / num_projections).item())


@torch.inference_mode()
def compute_mind_curve(
    real_features: torch.Tensor,
    gen_features: torch.Tensor,
    sample_sizes: list[int] | tuple[int, ...],
    *,
    num_projections: int = 1000,
    rng_seed: int = 2021,
    projection_batch_size: int = 1000,
) -> dict[int, float]:
    """Compute MIND for several sample sizes with shared projections."""
    if not sample_sizes:
        raise ValueError("sample_sizes must be non-empty")
    sizes = sorted(set(int(s) for s in sample_sizes))
    if sizes[0] < 2:
        raise ValueError("MIND needs sample sizes >= 2")
    if real_features.dim() != 2 or gen_features.dim() != 2:
        raise ValueError("MIND expects 2D feature tensors")
    if real_features.shape[1] != gen_features.shape[1]:
        raise ValueError(
            f"Feature dimensions differ: {real_features.shape[1]} vs {gen_features.shape[1]}"
        )
    if num_projections <= 0:
        raise ValueError("num_projections must be positive")

    device = gen_features.device
    real = real_features.to(device=device, dtype=torch.float32)
    gen = gen_features.to(device=device, dtype=torch.float32)

    max_n = min(max(sizes), real.shape[0], gen.shape[0])
    sizes = [min(s, max_n) for s in sizes]
    sizes = sorted(set(sizes))
    if max_n < 2:
        raise ValueError(f"MIND needs at least 2 samples, got {max_n}")

    if real.shape[0] != max_n:
        g = torch.Generator(device=device)
        g.manual_seed(rng_seed)
        real = real[torch.randperm(real.shape[0], device=device, generator=g)[:max_n]]
    else:
        real = real[:max_n]

    if gen.shape[0] != max_n:
        g = torch.Generator(device=device)
        g.manual_seed(rng_seed + 1)
        gen = gen[torch.randperm(gen.shape[0], device=device, generator=g)[:max_n]]
    else:
        gen = gen[:max_n]

    d = real.shape[1]
    alpha = 3.0 * d
    totals = {n: torch.zeros((), device=device, dtype=torch.float32) for n in sizes}
    done = 0

    while done < num_projections:
        m = min(projection_batch_size, num_projections - done)
        g = torch.Generator(device=device)
        g.manual_seed(rng_seed + 1000 + done)
        directions = torch.randn(m, d, device=device, generator=g, dtype=torch.float32)
        directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-12)

        real_proj = real @ directions.T
        gen_proj = gen @ directions.T
        for n in sizes:
            real_sorted = real_proj[:n].sort(dim=0).values
            gen_sorted = gen_proj[:n].sort(dim=0).values
            totals[n] = totals[n] + (real_sorted - gen_sorted).square().mean(dim=0).sum()
        done += m

    return {n: float((alpha * total / num_projections).item()) for n, total in totals.items()}
