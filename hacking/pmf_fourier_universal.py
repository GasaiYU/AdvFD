"""Fast universal Inception-FD hacking experiment for frozen pMF generators.

The experiment has three strictly separated cached splits:

* optimization: estimate a Fourier or full-resolution bandpass direction;
* validation: select/early-stop using Inception FD only;
* test: report the final Inception and CLIP dose response.

The generator is used only when a cache shard is missing.  During pattern
optimization the generator is unloaded, Inception is frozen, and CLIP is not
loaded. The only trainable object is one pattern shared by every image.
Optimization gradients come from one joint FD over the complete optimization
split, evaluated with a two-pass sufficient-statistics backward rather than
noisy per-batch FDs.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

# Allow ``python hacking/pmf_fourier_universal.py`` and torchrun with a file
# entry point, not only ``python -m ...``.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from frechet_distance.judges import infer_stats_path
from frechet_distance.losses import (
    compute_frechet_distance_loss,
    load_mu_and_sigma_reference,
    precompute_sigma_ref_sqrt,
)
from main_pmf_universal_pattern import (
    Moments,
    _bootstrap_mean_ci,
    _extract_primary_features,
    _generate_fixed_batch,
    _linear_slope,
    _load_frozen_pmf,
    _load_frozen_repr,
    _stats_to_fd,
    _to_unit_range,
)
from utils.distributed_util import (
    get_global_rank,
    get_world_size,
    is_enabled,
)


logger = logging.getLogger("FD_loss")
CACHE_VERSION = 1


def _barrier() -> None:
    if is_enabled():
        dist.barrier()


def _all_reduce_sum_(tensor: torch.Tensor) -> torch.Tensor:
    if is_enabled():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def _broadcast_float(value: float, src: int = 0) -> float:
    tensor = torch.tensor(
        [value if get_global_rank() == src else 0.0],
        device="cuda",
        dtype=torch.float64,
    )
    if is_enabled():
        dist.broadcast(tensor, src=src)
    return float(tensor.item())


def _atomic_json_dump(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp-rank{get_global_rank()}-pid{os.getpid()}"
    )
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _checkpoint_identity(path: str) -> dict[str, object]:
    resolved = Path(path).expanduser().resolve()
    identity: dict[str, object] = {"path": str(resolved)}
    if resolved.exists():
        stat = resolved.stat()
        identity.update({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return identity


def _cache_spec(args: argparse.Namespace) -> dict[str, object]:
    splits: dict[str, dict[str, int]] = {
        "optimization": {
            "images": args.hack_optimization_images,
            "seed": args.hack_optimization_seed,
        },
    }
    if not args.hack_overfit_only:
        splits.update(
            {
                "validation": {
                    "images": args.hack_validation_images,
                    "seed": args.hack_validation_seed,
                },
                "test": {
                    "images": args.hack_test_images,
                    "seed": args.hack_test_seed,
                },
            }
        )
    return {
        "version": CACHE_VERSION,
        "checkpoint": _checkpoint_identity(args.load_from),
        "model": args.model,
        "img_size": args.img_size,
        "patch_size": args.patch_size,
        "noise_scale": args.noise_scale,
        "cfg": args.cfg,
        "interval_min": args.interval_min,
        "interval_max": args.interval_max,
        "num_sampling_steps": args.num_sampling_steps,
        "same_noise": args.same_noise,
        "amp_dtype": args.dtype,
        "num_classes": args.num_classes,
        "world_size": get_world_size(),
        "cache_dtype": args.hack_cache_dtype,
        "overfit_only": args.hack_overfit_only,
        "splits": splits,
    }


def _fingerprint(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _index_path(cache_root: Path, split: str, rank: int) -> Path:
    return cache_root / split / f"rank_{rank:05d}" / "index.json"


def _rank_cache_valid(
    cache_root: Path,
    split: str,
    *,
    expected_count: int,
    fingerprint: str,
) -> bool:
    index_path = _index_path(cache_root, split, get_global_rank())
    if not index_path.exists():
        return False
    try:
        with index_path.open() as handle:
            index = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    if (
        index.get("fingerprint") != fingerprint
        or int(index.get("count", -1)) != expected_count
    ):
        return False
    entries = index.get("entries", [])
    if sum(int(entry["count"]) for entry in entries) != expected_count:
        return False
    return all((cache_root / entry["path"]).exists() for entry in entries)


def _encode_cached(images: torch.Tensor, dtype: str) -> torch.Tensor:
    if dtype == "uint8":
        return (
            images.add(1.0)
            .mul(127.5)
            .round()
            .clamp_(0, 255)
            .to(torch.uint8)
            .cpu()
        )
    if dtype == "float16":
        return images.to(dtype=torch.float16, device="cpu")
    raise ValueError(f"Unsupported cache dtype: {dtype}")


def _decode_cached(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    if images.dtype == torch.uint8:
        return (
            images.to(device=device, dtype=torch.float32, non_blocking=True)
            .mul_(2.0 / 255.0)
            .sub_(1.0)
        )
    return images.to(device=device, dtype=torch.float32, non_blocking=True)


@torch.no_grad()
def _generate_cache_split(
    args: argparse.Namespace,
    generator: nn.Module,
    *,
    cache_root: Path,
    split: str,
    total_images: int,
    seed: int,
    fingerprint: str,
) -> None:
    rank = get_global_rank()
    world_size = get_world_size()
    local_count = total_images // world_size
    shard_dir = cache_root / split / f"rank_{rank:05d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    local_start = 0
    chunk_index = 0
    while local_start < local_count:
        local_bsz = min(args.batch_size, local_count - local_start)
        global_start = rank * local_count + local_start
        labels = (
            torch.arange(
                global_start,
                global_start + local_bsz,
                device="cuda",
                dtype=torch.long,
            )
            % args.num_classes
        )
        images = _generate_fixed_batch(
            generator,
            args,
            batch_size=local_bsz,
            base_seed=seed,
            batch_index=chunk_index,
            labels=labels,
        )
        encoded = _encode_cached(images, args.hack_cache_dtype)
        chunk_path = shard_dir / f"chunk_{chunk_index:05d}.pt"
        temporary = chunk_path.with_name(
            f".{chunk_path.name}.tmp-pid{os.getpid()}"
        )
        torch.save({"images": encoded}, temporary)
        os.replace(temporary, chunk_path)
        entries.append(
            {
                "path": str(chunk_path.relative_to(cache_root)),
                "start": local_start,
                "count": local_bsz,
            }
        )
        local_start += local_bsz
        chunk_index += 1
        logger.info(
            "[cache:%s] rank=%d generated %d/%d local images",
            split,
            rank,
            local_start,
            local_count,
        )
        del images, encoded, labels

    _atomic_json_dump(
        {
            "version": CACHE_VERSION,
            "fingerprint": fingerprint,
            "split": split,
            "rank": rank,
            "world_size": world_size,
            "count": local_count,
            "dtype": args.hack_cache_dtype,
            "entries": entries,
        },
        _index_path(cache_root, split, rank),
    )


def _ensure_image_cache(args: argparse.Namespace) -> Path:
    cache_root = Path(args.hack_cache_dir).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_root / "manifest.json"
    spec = _cache_spec(args)
    fingerprint = _fingerprint(spec)
    split_specs = spec["splits"]
    assert isinstance(split_specs, dict)

    manifest_matches = False
    manifest: dict[str, object] = {}
    if manifest_path.exists():
        try:
            with manifest_path.open() as handle:
                manifest = json.load(handle)
            manifest_matches = (
                manifest.get("fingerprint") == fingerprint
                and manifest.get("spec") == spec
            )
        except (OSError, json.JSONDecodeError):
            manifest_matches = False

    # An overfit-only run needs just the optimization shard. Reuse that shard
    # from a compatible full optimization/validation/test cache instead of
    # forcing another 50k generator pass merely because the manifest is a
    # strict superset.
    if (
        args.hack_overfit_only
        and manifest_path.exists()
        and not manifest_matches
        and not args.hack_overwrite_cache
    ):
        existing_spec = manifest.get("spec")
        existing_fingerprint = manifest.get("fingerprint")
        if isinstance(existing_spec, dict) and isinstance(
            existing_fingerprint, str
        ):
            requested_core = {
                key: value
                for key, value in spec.items()
                if key not in {"overfit_only", "splits"}
            }
            existing_core = {
                key: value
                for key, value in existing_spec.items()
                if key not in {"overfit_only", "splits"}
            }
            existing_splits = existing_spec.get("splits")
            requested_optimization = split_specs.get("optimization")
            existing_optimization = (
                existing_splits.get("optimization")
                if isinstance(existing_splits, dict)
                else None
            )
            if (
                requested_core == existing_core
                and requested_optimization == existing_optimization
            ):
                spec = existing_spec
                fingerprint = existing_fingerprint
                split_specs = {
                    "optimization": existing_optimization
                }
                manifest_matches = True
                logger.info(
                    "Reusing optimization shard from compatible full cache: %s",
                    cache_root,
                )

    if manifest_path.exists() and not manifest_matches:
        if not args.hack_overwrite_cache:
            raise ValueError(
                f"Cache manifest {manifest_path} does not match this run. "
                "Use a different --hack_cache_dir or pass "
                "--hack_overwrite_cache to rewrite explicit cache shards."
            )
        logger.warning("Cache manifest mismatch; rewriting this run's shards")

    missing: list[str] = []
    for split, split_spec in split_specs.items():
        assert isinstance(split_spec, dict)
        local_count = int(split_spec["images"]) // get_world_size()
        if (
            args.hack_overwrite_cache
            or not manifest_matches
            or not _rank_cache_valid(
                cache_root,
                split,
                expected_count=local_count,
                fingerprint=fingerprint,
            )
        ):
            missing.append(split)

    need_local = 1 if missing else 0
    need_tensor = torch.tensor([need_local], device="cuda", dtype=torch.int32)
    if is_enabled():
        dist.all_reduce(need_tensor, op=dist.ReduceOp.MAX)
    if int(need_tensor.item()) != 0:
        logger.info(
            "At least one cache shard is missing; loading the frozen generator once"
        )
        generator = _load_frozen_pmf(args)
        for split, split_spec in split_specs.items():
            if split not in missing:
                continue
            assert isinstance(split_spec, dict)
            _generate_cache_split(
                args,
                generator,
                cache_root=cache_root,
                split=split,
                total_images=int(split_spec["images"]),
                seed=int(split_spec["seed"]),
                fingerprint=fingerprint,
            )
        _barrier()
        del generator
        gc.collect()
        torch.cuda.empty_cache()

    # Validate every rank before publishing the manifest.
    for split, split_spec in split_specs.items():
        assert isinstance(split_spec, dict)
        local_count = int(split_spec["images"]) // get_world_size()
        valid = _rank_cache_valid(
            cache_root,
            split,
            expected_count=local_count,
            fingerprint=fingerprint,
        )
        valid_tensor = torch.tensor(
            [1 if valid else 0], device="cuda", dtype=torch.int32
        )
        if is_enabled():
            dist.all_reduce(valid_tensor, op=dist.ReduceOp.MIN)
        if int(valid_tensor.item()) != 1:
            raise RuntimeError(f"Cache validation failed for split {split!r}")

    if get_global_rank() == 0:
        _atomic_json_dump(
            {"fingerprint": fingerprint, "spec": spec}, manifest_path
        )
        logger.info("Fixed generated-image cache is ready: %s", cache_root)
    _barrier()
    return cache_root


@dataclass(frozen=True)
class CacheEntry:
    path: Path
    start: int
    count: int


class CachedImageSplit:
    """One rank's immutable cache shard with range-based streaming."""

    def __init__(self, cache_root: Path, split: str):
        index_path = _index_path(cache_root, split, get_global_rank())
        with index_path.open() as handle:
            index = json.load(handle)
        self.name = split
        self.count = int(index["count"])
        self.entries = [
            CacheEntry(
                path=cache_root / entry["path"],
                start=int(entry["start"]),
                count=int(entry["count"]),
            )
            for entry in index["entries"]
        ]

    @staticmethod
    def _load_entry(entry: CacheEntry) -> torch.Tensor:
        payload = torch.load(
            entry.path, map_location="cpu", weights_only=True
        )
        images = payload["images"]
        if images.shape[0] != entry.count:
            raise RuntimeError(
                f"Corrupt cache chunk {entry.path}: "
                f"{images.shape[0]} != {entry.count}"
            )
        return images

    def load_all(self) -> torch.Tensor:
        return torch.cat(
            [self._load_entry(entry) for entry in self.entries], dim=0
        )

    def iter_range(
        self,
        start: int,
        end: int,
        batch_size: int,
    ) -> Iterator[torch.Tensor]:
        if not 0 <= start <= end <= self.count:
            raise ValueError(
                f"Invalid cache range [{start}, {end}) for count={self.count}"
            )
        pending: list[torch.Tensor] = []
        pending_count = 0
        for entry in self.entries:
            entry_end = entry.start + entry.count
            if entry_end <= start or entry.start >= end:
                continue
            images = self._load_entry(entry)
            lo = max(start, entry.start) - entry.start
            hi = min(end, entry_end) - entry.start
            piece = images[lo:hi]
            pending.append(piece)
            pending_count += piece.shape[0]
            while pending_count >= batch_size:
                merged = torch.cat(pending, dim=0)
                yield merged[:batch_size]
                remainder = merged[batch_size:]
                pending = [remainder] if remainder.numel() else []
                pending_count = remainder.shape[0]
        if pending_count:
            yield torch.cat(pending, dim=0)

    def iter_batches(self, batch_size: int) -> Iterator[torch.Tensor]:
        yield from self.iter_range(0, self.count, batch_size)


class FourierPattern(nn.Module):
    """Low-dimensional real Fourier pattern with no DC component."""

    def __init__(
        self,
        size: int,
        num_modes: int,
        min_radius: float,
        max_radius: float,
        eps: float = 1e-12,
    ):
        super().__init__()
        if size < 3:
            raise ValueError("Fourier pattern size must be at least 3")
        self.size = int(size)
        self.eps = float(eps)
        frequencies = self._select_frequencies(
            size, num_modes, min_radius, max_radius
        )
        self.frequencies = tuple(frequencies)
        self.coeff = nn.Parameter(
            torch.zeros(3, len(frequencies), 2, dtype=torch.float32)
        )

        coordinate = torch.arange(size, dtype=torch.float32)
        yy, xx = torch.meshgrid(coordinate, coordinate, indexing="ij")
        phases = torch.stack(
            [
                2.0 * math.pi * (kx * yy + ky * xx) / size
                for kx, ky in frequencies
            ]
        )
        self.register_buffer("cos_basis", torch.cos(phases), persistent=True)
        self.register_buffer("sin_basis", torch.sin(phases), persistent=True)

    @staticmethod
    def _select_frequencies(
        size: int,
        num_modes: int,
        min_radius: float,
        max_radius: float,
    ) -> list[tuple[int, int]]:
        # Excluding the even-grid Nyquist edge keeps every sine/cosine pair
        # unique and avoids zero-gradient imaginary/self-conjugate modes.
        lower = -(size // 2) + 1
        upper = size // 2
        candidates: list[tuple[float, float, int, int]] = []
        center = 0.5 * (min_radius + max_radius)
        for kx in range(lower, upper):
            for ky in range(lower, upper):
                if kx == 0 and ky == 0:
                    continue
                # Keep one representative from each +/- frequency pair.
                if not (kx > 0 or (kx == 0 and ky > 0)):
                    continue
                radius = math.sqrt(kx * kx + ky * ky) / size
                if min_radius <= radius <= max_radius:
                    angle = math.atan2(ky, kx)
                    candidates.append(
                        (abs(radius - center), angle, kx, ky)
                    )
        candidates.sort()
        if num_modes < 1 or num_modes > len(candidates):
            raise ValueError(
                f"Requested {num_modes} Fourier modes, but radius band "
                f"[{min_radius}, {max_radius}] at size={size} contains "
                f"{len(candidates)} unique modes"
            )
        return [(kx, ky) for _, _, kx, ky in candidates[:num_modes]]

    def patch(self, *, normalize: bool) -> torch.Tensor:
        cosine = torch.einsum(
            "cm,mhw->chw", self.coeff[..., 0], self.cos_basis
        )
        sine = torch.einsum(
            "cm,mhw->chw", self.coeff[..., 1], self.sin_basis
        )
        patch = (cosine + sine).unsqueeze(0)
        # This is theoretically zero already; subtraction makes the invariant
        # exact in floating point and protects future basis changes.
        patch = patch - patch.mean(dim=(-2, -1), keepdim=True)
        if normalize:
            rms = patch.square().mean().add(self.eps).sqrt()
            patch = patch / rms
        return patch

    @torch.no_grad()
    def normalize_coefficients_(self) -> None:
        rms = self.coeff.square().mean().add(self.eps).sqrt()
        self.coeff.div_(rms)


class SpatialBandpassPattern(nn.Module):
    """Full-resolution spatial noise projected to a fixed frequency band."""

    def __init__(
        self,
        size: int,
        min_radius: float,
        max_radius: float,
        eps: float = 1e-12,
    ):
        super().__init__()
        if size < 3:
            raise ValueError("Spatial pattern size must be at least 3")
        self.size = int(size)
        self.eps = float(eps)
        self.coeff = nn.Parameter(
            torch.zeros(3, size, size, dtype=torch.float32)
        )
        frequency_y = torch.fft.fftfreq(size)
        frequency_x = torch.fft.rfftfreq(size)
        radius = torch.sqrt(
            frequency_y[:, None].square()
            + frequency_x[None, :].square()
        )
        mask = (
            (radius >= float(min_radius))
            & (radius <= float(max_radius))
        )
        if not bool(mask.any()):
            raise ValueError(
                "Spatial frequency band contains no discrete frequencies"
            )
        self.register_buffer(
            "frequency_mask",
            mask.to(torch.float32),
            persistent=True,
        )

    def _bandpass(self, value: torch.Tensor) -> torch.Tensor:
        frequency = torch.fft.rfft2(value, norm="ortho")
        return torch.fft.irfft2(
            frequency * self.frequency_mask,
            s=(self.size, self.size),
            norm="ortho",
        )

    def patch(self, *, normalize: bool) -> torch.Tensor:
        patch = self._bandpass(self.coeff).unsqueeze(0)
        patch = patch - patch.mean(dim=(-2, -1), keepdim=True)
        if normalize:
            rms = patch.square().mean().add(self.eps).sqrt()
            patch = patch / rms
        return patch

    @torch.no_grad()
    def normalize_coefficients_(self) -> None:
        projected = self._bandpass(self.coeff)
        projected = projected - projected.mean(
            dim=(-2, -1), keepdim=True
        )
        rms = projected.square().mean().add(self.eps).sqrt()
        self.coeff.copy_(projected / rms)


UniversalHackingPattern = FourierPattern | SpatialBandpassPattern


def _tile_patch(
    patch: torch.Tensor,
    height: int,
    width: int,
    phase: tuple[int, int],
) -> torch.Tensor:
    patch = torch.roll(patch, shifts=phase, dims=(-2, -1))
    repeat_h = math.ceil(height / patch.shape[-2])
    repeat_w = math.ceil(width / patch.shape[-1])
    return patch.repeat(1, 1, repeat_h, repeat_w)[..., :height, :width]


def _apply_fourier_pattern(
    images: torch.Tensor,
    pattern: UniversalHackingPattern,
    *,
    alpha: float,
    phase: tuple[int, int] = (0, 0),
    normalize: bool = True,
) -> torch.Tensor:
    patch = pattern.patch(normalize=normalize)
    tiled = _tile_patch(patch, images.shape[-2], images.shape[-1], phase)
    return (images + float(alpha) * tiled).clamp(-1.0, 1.0)


def _phase_for(
    seed: int,
    sequence_index: int,
    pattern_size: int,
    enabled: bool,
) -> tuple[int, int]:
    if not enabled:
        return (0, 0)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1_000_003 * sequence_index)
    values = torch.randint(
        0, pattern_size, (2,), generator=generator
    ).tolist()
    return int(values[0]), int(values[1])


def _full_cached_fd_gradient(
    args: argparse.Namespace,
    inception: nn.Module,
    pattern: UniversalHackingPattern,
    optimization_images: torch.Tensor,
    mu_ref: torch.Tensor,
    sigma_ref: torch.Tensor,
    sigma_ref_sqrt: torch.Tensor | None,
    *,
    phase_round: int,
    normalize_pattern: bool,
) -> tuple[float, torch.Tensor]:
    """Exact full-split FD gradient without retaining every image graph.

    The first no-grad pass forms the global feature mean and covariance in
    float64, avoiding catastrophic cancellation from casting raw first/second
    moments to float32. FD is differentiated with respect to those compact
    statistics. A second pass applies the analytic per-feature chain rule and
    backpropagates it batch by batch to the shared Fourier coefficients.
    Summed across batches and ranks, this is the gradient of one joint FD, not
    an average of noisy per-batch FDs.
    """
    device = torch.device("cuda")
    local_bsz = args.hack_gradient_batch_size // get_world_size()
    num_local_batches = math.ceil(optimization_images.shape[0] / local_bsz)
    feat_dim = int(mu_ref.shape[0])
    feat_sum = torch.zeros(feat_dim, device=device, dtype=torch.float64)
    feat_outer = torch.zeros(
        feat_dim, feat_dim, device=device, dtype=torch.float64
    )
    local_count = 0

    # Pass 1: full 5k sufficient statistics, without retaining activations.
    with torch.no_grad():
        for batch_index, start in enumerate(
            range(0, optimization_images.shape[0], local_bsz)
        ):
            cached_images = optimization_images[start : start + local_bsz]
            phase = _phase_for(
                args.hack_phase_seed,
                phase_round * num_local_batches + batch_index,
                pattern.size,
                args.hack_random_phase,
            )
            images = _decode_cached(cached_images, device)
            perturbed = _to_unit_range(
                _apply_fourier_pattern(
                    images,
                    pattern,
                    alpha=args.hack_train_alpha,
                    phase=phase,
                    normalize=normalize_pattern,
                )
            )
            features = _extract_primary_features(
                inception,
                perturbed,
                use_amp=False,
                amp_dtype=args.amp_dtype,
            )
            features64 = features.double()
            feat_sum.add_(features64.sum(dim=0))
            feat_outer.addmm_(features64.T, features64)
            local_count += features.shape[0]
            del images, perturbed, features, features64

    count = torch.tensor([local_count], device=device, dtype=torch.int64)
    _all_reduce_sum_(feat_sum)
    _all_reduce_sum_(feat_outer)
    _all_reduce_sum_(count)
    global_count = int(count.item())
    if global_count != args.hack_optimization_images:
        raise RuntimeError(
            f"Full optimization statistics contain {global_count} images, "
            f"expected {args.hack_optimization_images}"
        )

    # Keep the centered statistics and the 2048-D covariance square-root
    # calculation in float64. Casting either raw S/Q or the resulting
    # covariance to float32 creates a visible positive FID bias for nearly
    # singular Inception covariances.
    mu64 = feat_sum / global_count
    sigma64 = (
        feat_outer
        - feat_sum.unsqueeze(1) * feat_sum.unsqueeze(0) / global_count
    ) / (global_count - 1)
    mu_variable = mu64.detach().requires_grad_(True)
    sigma_variable = sigma64.detach().requires_grad_(True)
    sigma_for_fd = sigma_variable
    if args.hack_cov_eps > 0:
        sigma_for_fd = sigma_for_fd + args.hack_cov_eps * torch.eye(
            sigma_for_fd.shape[0],
            device=sigma_for_fd.device,
            dtype=sigma_for_fd.dtype,
        )
    fid = compute_frechet_distance_loss(
        mu_ref,
        sigma_ref,
        mu=mu_variable,
        sigma=sigma_for_fd,
        sigma_ref_sqrt=sigma_ref_sqrt,
    )
    grad_mu, grad_sigma = torch.autograd.grad(
        fid,
        (mu_variable, sigma_variable),
        create_graph=False,
    )
    grad_mu = grad_mu.detach()
    grad_sigma = grad_sigma.detach()
    global_mu = mu_variable.detach()
    symmetric_grad_sigma = (grad_sigma + grad_sigma.T).detach()
    fid_value = float(fid.detach())

    # For unbiased covariance C = sum_i (f_i-mu)(f_i-mu)^T/(N-1):
    # dL/df_i = dL/dmu/N
    #          + (f_i-mu) @ (dL/dC + dL/dC^T)/(N-1).
    pattern_gradient = torch.zeros_like(pattern.coeff)
    for batch_index, start in enumerate(
        range(0, optimization_images.shape[0], local_bsz)
    ):
        cached_images = optimization_images[start : start + local_bsz]
        phase = _phase_for(
            args.hack_phase_seed,
            phase_round * num_local_batches + batch_index,
            pattern.size,
            args.hack_random_phase,
        )
        images = _decode_cached(cached_images, device)
        perturbed = _to_unit_range(
            _apply_fourier_pattern(
                images,
                pattern,
                alpha=args.hack_train_alpha,
                phase=phase,
                normalize=normalize_pattern,
            )
        )
        features = _extract_primary_features(
            inception,
            perturbed,
            use_amp=False,
            amp_dtype=args.amp_dtype,
        )
        feature_gradient = (
            grad_mu.unsqueeze(0) / global_count
            + (features.detach() - global_mu)
            @ symmetric_grad_sigma
            / (global_count - 1)
        )
        statistic_surrogate = (
            features * feature_gradient.detach()
        ).sum()
        batch_gradient = torch.autograd.grad(
            statistic_surrogate,
            pattern.coeff,
            create_graph=False,
        )[0]
        pattern_gradient.add_(batch_gradient.detach())
        del (
            images,
            perturbed,
            features,
            feature_gradient,
            statistic_surrogate,
            batch_gradient,
        )

    _all_reduce_sum_(pattern_gradient)
    if not torch.isfinite(pattern_gradient).all():
        raise FloatingPointError(
            "Non-finite Fourier gradient; increase --hack_cov_eps"
        )
    del (
        feat_sum,
        feat_outer,
        count,
        mu64,
        sigma64,
        mu_variable,
        sigma_variable,
        sigma_for_fd,
        fid,
        grad_mu,
        grad_sigma,
        global_mu,
        symmetric_grad_sigma,
    )
    logger.info(
        "[full-gradient] images=%d streaming_global_bsz=%d "
        "joint_fid=%.6f phase_round=%d",
        global_count,
        args.hack_gradient_batch_size,
        fid_value,
        phase_round,
    )
    return fid_value, pattern_gradient


@torch.no_grad()
def _cached_inception_fd(
    args: argparse.Namespace,
    inception: nn.Module,
    pattern: UniversalHackingPattern,
    split: CachedImageSplit,
    ref_mu: np.ndarray,
    ref_sigma: np.ndarray,
    *,
    alpha: float,
) -> float:
    moments = Moments.create(ref_mu.shape[0], torch.device("cuda"))
    for cached_images in split.iter_batches(args.eval_bsz):
        images = _decode_cached(cached_images, torch.device("cuda"))
        perturbed = _to_unit_range(
            _apply_fourier_pattern(
                images,
                pattern,
                alpha=alpha,
                phase=(0, 0),
                normalize=True,
            )
        )
        features = _extract_primary_features(
            inception,
            perturbed,
            use_amp=False,
            amp_dtype=args.amp_dtype,
        )
        moments.update(features)
        del cached_images, images, perturbed, features

    count = torch.tensor(
        [moments.count], device="cuda", dtype=torch.int64
    )
    if is_enabled():
        dist.reduce(moments.feat_sum, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(moments.feat_outer, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(count, dst=0, op=dist.ReduceOp.SUM)
    value = 0.0
    if get_global_rank() == 0:
        value = _stats_to_fd(
            moments.feat_sum.cpu().numpy(),
            moments.feat_outer.cpu().numpy(),
            int(count.item()),
            ref_mu,
            ref_sigma,
        )
    return _broadcast_float(value)


def _save_pattern_checkpoint(
    args: argparse.Namespace,
    pattern: UniversalHackingPattern,
    *,
    name: str,
    stage: str,
    step: int,
    validation_fid: float,
    validation_baseline_fid: float,
) -> Path | None:
    if get_global_rank() != 0:
        return None
    path = Path(args.ckpt_dir) / name
    pattern_type = (
        "fourier"
        if isinstance(pattern, FourierPattern)
        else "spatial_bandpass"
    )
    payload: dict[str, object] = {
        "pattern": pattern.state_dict(),
        "pattern_type": pattern_type,
        "pattern_size": pattern.size,
        "fourier_min_radius": args.hack_fourier_min_radius,
        "fourier_max_radius": args.hack_fourier_max_radius,
        "stage": stage,
        "step": step,
        "validation_fid": validation_fid,
        "validation_baseline_fid": validation_baseline_fid,
        "train_alpha": args.hack_train_alpha,
        "generator_checkpoint": args.load_from,
        "spatial_pattern": (
            pattern.patch(normalize=True)[0].detach().cpu()
        ),
    }
    if isinstance(pattern, FourierPattern):
        payload.update(
            {
                "frequencies": [
                    list(pair) for pair in pattern.frequencies
                ],
                "fourier_modes": len(pattern.frequencies),
            }
        )
    torch.save(payload, path)
    logger.info("Saved %s pattern checkpoint: %s", pattern_type, path)
    return path


def _load_pattern_checkpoint(
    path: str | Path,
    pattern: UniversalHackingPattern,
) -> dict[str, object]:
    checkpoint = torch.load(
        path, map_location="cuda", weights_only=False
    )
    expected_type = (
        "fourier"
        if isinstance(pattern, FourierPattern)
        else "spatial_bandpass"
    )
    checkpoint_type = checkpoint.get("pattern_type", "fourier")
    if checkpoint_type != expected_type:
        raise ValueError(
            f"Checkpoint pattern_type={checkpoint_type!r} does not match "
            f"current pattern_type={expected_type!r}"
        )
    if isinstance(pattern, FourierPattern):
        expected = [list(pair) for pair in pattern.frequencies]
        if checkpoint.get("frequencies") != expected:
            raise ValueError(
                "Checkpoint Fourier frequencies do not match current "
                "arguments"
            )
    pattern.load_state_dict(checkpoint["pattern"])
    logger.info(
        "Loaded %s pattern %s (stage=%s step=%s validation_fid=%s)",
        checkpoint_type,
        path,
        checkpoint.get("stage"),
        checkpoint.get("step"),
        checkpoint.get("validation_fid"),
    )
    return checkpoint


@torch.no_grad()
def _save_pattern_preview(
    args: argparse.Namespace,
    pattern: UniversalHackingPattern,
) -> None:
    if get_global_rank() != 0:
        return
    from PIL import Image

    patch = pattern.patch(normalize=True)[0].float().cpu()
    np.save(Path(args.log_dir) / "fourier_pattern.npy", patch.numpy())
    # A robust display mapping only; the .npy file contains the exact pattern.
    display = patch.div(3.0).mul(0.5).add(0.5).clamp(0.0, 1.0)
    tiled = _tile_patch(
        display.unsqueeze(0),
        args.img_size,
        args.img_size,
        (0, 0),
    )[0]
    image = tiled.mul(255).byte().permute(1, 2, 0).numpy()
    Image.fromarray(image).save(Path(args.log_dir) / "fourier_pattern.png")


def _optimize_pattern(
    args: argparse.Namespace,
    inception: nn.Module,
    pattern: UniversalHackingPattern,
    cache_root: Path,
    wandb_logger=None,
) -> dict[str, object]:
    optimization_split = CachedImageSplit(cache_root, "optimization")
    validation_split = CachedImageSplit(cache_root, "validation")
    optimization_images = optimization_split.load_all()

    mu_ref, sigma_ref = load_mu_and_sigma_reference(args.fid_stats_path)
    sigma_ref_sqrt = (
        precompute_sigma_ref_sqrt(sigma_ref)
        if args.fd_eigvalsh
        else None
    )
    reference = np.load(args.fid_stats_path)
    ref_mu = np.asarray(reference["mu"], dtype=np.float64)
    ref_sigma = np.asarray(reference["sigma"], dtype=np.float64)

    pattern.coeff.data.zero_()
    validation_baseline = _cached_inception_fd(
        args,
        inception,
        pattern,
        validation_split,
        ref_mu,
        ref_sigma,
        alpha=args.hack_train_alpha,
    )
    logger.info(
        "[validation] zero-pattern FID=%.6f at alpha=%.8f",
        validation_baseline,
        args.hack_train_alpha,
    )

    started = time.perf_counter()
    one_shot_full_fid, one_shot_gradient = _full_cached_fd_gradient(
        args,
        inception,
        pattern,
        optimization_images,
        mu_ref,
        sigma_ref,
        sigma_ref_sqrt,
        phase_round=0,
        normalize_pattern=False,
    )
    gradient_rms = float(
        one_shot_gradient.square().mean().sqrt().item()
    )
    if not math.isfinite(gradient_rms) or gradient_rms <= 0:
        raise FloatingPointError(
            f"Invalid one-shot gradient RMS: {gradient_rms}"
        )
    with torch.no_grad():
        pattern.coeff.copy_(
            -one_shot_gradient
            / one_shot_gradient.square().mean().add(1e-12).sqrt()
        )
        pattern.normalize_coefficients_()

    one_shot_val = _cached_inception_fd(
        args,
        inception,
        pattern,
        validation_split,
        ref_mu,
        ref_sigma,
        alpha=args.hack_train_alpha,
    )
    best_val = one_shot_val
    best_state = {
        key: value.detach().clone()
        for key, value in pattern.state_dict().items()
    }
    best_stage = "one_shot"
    best_step = 0
    _save_pattern_checkpoint(
        args,
        pattern,
        name="fourier_pattern_best.pth",
        stage=best_stage,
        step=best_step,
        validation_fid=best_val,
        validation_baseline_fid=validation_baseline,
    )
    logger.info(
        "[one-shot] full_optimization_fid=%.6f gradient_rms=%.6g "
        "validation_fid=%.6f delta_vs_zero=%+.6f elapsed=%.1fs",
        one_shot_full_fid,
        gradient_rms,
        one_shot_val,
        one_shot_val - validation_baseline,
        time.perf_counter() - started,
    )
    if wandb_logger is not None:
        wandb_logger.update(
            {
                "hack/validation_zero_fid": validation_baseline,
                "hack/validation_fid": one_shot_val,
                "hack/validation_delta": one_shot_val
                - validation_baseline,
                "hack/gradient_rms": gradient_rms,
                "hack/one_shot_full_optimization_fid": one_shot_full_fid,
            },
            step=0,
        )

    checks_without_improvement = 0
    last_validated_step = 0
    for step in range(1, args.hack_pgd_steps + 1):
        full_optimization_fid, gradient = _full_cached_fd_gradient(
            args,
            inception,
            pattern,
            optimization_images,
            mu_ref,
            sigma_ref,
            sigma_ref_sqrt,
            phase_round=step,
            normalize_pattern=True,
        )
        gradient_rms_tensor = gradient.square().mean().add(1e-12).sqrt()
        with torch.no_grad():
            pattern.coeff.add_(
                gradient / gradient_rms_tensor,
                alpha=-args.hack_pgd_step_size,
            )
            pattern.normalize_coefficients_()

        should_validate = (
            step % args.hack_validate_every == 0
            or step == args.hack_pgd_steps
        )
        if not should_validate:
            continue
        last_validated_step = step
        validation_fid = _cached_inception_fd(
            args,
            inception,
            pattern,
            validation_split,
            ref_mu,
            ref_sigma,
            alpha=args.hack_train_alpha,
        )
        improved = validation_fid < (
            best_val - args.hack_min_validation_improvement
        )
        if improved:
            best_val = validation_fid
            best_state = {
                key: value.detach().clone()
                for key, value in pattern.state_dict().items()
            }
            best_stage = "pgd"
            best_step = step
            checks_without_improvement = 0
            _save_pattern_checkpoint(
                args,
                pattern,
                name="fourier_pattern_best.pth",
                stage=best_stage,
                step=best_step,
                validation_fid=best_val,
                validation_baseline_fid=validation_baseline,
            )
        else:
            checks_without_improvement += 1
        logger.info(
            "[PGD] step=%d/%d full_optimization_fid=%.6f "
            "validation_fid=%.6f "
            "best=%.6f improved=%s patience=%d/%d",
            step,
            args.hack_pgd_steps,
            full_optimization_fid,
            validation_fid,
            best_val,
            improved,
            checks_without_improvement,
            args.hack_early_stop_patience,
        )
        if wandb_logger is not None:
            wandb_logger.update(
                {
                    "hack/validation_fid": validation_fid,
                    "hack/validation_delta": validation_fid
                    - validation_baseline,
                    "hack/best_validation_fid": best_val,
                    "hack/pgd_full_optimization_fid": full_optimization_fid,
                },
                step=step,
            )
        if (
            args.hack_early_stop_patience > 0
            and checks_without_improvement
            >= args.hack_early_stop_patience
        ):
            logger.info("[PGD] validation early stopping at step=%d", step)
            break

    pattern.load_state_dict(best_state)
    _save_pattern_checkpoint(
        args,
        pattern,
        name="fourier_pattern_selected.pth",
        stage=best_stage,
        step=best_step,
        validation_fid=best_val,
        validation_baseline_fid=validation_baseline,
    )
    del optimization_images, mu_ref, sigma_ref, sigma_ref_sqrt
    torch.cuda.empty_cache()
    return {
        "validation_baseline_fid": validation_baseline,
        "validation_best_fid": best_val,
        "validation_delta": best_val - validation_baseline,
        "validation_improved": best_val < validation_baseline,
        "selected_stage": best_stage,
        "selected_pgd_step": best_step,
        "last_validated_pgd_step": last_validated_step,
        "one_shot_gradient_rms": gradient_rms,
        "one_shot_full_optimization_fid": one_shot_full_fid,
    }


def _clone_pattern_state(
    pattern: UniversalHackingPattern,
) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().clone()
        for key, value in pattern.state_dict().items()
    }


def _overfit_pattern(
    args: argparse.Namespace,
    inception: nn.Module,
    pattern: UniversalHackingPattern,
    cache_root: Path,
    wandb_logger=None,
) -> dict[str, object]:
    """Fit and select exclusively on the same fixed 50k image cache."""
    split = CachedImageSplit(cache_root, "optimization")
    optimization_images = split.load_all()
    mu_ref, sigma_ref = load_mu_and_sigma_reference(args.fid_stats_path)
    sigma_ref_sqrt = (
        precompute_sigma_ref_sqrt(sigma_ref)
        if args.fd_eigvalsh
        else None
    )
    reference = np.load(args.fid_stats_path)
    ref_mu = np.asarray(reference["mu"], dtype=np.float64)
    ref_sigma = np.asarray(reference["sigma"], dtype=np.float64)
    history: list[dict[str, object]] = []

    pattern.coeff.data.zero_()
    baseline_fid = _cached_inception_fd(
        args,
        inception,
        pattern,
        split,
        ref_mu,
        ref_sigma,
        alpha=args.hack_train_alpha,
    )
    require_nonzero = args.hack_require_nonzero_selection
    best_fid = math.inf if require_nonzero else baseline_fid
    best_state: dict[str, torch.Tensor] | None = (
        None if require_nonzero else _clone_pattern_state(pattern)
    )
    best_stage = "none" if require_nonzero else "zero"
    best_step = 0
    history.append(
        {
            "step": 0,
            "stage": "zero",
            "fid": baseline_fid,
            "delta_from_baseline": 0.0,
            "step_size": 0.0,
            "accepted": not require_nonzero,
        }
    )
    if not require_nonzero:
        _save_pattern_checkpoint(
            args,
            pattern,
            name="fourier_pattern_overfit_best.pth",
            stage=best_stage,
            step=best_step,
            validation_fid=best_fid,
            validation_baseline_fid=baseline_fid,
        )

    initial_checkpoint: dict[str, object] | None = None
    zero_joint_fid: float | None = None
    zero_gradient_rms: float | None = None
    descent_fid: float | None = None
    opposite_fid: float | None = None

    if args.hack_init_pattern_checkpoint is not None:
        # Continuation deliberately starts from the previously selected
        # direction instead of recomputing a normalized finite jump at zero.
        initial_checkpoint = _load_pattern_checkpoint(
            args.hack_init_pattern_checkpoint,
            pattern,
        )
        current_fid = _cached_inception_fd(
            args,
            inception,
            pattern,
            split,
            ref_mu,
            ref_sigma,
            alpha=args.hack_train_alpha,
        )
        current_stage = "initial_checkpoint"
        history.append(
            {
                "step": 0,
                "stage": current_stage,
                "fid": current_fid,
                "delta_from_baseline": current_fid - baseline_fid,
                "step_size": 0.0,
                "accepted": True,
                "source_checkpoint": args.hack_init_pattern_checkpoint,
                "source_train_alpha": initial_checkpoint.get("train_alpha"),
            }
        )
        if current_fid < best_fid:
            best_fid = current_fid
            best_state = _clone_pattern_state(pattern)
            best_stage = current_stage
            _save_pattern_checkpoint(
                args,
                pattern,
                name="fourier_pattern_overfit_best.pth",
                stage=best_stage,
                step=0,
                validation_fid=best_fid,
                validation_baseline_fid=baseline_fid,
            )
        logger.info(
            "[overfit continuation] baseline=%.6f initial_fid=%.6f "
            "source_alpha=%s target_alpha=%.8f best=%.6f",
            baseline_fid,
            current_fid,
            initial_checkpoint.get("train_alpha"),
            args.hack_train_alpha,
            best_fid,
        )
    else:
        # The local derivative only fixes an orientation. Since RMS
        # normalization turns it into a finite perturbation, evaluate both
        # signs on the same overfit set before iterative refinement.
        zero_joint_fid, zero_gradient = _full_cached_fd_gradient(
            args,
            inception,
            pattern,
            optimization_images,
            mu_ref,
            sigma_ref,
            sigma_ref_sqrt,
            phase_round=0,
            normalize_pattern=False,
        )
        zero_gradient_rms = float(
            zero_gradient.square().mean().sqrt().item()
        )
        if (
            not math.isfinite(zero_gradient_rms)
            or zero_gradient_rms <= 0
        ):
            raise FloatingPointError(
                f"Invalid zero-pattern gradient RMS: {zero_gradient_rms}"
            )

        with torch.no_grad():
            pattern.coeff.copy_(
                -zero_gradient
                / zero_gradient.square().mean().add(1e-12).sqrt()
            )
            pattern.normalize_coefficients_()
        descent_fid = _cached_inception_fd(
            args,
            inception,
            pattern,
            split,
            ref_mu,
            ref_sigma,
            alpha=args.hack_train_alpha,
        )
        descent_state = _clone_pattern_state(pattern)
        history.append(
            {
                "step": 0,
                "stage": "negative_gradient",
                "fid": descent_fid,
                "delta_from_baseline": descent_fid - baseline_fid,
                "step_size": 0.0,
                "accepted": False,
            }
        )

        with torch.no_grad():
            pattern.coeff.mul_(-1.0)
        opposite_fid = _cached_inception_fd(
            args,
            inception,
            pattern,
            split,
            ref_mu,
            ref_sigma,
            alpha=args.hack_train_alpha,
        )
        opposite_state = _clone_pattern_state(pattern)
        history.append(
            {
                "step": 0,
                "stage": "positive_gradient",
                "fid": opposite_fid,
                "delta_from_baseline": opposite_fid - baseline_fid,
                "step_size": 0.0,
                "accepted": False,
            }
        )

        if descent_fid <= opposite_fid:
            pattern.load_state_dict(descent_state)
            current_fid = descent_fid
            current_stage = "negative_gradient"
        else:
            pattern.load_state_dict(opposite_state)
            current_fid = opposite_fid
            current_stage = "positive_gradient"
        for row in history[-2:]:
            row["accepted"] = row["stage"] == current_stage

        if current_fid < best_fid:
            best_fid = current_fid
            best_state = _clone_pattern_state(pattern)
            best_stage = current_stage
            _save_pattern_checkpoint(
                args,
                pattern,
                name="fourier_pattern_overfit_best.pth",
                stage=best_stage,
                step=0,
                validation_fid=best_fid,
                validation_baseline_fid=baseline_fid,
            )
        logger.info(
            "[overfit] baseline=%.6f joint_zero_fid=%.6f "
            "negative_gradient=%.6f positive_gradient=%.6f selected=%s",
            baseline_fid,
            zero_joint_fid,
            descent_fid,
            opposite_fid,
            current_stage,
        )

    completed_steps = 0
    for step in range(1, args.hack_pgd_steps + 1):
        joint_fid, gradient = _full_cached_fd_gradient(
            args,
            inception,
            pattern,
            optimization_images,
            mu_ref,
            sigma_ref,
            sigma_ref_sqrt,
            phase_round=0,
            normalize_pattern=True,
        )
        gradient_rms = gradient.square().mean().add(1e-12).sqrt()
        origin_state = _clone_pattern_state(pattern)
        accepted = False
        step_size = args.hack_pgd_step_size
        candidate_fid = math.inf
        used_backtracks = 0

        for backtrack in range(args.hack_overfit_backtracks + 1):
            pattern.load_state_dict(origin_state)
            with torch.no_grad():
                pattern.coeff.add_(
                    gradient / gradient_rms,
                    alpha=-step_size,
                )
                pattern.normalize_coefficients_()
            candidate_fid = _cached_inception_fd(
                args,
                inception,
                pattern,
                split,
                ref_mu,
                ref_sigma,
                alpha=args.hack_train_alpha,
            )
            history.append(
                {
                    "step": step,
                    "stage": f"pgd_backtrack_{backtrack}",
                    "fid": candidate_fid,
                    "delta_from_baseline": candidate_fid - baseline_fid,
                    "step_size": step_size,
                    "accepted": candidate_fid
                    < current_fid
                    - args.hack_min_validation_improvement,
                    "joint_fid_before_update": joint_fid,
                    "gradient_rms": float(gradient_rms.item()),
                }
            )
            if (
                candidate_fid
                < current_fid - args.hack_min_validation_improvement
            ):
                accepted = True
                used_backtracks = backtrack
                break
            step_size *= args.hack_overfit_backtrack_factor

        if not accepted:
            pattern.load_state_dict(origin_state)
            logger.info(
                "[overfit] step=%d found no decreasing update after %d "
                "backtracks; stopping at FID=%.6f",
                step,
                args.hack_overfit_backtracks,
                current_fid,
            )
            break

        current_fid = candidate_fid
        completed_steps = step
        if current_fid < best_fid:
            best_fid = current_fid
            best_state = _clone_pattern_state(pattern)
            best_stage = "pgd"
            best_step = step
            _save_pattern_checkpoint(
                args,
                pattern,
                name="fourier_pattern_overfit_best.pth",
                stage=best_stage,
                step=best_step,
                validation_fid=best_fid,
                validation_baseline_fid=baseline_fid,
            )
        logger.info(
            "[overfit] step=%d/%d FID %.6f -> %.6f "
            "step_size=%.6g backtracks=%d best=%.6f",
            step,
            args.hack_pgd_steps,
            joint_fid,
            current_fid,
            step_size,
            used_backtracks,
            best_fid,
        )
        if wandb_logger is not None:
            wandb_logger.update(
                {
                    "hack_overfit/fid": current_fid,
                    "hack_overfit/best_fid": best_fid,
                    "hack_overfit/delta": current_fid - baseline_fid,
                    "hack_overfit/step_size": step_size,
                    "hack_overfit/backtracks": used_backtracks,
                },
                step=step,
            )

    if best_state is None:
        raise RuntimeError(
            "No nonzero pattern candidate was produced for selection"
        )
    pattern.load_state_dict(best_state)
    selected_raw_rms = float(
        pattern.patch(normalize=False).square().mean().sqrt().item()
    )
    selected_nonzero = (
        math.isfinite(selected_raw_rms) and selected_raw_rms > 1e-6
    )
    if require_nonzero and not selected_nonzero:
        raise RuntimeError(
            "Nonzero checkpoint selection produced a zero pattern"
        )
    _save_pattern_checkpoint(
        args,
        pattern,
        name="fourier_pattern_overfit_selected.pth",
        stage=best_stage,
        step=best_step,
        validation_fid=best_fid,
        validation_baseline_fid=baseline_fid,
    )

    dose_rows: list[dict[str, float | int]] = []
    dose_values: list[float] = []
    for alpha in args.hack_eval_alphas:
        value = _cached_inception_fd(
            args,
            inception,
            pattern,
            split,
            ref_mu,
            ref_sigma,
            alpha=alpha,
        )
        dose_values.append(value)
        dose_rows.append(
            {
                "alpha": alpha,
                "inception_fid": value,
                "num_images": args.hack_optimization_images,
            }
        )
        logger.info(
            "[overfit dose] alpha=%.8f inception_fid=%.6f",
            alpha,
            value,
        )
    beta_inception = _linear_slope(
        list(args.hack_eval_alphas), dose_values
    )
    improvement_tolerance = args.hack_min_validation_improvement
    alpha_span = (
        args.hack_eval_alphas[-1] - args.hack_eval_alphas[0]
    )
    slope_tolerance = improvement_tolerance / max(alpha_span, 1e-12)
    dose_best_index = int(np.argmin(dose_values))
    dose_best_fid = dose_values[dose_best_index]
    fit_success = (
        selected_nonzero
        and best_fid < baseline_fid - improvement_tolerance
    )
    dose_response_success = (
        fit_success
        and dose_best_fid < baseline_fid - improvement_tolerance
        and beta_inception < -slope_tolerance
    )
    summary: dict[str, object] = {
        "mode": "overfit_only",
        "optimization_images": args.hack_optimization_images,
        "optimization_seed": args.hack_optimization_seed,
        "train_alpha": args.hack_train_alpha,
        "baseline_fid": baseline_fid,
        "best_fid": best_fid,
        "best_delta": best_fid - baseline_fid,
        "fit_success": fit_success,
        "success_fid_tolerance": improvement_tolerance,
        "selected_stage": best_stage,
        "selected_step": best_step,
        "require_nonzero_selection": require_nonzero,
        "selected_pattern_nonzero": selected_nonzero,
        "selected_pattern_raw_rms": selected_raw_rms,
        "selected_applied_model_rms": args.hack_train_alpha,
        "selected_applied_pixel_rms": args.hack_train_alpha / 2.0,
        "completed_pgd_steps": completed_steps,
        "zero_gradient_rms": zero_gradient_rms,
        "negative_gradient_fid": descent_fid,
        "positive_gradient_fid": opposite_fid,
        "initial_pattern_checkpoint": args.hack_init_pattern_checkpoint,
        "initial_pattern_train_alpha": (
            initial_checkpoint.get("train_alpha")
            if initial_checkpoint is not None
            else None
        ),
        "alphas": list(args.hack_eval_alphas),
        "dose_response_beta_inception": beta_inception,
        "dose_response_slope_tolerance": slope_tolerance,
        "dose_response_best_fid": dose_best_fid,
        "dose_response_best_alpha": args.hack_eval_alphas[
            dose_best_index
        ],
        "dose_response_success": dose_response_success,
        "random_phase": args.hack_random_phase,
    }
    if get_global_rank() == 0:
        history_path = Path(args.log_dir) / "overfit_history.csv"
        history_fields: list[str] = []
        for row in history:
            for key in row:
                if key not in history_fields:
                    history_fields.append(key)
        with history_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=history_fields)
            writer.writeheader()
            writer.writerows(history)
        dose_path = Path(args.log_dir) / "overfit_dose_response.csv"
        with dose_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["alpha", "inception_fid", "num_images"],
            )
            writer.writeheader()
            writer.writerows(dose_rows)
        summary_path = Path(args.log_dir) / "overfit_summary.json"
        with summary_path.open("w") as handle:
            json.dump(summary, handle, indent=2)
        logger.info("Overfit history: %s", history_path)
        logger.info("Overfit dose response: %s", dose_path)
        logger.info("Overfit summary: %s", summary_path)
        logger.info(
            "[overfit result] baseline=%.6f best=%.6f delta=%+.6f "
            "beta_inc=%.6g fit_success=%s dose_success=%s",
            baseline_fid,
            best_fid,
            best_fid - baseline_fid,
            beta_inception,
            summary["fit_success"],
            summary["dose_response_success"],
        )

    del optimization_images, mu_ref, sigma_ref, sigma_ref_sqrt
    torch.cuda.empty_cache()
    return summary


def _empty_eval_totals(
    metric_dims: dict[str, int],
    alphas: list[float],
) -> dict[str, dict[float, dict[str, np.ndarray | int]]]:
    return {
        metric: {
            alpha: {
                "sum": np.zeros(dim, dtype=np.float64),
                "outer": np.zeros((dim, dim), dtype=np.float64),
                "count": 0,
            }
            for alpha in alphas
        }
        for metric, dim in metric_dims.items()
    }


def _reduce_eval_block(
    accumulators: dict[str, dict[float, Moments]],
    totals: dict[str, dict[float, dict[str, np.ndarray | int]]],
    references: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, dict[float, float]]:
    results = {metric: {} for metric in accumulators}
    for metric, alpha_moments in accumulators.items():
        ref_mu, ref_sigma = references[metric]
        for alpha, moments in alpha_moments.items():
            count = torch.tensor(
                [moments.count], device="cuda", dtype=torch.int64
            )
            if is_enabled():
                dist.reduce(
                    moments.feat_sum, dst=0, op=dist.ReduceOp.SUM
                )
                dist.reduce(
                    moments.feat_outer, dst=0, op=dist.ReduceOp.SUM
                )
                dist.reduce(count, dst=0, op=dist.ReduceOp.SUM)
            if get_global_rank() == 0:
                feat_sum = moments.feat_sum.cpu().numpy().copy()
                feat_outer = moments.feat_outer.cpu().numpy().copy()
                global_count = int(count.item())
                total = totals[metric][alpha]
                total["sum"] += feat_sum
                total["outer"] += feat_outer
                total["count"] = int(total["count"]) + global_count
                results[metric][alpha] = _stats_to_fd(
                    feat_sum,
                    feat_outer,
                    global_count,
                    ref_mu,
                    ref_sigma,
                )
    return results


@torch.no_grad()
def _evaluate_test(
    args: argparse.Namespace,
    inception: nn.Module,
    inception_dim: int,
    pattern: UniversalHackingPattern,
    cache_root: Path,
    optimization_summary: dict[str, object],
) -> dict[str, object]:
    # This is deliberately the first point at which CLIP is loaded.
    logger.info(
        "[final test] pattern is selected; now loading frozen CLIP %s",
        args.hack_clip_model,
    )
    clip, clip_dim = _load_frozen_repr(
        args.hack_clip_model,
        target_size=args.hack_clip_target_size,
    )
    inception_ref_file = np.load(args.fid_stats_path)
    clip_ref_file = np.load(args.hack_clip_stats_path)
    references = {
        "inception": (
            np.asarray(inception_ref_file["mu"], dtype=np.float64),
            np.asarray(inception_ref_file["sigma"], dtype=np.float64),
        ),
        "clip": (
            np.asarray(clip_ref_file["mu"], dtype=np.float64),
            np.asarray(clip_ref_file["sigma"], dtype=np.float64),
        ),
    }
    if references["inception"][0].shape[0] != inception_dim:
        raise ValueError("Inception reference-stat dimension mismatch")
    if references["clip"][0].shape[0] != clip_dim:
        raise ValueError("CLIP reference-stat dimension mismatch")

    alphas = list(args.hack_eval_alphas)
    split = CachedImageSplit(cache_root, "test")
    if split.count % args.hack_eval_blocks:
        raise ValueError(
            "Each rank's cached test shard must divide evenly into eval blocks"
        )
    local_block_count = split.count // args.hack_eval_blocks
    metric_dims = {"inception": inception_dim, "clip": clip_dim}
    totals = _empty_eval_totals(metric_dims, alphas)
    block_slopes: list[dict[str, float]] = []
    started = time.perf_counter()

    for block in range(args.hack_eval_blocks):
        accumulators = {
            metric: {
                alpha: Moments.create(dim, torch.device("cuda"))
                for alpha in alphas
            }
            for metric, dim in metric_dims.items()
        }
        start = block * local_block_count
        end = start + local_block_count
        for cached_images in split.iter_range(
            start, end, args.eval_bsz
        ):
            base = _decode_cached(cached_images, torch.device("cuda"))
            for alpha in alphas:
                images = _to_unit_range(
                    _apply_fourier_pattern(
                        base,
                        pattern,
                        alpha=alpha,
                        phase=(0, 0),
                        normalize=True,
                    )
                )
                inc_features = _extract_primary_features(
                    inception,
                    images,
                    use_amp=False,
                    amp_dtype=args.amp_dtype,
                )
                clip_features = _extract_primary_features(
                    clip,
                    images,
                    use_amp=args.enable_amp,
                    amp_dtype=args.amp_dtype,
                )
                accumulators["inception"][alpha].update(inc_features)
                accumulators["clip"][alpha].update(clip_features)
                del images, inc_features, clip_features
            del cached_images, base

        block_values = _reduce_eval_block(
            accumulators, totals, references
        )
        if get_global_rank() == 0:
            inc_values = [
                block_values["inception"][alpha] for alpha in alphas
            ]
            clip_fdr_values = [
                block_values["clip"][alpha]
                / args.hack_clip_fdr_denominator
                for alpha in alphas
            ]
            slopes = {
                "beta_inception": _linear_slope(alphas, inc_values),
                "beta_clip_fdr": _linear_slope(
                    alphas, clip_fdr_values
                ),
            }
            block_slopes.append(slopes)
            logger.info(
                "[final test] block=%d/%d beta_inc=%.6g "
                "beta_clip_fdr=%.6g elapsed=%.1fs",
                block + 1,
                args.hack_eval_blocks,
                slopes["beta_inception"],
                slopes["beta_clip_fdr"],
                time.perf_counter() - started,
            )
        del accumulators
        _barrier()

    rows: list[dict[str, float | int]] = []
    summary: dict[str, object] = {}
    if get_global_rank() == 0:
        final_values: dict[str, list[float]] = {
            "inception": [],
            "clip": [],
        }
        for metric in ("inception", "clip"):
            ref_mu, ref_sigma = references[metric]
            for alpha in alphas:
                total = totals[metric][alpha]
                final_values[metric].append(
                    _stats_to_fd(
                        total["sum"],
                        total["outer"],
                        int(total["count"]),
                        ref_mu,
                        ref_sigma,
                    )
                )
        clip_fdr = [
            value / args.hack_clip_fdr_denominator
            for value in final_values["clip"]
        ]
        for alpha, inc_fd, clip_fd, clip_ratio in zip(
            alphas,
            final_values["inception"],
            final_values["clip"],
            clip_fdr,
        ):
            rows.append(
                {
                    "alpha": alpha,
                    "inception_fid": inc_fd,
                    "clip_fd": clip_fd,
                    "clip_fdr": clip_ratio,
                    "num_images": args.hack_test_images,
                }
            )
        beta_inc = _linear_slope(alphas, final_values["inception"])
        beta_clip = _linear_slope(alphas, final_values["clip"])
        beta_clip_fdr = _linear_slope(alphas, clip_fdr)
        inc_ci = _bootstrap_mean_ci(
            [row["beta_inception"] for row in block_slopes],
            repeats=args.hack_bootstrap_repeats,
            confidence=args.hack_confidence,
            seed=args.hack_bootstrap_seed,
            point_estimate=beta_inc,
        )
        clip_ci = _bootstrap_mean_ci(
            [row["beta_clip_fdr"] for row in block_slopes],
            repeats=args.hack_bootstrap_repeats,
            confidence=args.hack_confidence,
            seed=args.hack_bootstrap_seed + 1,
            point_estimate=beta_clip_fdr,
        )
        conflict = max(0.0, -beta_inc) * max(0.0, beta_clip_fdr)
        summary = {
            **optimization_summary,
            "beta_inception": beta_inc,
            "beta_clip": beta_clip,
            "beta_clip_fdr": beta_clip_fdr,
            "beta_inception_ci": inc_ci,
            "beta_clip_fdr_ci": clip_ci,
            "conflict": conflict,
            "success": beta_inc < 0 and beta_clip_fdr > 0,
            "ci_sign_success": (
                inc_ci is not None
                and clip_ci is not None
                and inc_ci[1] < 0
                and clip_ci[0] > 0
            ),
            "alphas": alphas,
            "test_images": args.hack_test_images,
            "test_seed": args.hack_test_seed,
            "eval_blocks": args.hack_eval_blocks,
            "clip_model": args.hack_clip_model,
            "clip_fdr_denominator": args.hack_clip_fdr_denominator,
            "inception_stats": args.fid_stats_path,
            "clip_stats": args.hack_clip_stats_path,
        }
        csv_path = Path(args.log_dir) / "fourier_dose_response.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "alpha",
                    "inception_fid",
                    "clip_fd",
                    "clip_fdr",
                    "num_images",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        summary_path = Path(args.log_dir) / "fourier_summary.json"
        with summary_path.open("w") as handle:
            json.dump(summary, handle, indent=2)
        logger.info("Final dose response: %s", csv_path)
        logger.info("Final summary: %s", summary_path)
        logger.info(
            "Result beta_inc=%.6g CI=%s beta_clip_fdr=%.6g CI=%s "
            "conflict=%.6g success=%s",
            beta_inc,
            inc_ci,
            beta_clip_fdr,
            clip_ci,
            conflict,
            summary["success"],
        )
    del clip, totals
    torch.cuda.empty_cache()
    return summary


def _validate_args(args: argparse.Namespace) -> None:
    if not args.model.startswith("pMF_"):
        raise ValueError("This hacking entry point supports pMF models only")
    if args.tokenizer is not None:
        raise ValueError("Pixel-space pMF caching requires --tokenizer None")
    if not args.load_from:
        raise ValueError("--load_from is required")
    expected_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    split_counts = (
        (args.hack_optimization_images,)
        if args.hack_overfit_only
        else (
            args.hack_optimization_images,
            args.hack_validation_images,
            args.hack_test_images,
        )
    )
    if any(count < 2 or count % expected_world_size for count in split_counts):
        raise ValueError(
            "Every cache split must contain at least two images and be "
            "divisible by WORLD_SIZE"
        )
    if (
        args.hack_gradient_batch_size < 2
        or args.hack_gradient_batch_size % expected_world_size
    ):
        raise ValueError(
            "--hack_gradient_batch_size must be >=2 and divisible by WORLD_SIZE"
        )
    if args.hack_pgd_steps < 0:
        raise ValueError("--hack_pgd_steps must be non-negative")
    if args.hack_validate_every < 1:
        raise ValueError("--hack_validate_every must be positive")
    if args.hack_pgd_step_size <= 0:
        raise ValueError("--hack_pgd_step_size must be positive")
    if args.hack_cov_eps < 0:
        raise ValueError("--hack_cov_eps must be non-negative")
    if not (
        0
        < args.hack_fourier_min_radius
        < args.hack_fourier_max_radius
        < math.sqrt(0.5)
    ):
        raise ValueError("Invalid Fourier radius band")
    if args.hack_train_alpha <= 0:
        raise ValueError("--hack_train_alpha must be positive")
    if (
        not args.hack_eval_alphas
        or args.hack_eval_alphas[0] != 0
        or any(
            not math.isfinite(alpha) or alpha < 0
            for alpha in args.hack_eval_alphas
        )
        or args.hack_eval_alphas
        != sorted(set(args.hack_eval_alphas))
    ):
        raise ValueError(
            "--hack_eval_alphas must be unique, sorted, and start at zero"
        )
    if args.hack_eval_blocks < 1:
        raise ValueError("--hack_eval_blocks must be positive")
    if not args.hack_overfit_only:
        if args.hack_test_images % args.hack_eval_blocks:
            raise ValueError(
                "--hack_test_images must be divisible by --hack_eval_blocks"
            )
        if (
            args.hack_test_images // expected_world_size
        ) % args.hack_eval_blocks:
            raise ValueError(
                "Each rank's test shard must divide evenly into eval blocks"
            )
    if not 0 < args.hack_confidence < 1:
        raise ValueError("--hack_confidence must be in (0,1)")
    if args.hack_clip_fdr_denominator <= 0:
        raise ValueError("--hack_clip_fdr_denominator must be positive")
    if args.hack_bootstrap_repeats < 1:
        raise ValueError("--hack_bootstrap_repeats must be positive")
    if args.hack_cache_only and args.hack_eval_only:
        raise ValueError(
            "--hack_cache_only and --hack_eval_only are mutually exclusive"
        )
    if args.hack_overfit_only and args.hack_eval_only:
        raise ValueError(
            "--hack_overfit_only and --hack_eval_only are mutually exclusive"
        )
    if (
        args.hack_init_pattern_checkpoint is not None
        and not args.hack_overfit_only
    ):
        raise ValueError(
            "--hack_init_pattern_checkpoint currently requires "
            "--hack_overfit_only"
        )
    if args.hack_overfit_backtracks < 0:
        raise ValueError("--hack_overfit_backtracks must be non-negative")
    if not 0 < args.hack_overfit_backtrack_factor < 1:
        raise ValueError(
            "--hack_overfit_backtrack_factor must be in (0,1)"
        )


def get_args_parser() -> argparse.ArgumentParser:
    from main_fd import get_args_parser as get_parent_parser

    parser = argparse.ArgumentParser(
        "Cached Fourier universal pMF hacking",
        parents=[get_parent_parser()],
        add_help=True,
        conflict_handler="resolve",
    )
    parser.add_argument(
        "--hack_cache_dir",
        type=str,
        default="./work_dirs/hacking_cache/pMF",
    )
    parser.add_argument(
        "--hack_cache_dtype",
        choices=["uint8", "float16"],
        default="uint8",
        help=(
            "uint8 is ~20.6GB for the default 105k 256px images; "
            "float16 is lossless-ish but 2x larger"
        ),
    )
    parser.add_argument("--hack_overwrite_cache", action="store_true")
    parser.add_argument("--hack_optimization_images", type=int, default=50_000)
    parser.add_argument("--hack_validation_images", type=int, default=5_000)
    parser.add_argument("--hack_test_images", type=int, default=50_000)
    parser.add_argument("--hack_optimization_seed", type=int, default=12_345)
    parser.add_argument("--hack_validation_seed", type=int, default=23_456)
    parser.add_argument("--hack_test_seed", type=int, default=34_567)
    parser.add_argument("--hack_pattern_size", type=int, default=16)
    parser.add_argument(
        "--hack_pattern_parameterization",
        choices=["fourier", "spatial_bandpass"],
        default="fourier",
        help=(
            "spatial_bandpass learns one full-resolution universal noise "
            "while projecting it to the configured Fourier radius band"
        ),
    )
    parser.add_argument("--hack_fourier_modes", type=int, default=48)
    parser.add_argument(
        "--hack_fourier_min_radius", type=float, default=0.15
    )
    parser.add_argument(
        "--hack_fourier_max_radius", type=float, default=0.55
    )
    parser.add_argument(
        "--hack_train_alpha", type=float, default=8 / 255
    )
    parser.add_argument(
        "--hack_gradient_batch_size",
        type=int,
        default=256,
        help=(
            "global streaming batch size; every FD gradient still uses the "
            "complete optimization split"
        ),
    )
    parser.add_argument("--hack_phase_seed", type=int, default=56_789)
    parser.add_argument(
        "--hack_random_phase",
        action="store_true",
        dest="hack_random_phase",
    )
    parser.add_argument(
        "--hack_no_random_phase",
        action="store_false",
        dest="hack_random_phase",
    )
    parser.set_defaults(hack_random_phase=True)
    parser.add_argument(
        "--hack_cov_eps",
        type=float,
        default=0.0,
        help=(
            "optional fake-covariance diagonal stabilization; the default "
            "full-50k objective matches ordinary FID without it"
        ),
    )
    parser.add_argument("--hack_pgd_steps", type=int, default=0)
    parser.add_argument("--hack_pgd_step_size", type=float, default=0.25)
    parser.add_argument("--hack_validate_every", type=int, default=5)
    parser.add_argument(
        "--hack_early_stop_patience", type=int, default=2
    )
    parser.add_argument(
        "--hack_min_validation_improvement", type=float, default=1e-4
    )
    parser.add_argument(
        "--hack_eval_alphas",
        type=float,
        nargs="+",
        default=[0.0, 2 / 255, 4 / 255, 8 / 255, 16 / 255],
    )
    parser.add_argument("--hack_eval_blocks", type=int, default=10)
    parser.add_argument(
        "--hack_clip_model",
        type=str,
        default="vit_large_patch14_clip_224.openai",
    )
    parser.add_argument("--hack_clip_target_size", type=int, default=256)
    parser.add_argument("--hack_clip_stats_path", type=str, default=None)
    parser.add_argument(
        "--hack_clip_fdr_denominator", type=float, default=5.60
    )
    parser.add_argument(
        "--hack_bootstrap_repeats", type=int, default=10_000
    )
    parser.add_argument("--hack_bootstrap_seed", type=int, default=20_21)
    parser.add_argument("--hack_confidence", type=float, default=0.95)
    parser.add_argument("--hack_cache_only", action="store_true")
    parser.add_argument("--hack_eval_only", action="store_true")
    parser.add_argument(
        "--hack_overfit_only",
        action="store_true",
        help=(
            "use only the fixed optimization split for fitting, checkpoint "
            "selection, and Inception dose-response debugging"
        ),
    )
    parser.add_argument(
        "--hack_overfit_backtracks",
        type=int,
        default=5,
        help="maximum PGD step halvings on the same 50k optimization set",
    )
    parser.add_argument(
        "--hack_overfit_backtrack_factor",
        type=float,
        default=0.5,
    )
    parser.add_argument("--hack_skip_final_eval", action="store_true")
    parser.add_argument("--hack_pattern_checkpoint", type=str, default=None)
    parser.add_argument(
        "--hack_init_pattern_checkpoint",
        type=str,
        default=None,
        help=(
            "initialize --hack_overfit_only refinement from a compatible "
            "selected Fourier-pattern checkpoint"
        ),
    )
    parser.add_argument(
        "--hack_require_nonzero_selection",
        action="store_true",
        help=(
            "exclude the clean zero pattern from checkpoint selection; "
            "success still requires the selected nonzero pattern to beat "
            "the clean 50k baseline"
        ),
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    from utils.setup_util import setup

    if args.hack_overfit_only:
        # A strict overfit/debug run must optimize and evaluate the identical
        # transformation. Random phase belongs to the later generalization
        # experiment, not this optimizer sanity check.
        args.hack_random_phase = False
    _validate_args(args)
    wandb_logger = setup(args)
    cache_root = _ensure_image_cache(args)
    if args.hack_cache_only:
        logger.info("Cache-only run complete")
        if wandb_logger is not None:
            wandb_logger.finish()
        return {"cache_dir": str(cache_root)}
    if not Path(args.fid_stats_path).exists():
        raise FileNotFoundError(
            f"Inception reference stats not found: {args.fid_stats_path}"
        )

    inception, inception_dim = _load_frozen_repr(
        "inception", target_size=299
    )
    if args.hack_pattern_parameterization == "fourier":
        pattern: UniversalHackingPattern = FourierPattern(
            args.hack_pattern_size,
            args.hack_fourier_modes,
            args.hack_fourier_min_radius,
            args.hack_fourier_max_radius,
        ).cuda()
        logger.info(
            "Fourier pattern: size=%d modes=%d trainable=%d "
            "frequencies=%s",
            pattern.size,
            len(pattern.frequencies),
            pattern.coeff.numel(),
            pattern.frequencies,
        )
    else:
        pattern = SpatialBandpassPattern(
            args.hack_pattern_size,
            args.hack_fourier_min_radius,
            args.hack_fourier_max_radius,
        ).cuda()
        logger.info(
            "Spatial bandpass pattern: size=%d trainable=%d "
            "radius=[%.4f, %.4f]",
            pattern.size,
            pattern.coeff.numel(),
            args.hack_fourier_min_radius,
            args.hack_fourier_max_radius,
        )

    optimization_summary: dict[str, object]
    if args.hack_overfit_only:
        optimization_summary = _overfit_pattern(
            args,
            inception,
            pattern,
            cache_root,
            wandb_logger=wandb_logger,
        )
        _save_pattern_preview(args, pattern)
        if wandb_logger is not None and get_global_rank() == 0:
            wandb_logger.update(
                {
                    f"hack_overfit/final_{key}": value
                    for key, value in optimization_summary.items()
                    if isinstance(value, (int, float, bool))
                },
                step=int(
                    optimization_summary.get("selected_step", 0) or 0
                ),
            )
        _barrier()
        if wandb_logger is not None:
            wandb_logger.finish()
        return optimization_summary
    if args.hack_eval_only:
        checkpoint_path = args.hack_pattern_checkpoint
        if checkpoint_path is None:
            checkpoint_path = str(
                Path(args.ckpt_dir) / "fourier_pattern_selected.pth"
            )
        checkpoint = _load_pattern_checkpoint(checkpoint_path, pattern)
        optimization_summary = {
            "validation_baseline_fid": checkpoint.get(
                "validation_baseline_fid"
            ),
            "validation_best_fid": checkpoint.get("validation_fid"),
            "selected_stage": checkpoint.get("stage"),
            "selected_pgd_step": checkpoint.get("step"),
        }
    else:
        optimization_summary = _optimize_pattern(
            args,
            inception,
            pattern,
            cache_root,
            wandb_logger=wandb_logger,
        )
    _save_pattern_preview(args, pattern)

    summary = optimization_summary
    if not args.hack_skip_final_eval:
        if args.hack_clip_stats_path is None:
            args.hack_clip_stats_path = infer_stats_path(
                args.hack_clip_model,
                args.img_size,
                args.hack_clip_target_size,
            )
        if not Path(args.hack_clip_stats_path).exists():
            raise FileNotFoundError(
                f"CLIP reference stats not found: "
                f"{args.hack_clip_stats_path}"
            )
        summary = _evaluate_test(
            args,
            inception,
            inception_dim,
            pattern,
            cache_root,
            optimization_summary,
        )
        if wandb_logger is not None and get_global_rank() == 0:
            wandb_logger.update(
                {
                    f"hack/final_{key}": value
                    for key, value in summary.items()
                    if isinstance(value, (int, float, bool))
                },
                step=int(
                    optimization_summary.get("selected_pgd_step", 0) or 0
                ),
            )
    _barrier()
    if wandb_logger is not None:
        wandb_logger.finish()
    return summary


def _cleanup_distributed() -> None:
    if is_enabled():
        dist.destroy_process_group()


if __name__ == "__main__":
    parsed_args = get_args_parser().parse_args()
    try:
        run(parsed_args)
    finally:
        _cleanup_distributed()
