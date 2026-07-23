"""Fast universal Inception-FD hacking experiment for frozen pMF generators.

The experiment has three strictly separated cached splits:

* optimization: estimate a low-dimensional Fourier descent direction;
* validation: select/early-stop using Inception FD only;
* test: report the final Inception and CLIP dose response.

The generator is used only when a cache shard is missing.  During pattern
optimization the generator is unloaded, Inception is frozen, and CLIP is not
loaded.  The only trainable object is a small set of real Fourier
coefficients shared by every image.
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
    diff_all_gather,
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
        "splits": {
            "optimization": {
                "images": args.hack_optimization_images,
                "seed": args.hack_optimization_seed,
            },
            "validation": {
                "images": args.hack_validation_images,
                "seed": args.hack_validation_seed,
            },
            "test": {
                "images": args.hack_test_images,
                "seed": args.hack_test_seed,
            },
        },
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

    manifest_matches = False
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
    if manifest_path.exists() and not manifest_matches:
        if not args.hack_overwrite_cache:
            raise ValueError(
                f"Cache manifest {manifest_path} does not match this run. "
                "Use a different --hack_cache_dir or pass "
                "--hack_overwrite_cache to rewrite explicit cache shards."
            )
        logger.warning("Cache manifest mismatch; rewriting this run's shards")

    split_specs = spec["splits"]
    assert isinstance(split_specs, dict)
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
    pattern: FourierPattern,
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


class FixedBatchSampler:
    """Deterministic shuffled sampler shared in structure across ranks."""

    def __init__(self, count: int, batch_size: int, seed: int):
        if batch_size > count:
            raise ValueError(
                f"Local gradient batch {batch_size} exceeds cache shard {count}"
            )
        self.count = count
        self.batch_size = batch_size
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)
        self.permutation = torch.randperm(count, generator=self.generator)
        self.cursor = 0

    def next(self) -> torch.Tensor:
        if self.cursor + self.batch_size > self.count:
            self.permutation = torch.randperm(
                self.count, generator=self.generator
            )
            self.cursor = 0
        indices = self.permutation[
            self.cursor : self.cursor + self.batch_size
        ]
        self.cursor += self.batch_size
        return indices


def _batch_fd_gradient(
    args: argparse.Namespace,
    inception: nn.Module,
    pattern: FourierPattern,
    cached_images: torch.Tensor,
    mu_ref: torch.Tensor,
    sigma_ref: torch.Tensor,
    sigma_ref_sqrt: torch.Tensor | None,
    *,
    phase: tuple[int, int],
    normalize_pattern: bool,
) -> tuple[float, torch.Tensor]:
    images = _decode_cached(cached_images, torch.device("cuda"))
    perturbed = _to_unit_range(
        _apply_fourier_pattern(
            images,
            pattern,
            alpha=args.hack_train_alpha,
            phase=phase,
            normalize=normalize_pattern,
        )
    )
    local_features = _extract_primary_features(
        inception,
        perturbed,
        use_amp=False,
        amp_dtype=args.amp_dtype,
    )
    global_features = diff_all_gather(local_features)
    mu = global_features.mean(dim=0)
    centered = global_features - mu
    sigma = centered.T @ centered / (global_features.shape[0] - 1)
    if args.hack_cov_eps > 0:
        sigma = sigma + args.hack_cov_eps * torch.eye(
            sigma.shape[0], device=sigma.device, dtype=sigma.dtype
        )
    fid = compute_frechet_distance_loss(
        mu_ref,
        sigma_ref,
        mu=mu,
        sigma=sigma,
        sigma_ref_sqrt=sigma_ref_sqrt,
    )
    gradient = torch.autograd.grad(
        fid, pattern.coeff, create_graph=False
    )[0].detach()
    _all_reduce_sum_(gradient)
    if not torch.isfinite(gradient).all():
        raise FloatingPointError(
            "Non-finite Fourier gradient; increase --hack_cov_eps"
        )
    value = float(fid.detach())
    del images, perturbed, local_features, global_features, mu, centered, sigma
    return value, gradient


@torch.no_grad()
def _cached_inception_fd(
    args: argparse.Namespace,
    inception: nn.Module,
    pattern: FourierPattern,
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
    pattern: FourierPattern,
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
    payload = {
        "pattern": pattern.state_dict(),
        "pattern_size": pattern.size,
        "frequencies": [list(pair) for pair in pattern.frequencies],
        "fourier_modes": len(pattern.frequencies),
        "fourier_min_radius": args.hack_fourier_min_radius,
        "fourier_max_radius": args.hack_fourier_max_radius,
        "stage": stage,
        "step": step,
        "validation_fid": validation_fid,
        "validation_baseline_fid": validation_baseline_fid,
        "train_alpha": args.hack_train_alpha,
        "generator_checkpoint": args.load_from,
    }
    torch.save(payload, path)
    logger.info("Saved Fourier pattern checkpoint: %s", path)
    return path


def _load_pattern_checkpoint(
    path: str | Path,
    pattern: FourierPattern,
) -> dict[str, object]:
    checkpoint = torch.load(
        path, map_location="cuda", weights_only=False
    )
    expected = [list(pair) for pair in pattern.frequencies]
    if checkpoint.get("frequencies") != expected:
        raise ValueError(
            "Checkpoint Fourier frequencies do not match current arguments"
        )
    pattern.load_state_dict(checkpoint["pattern"])
    logger.info(
        "Loaded Fourier pattern %s (stage=%s step=%s validation_fid=%s)",
        path,
        checkpoint.get("stage"),
        checkpoint.get("step"),
        checkpoint.get("validation_fid"),
    )
    return checkpoint


@torch.no_grad()
def _save_pattern_preview(
    args: argparse.Namespace,
    pattern: FourierPattern,
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


def _average_gradient(
    args: argparse.Namespace,
    inception: nn.Module,
    pattern: FourierPattern,
    optimization_images: torch.Tensor,
    sampler: FixedBatchSampler,
    mu_ref: torch.Tensor,
    sigma_ref: torch.Tensor,
    sigma_ref_sqrt: torch.Tensor | None,
    *,
    batches: int,
    sequence_start: int,
    normalize_pattern: bool,
) -> tuple[float, torch.Tensor]:
    gradient_sum = torch.zeros_like(pattern.coeff)
    fid_sum = 0.0
    for offset in range(batches):
        indices = sampler.next()
        phase = _phase_for(
            args.hack_phase_seed,
            sequence_start + offset,
            pattern.size,
            args.hack_random_phase,
        )
        fid, gradient = _batch_fd_gradient(
            args,
            inception,
            pattern,
            optimization_images[indices],
            mu_ref,
            sigma_ref,
            sigma_ref_sqrt,
            phase=phase,
            normalize_pattern=normalize_pattern,
        )
        gradient_sum.add_(gradient)
        fid_sum += fid
        logger.info(
            "[gradient] batch=%d/%d phase=%s raw_batch_fid=%.6f",
            offset + 1,
            batches,
            phase,
            fid,
        )
    return fid_sum / batches, gradient_sum / batches


def _optimize_pattern(
    args: argparse.Namespace,
    inception: nn.Module,
    pattern: FourierPattern,
    cache_root: Path,
    wandb_logger=None,
) -> dict[str, object]:
    optimization_split = CachedImageSplit(cache_root, "optimization")
    validation_split = CachedImageSplit(cache_root, "validation")
    optimization_images = optimization_split.load_all()
    local_gradient_bsz = args.hack_gradient_batch_size // get_world_size()
    sampler = FixedBatchSampler(
        optimization_split.count,
        local_gradient_bsz,
        args.hack_gradient_seed,
    )

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
    one_shot_batch_fid, one_shot_gradient = _average_gradient(
        args,
        inception,
        pattern,
        optimization_images,
        sampler,
        mu_ref,
        sigma_ref,
        sigma_ref_sqrt,
        batches=args.hack_gradient_batches,
        sequence_start=0,
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
        "[one-shot] mean_batch_fid=%.6f gradient_rms=%.6g "
        "validation_fid=%.6f delta_vs_zero=%+.6f elapsed=%.1fs",
        one_shot_batch_fid,
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
            },
            step=0,
        )

    checks_without_improvement = 0
    sequence = args.hack_gradient_batches
    last_validated_step = 0
    for step in range(1, args.hack_pgd_steps + 1):
        mean_batch_fid, gradient = _average_gradient(
            args,
            inception,
            pattern,
            optimization_images,
            sampler,
            mu_ref,
            sigma_ref,
            sigma_ref_sqrt,
            batches=args.hack_pgd_batches_per_step,
            sequence_start=sequence,
            normalize_pattern=True,
        )
        sequence += args.hack_pgd_batches_per_step
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
            "[PGD] step=%d/%d mean_batch_fid=%.6f validation_fid=%.6f "
            "best=%.6f improved=%s patience=%d/%d",
            step,
            args.hack_pgd_steps,
            mean_batch_fid,
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
                    "hack/pgd_batch_fid": mean_batch_fid,
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
        "one_shot_mean_batch_fid": one_shot_batch_fid,
    }


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
    pattern: FourierPattern,
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
        args.hack_optimization_images,
        args.hack_validation_images,
        args.hack_test_images,
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
    if args.hack_gradient_batches < 1:
        raise ValueError("--hack_gradient_batches must be positive")
    if args.hack_pgd_steps < 0 or args.hack_pgd_batches_per_step < 1:
        raise ValueError("Invalid PGD step/batch configuration")
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
        help="uint8 is ~12GB for 60k 256px images; float16 is lossless-ish but 2x larger",
    )
    parser.add_argument("--hack_overwrite_cache", action="store_true")
    parser.add_argument("--hack_optimization_images", type=int, default=5_000)
    parser.add_argument("--hack_validation_images", type=int, default=5_000)
    parser.add_argument("--hack_test_images", type=int, default=50_000)
    parser.add_argument("--hack_optimization_seed", type=int, default=12_345)
    parser.add_argument("--hack_validation_seed", type=int, default=23_456)
    parser.add_argument("--hack_test_seed", type=int, default=34_567)
    parser.add_argument("--hack_pattern_size", type=int, default=16)
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
        help="global batch size used by each averaged FD gradient",
    )
    parser.add_argument("--hack_gradient_batches", type=int, default=16)
    parser.add_argument("--hack_gradient_seed", type=int, default=45_678)
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
    parser.add_argument("--hack_cov_eps", type=float, default=1e-4)
    parser.add_argument("--hack_pgd_steps", type=int, default=20)
    parser.add_argument(
        "--hack_pgd_batches_per_step", type=int, default=1
    )
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
    parser.add_argument("--hack_skip_final_eval", action="store_true")
    parser.add_argument("--hack_pattern_checkpoint", type=str, default=None)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    from utils.setup_util import setup

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
    pattern = FourierPattern(
        args.hack_pattern_size,
        args.hack_fourier_modes,
        args.hack_fourier_min_radius,
        args.hack_fourier_max_radius,
    ).cuda()
    logger.info(
        "Fourier pattern: size=%d modes=%d trainable=%d frequencies=%s",
        pattern.size,
        len(pattern.frequencies),
        pattern.coeff.numel(),
        pattern.frequencies,
    )

    optimization_summary: dict[str, object]
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
