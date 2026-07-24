"""Extract a deterministic natural-spectrum RGB pattern from images.

This utility performs no metric optimization and has no trainable parameters.
It streams a fixed generated-image cache or flat image directory once,
estimates the RGB cross-spectral density of high-pass residuals, and uses a
seeded closed-form Gaussian synthesis to produce one zero-mean, unit-RMS
spatial pattern.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


logger = logging.getLogger("spectral_pattern")
ALGORITHM_VERSION = 2
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
OUTPUT_NAMES = (
    "spectral_stats.npz",
    "spectral_pattern.npy",
    "spectral_pattern.png",
    "spectral_pattern_tiled.png",
    "radial_psd.csv",
    "manifest.json",
)


@dataclass(frozen=True)
class CacheEntry:
    rank: int
    start: int
    count: int
    path: Path
    index_path: Path


@dataclass(frozen=True)
class CacheInventory:
    cache_root: Path
    split: str
    entries: tuple[CacheEntry, ...]
    index_paths: tuple[Path, ...]
    total_images: int
    dtype_name: str
    fingerprint: str


@dataclass(frozen=True)
class ImageDirectoryInventory:
    image_dir: Path
    image_paths: tuple[Path, ...]
    total_images: int
    fingerprint: str


SourceInventory = CacheInventory | ImageDirectoryInventory


@dataclass(frozen=True)
class ExtractionResult:
    pattern: np.ndarray
    cross_spectrum: np.ndarray
    source_psd: np.ndarray
    pattern_psd: np.ndarray
    radial_frequency: np.ndarray
    radial_source_energy: np.ndarray
    radial_pattern_energy: np.ndarray
    residual_channel_energy: np.ndarray
    num_images: int
    height: int
    width: int
    seed: int
    blur_sigma: float
    pattern_channel_mean: np.ndarray
    pattern_rms: float
    radial_psd_cosine: float
    unit_rms_normalized: bool


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npy(array: np.ndarray, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_png(array: np.ndarray, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        Image.fromarray(array).save(temporary, format="PNG")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _resolve_cache_path(cache_root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Invalid cache entry path: {value!r}")
    resolved = (cache_root / value).resolve()
    try:
        resolved.relative_to(cache_root)
    except ValueError as error:
        raise ValueError(
            f"Cache entry escapes cache root: {value!r}"
        ) from error
    return resolved


def discover_cache(cache_root: str | Path, split: str) -> CacheInventory:
    """Discover and validate all rank shards for one cache split."""
    root = Path(cache_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Cache root not found: {root}")
    split_root = root / split
    if not split_root.is_dir():
        raise NotADirectoryError(f"Cache split not found: {split_root}")

    index_paths = tuple(sorted(split_root.glob("rank_*/index.json")))
    if not index_paths:
        raise FileNotFoundError(
            f"No rank_*/index.json files found under {split_root}"
        )

    entries: list[CacheEntry] = []
    seen_ranks: set[int] = set()
    fingerprints: set[str] = set()
    dtype_names: set[str] = set()
    declared_world_sizes: set[int] = set()

    for index_path in index_paths:
        with index_path.open() as handle:
            index = json.load(handle)
        rank = int(index.get("rank", -1))
        if rank < 0 or rank in seen_ranks:
            raise ValueError(
                f"Invalid or duplicate rank {rank} in {index_path}"
            )
        seen_ranks.add(rank)
        declared_world_sizes.add(int(index.get("world_size", -1)))
        fingerprints.add(str(index.get("fingerprint", "")))
        dtype_names.add(str(index.get("dtype", "")))

        declared_count = int(index.get("count", -1))
        if declared_count < 1:
            raise ValueError(f"Invalid count in {index_path}")
        raw_entries = index.get("entries")
        if not isinstance(raw_entries, list) or not raw_entries:
            raise ValueError(f"No cache entries in {index_path}")

        expected_start = 0
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, dict):
                raise ValueError(f"Invalid entry in {index_path}")
            start = int(raw_entry.get("start", -1))
            count = int(raw_entry.get("count", -1))
            if start != expected_start or count < 1:
                raise ValueError(
                    f"Non-contiguous cache ranges in {index_path}: "
                    f"expected start={expected_start}, found "
                    f"start={start}, count={count}"
                )
            path = _resolve_cache_path(root, raw_entry.get("path"))
            if not path.is_file():
                raise FileNotFoundError(f"Cache chunk not found: {path}")
            entries.append(
                CacheEntry(
                    rank=rank,
                    start=start,
                    count=count,
                    path=path,
                    index_path=index_path,
                )
            )
            expected_start += count
        if expected_start != declared_count:
            raise ValueError(
                f"Entry counts in {index_path} sum to {expected_start}, "
                f"but index declares {declared_count}"
            )

    if len(fingerprints) != 1 or "" in fingerprints:
        raise ValueError("Cache rank indices have inconsistent fingerprints")
    if len(dtype_names) != 1 or not dtype_names <= {"uint8", "float16"}:
        raise ValueError(
            f"Cache rank indices have unsupported dtypes: {dtype_names}"
        )
    if len(declared_world_sizes) != 1:
        raise ValueError("Cache rank indices disagree on world_size")
    world_size = next(iter(declared_world_sizes))
    if world_size != len(seen_ranks) or seen_ranks != set(range(world_size)):
        raise ValueError(
            f"Expected contiguous ranks [0,{world_size}), found "
            f"{sorted(seen_ranks)}"
        )

    entries.sort(key=lambda item: (item.rank, item.start))
    total_images = sum(entry.count for entry in entries)
    return CacheInventory(
        cache_root=root,
        split=split,
        entries=tuple(entries),
        index_paths=index_paths,
        total_images=total_images,
        dtype_name=next(iter(dtype_names)),
        fingerprint=next(iter(fingerprints)),
    )


def discover_image_directory(
    image_dir: str | Path,
) -> ImageDirectoryInventory:
    """Discover a flat, deterministically ordered generated-image folder."""
    root = Path(image_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Image directory not found: {root}")
    paths = tuple(
        sorted(
            path
            for path in root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    )
    if not paths:
        raise FileNotFoundError(f"No supported images found in {root}")
    digest = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\0")
    return ImageDirectoryInventory(
        image_dir=root,
        image_paths=paths,
        total_images=len(paths),
        fingerprint=digest.hexdigest(),
    )


def iter_cached_batches(
    inventory: CacheInventory,
    *,
    num_images: int,
    batch_size: int,
) -> Iterator[torch.Tensor]:
    """Yield ordered CPU cache tensors without loading a full split."""
    if num_images < 1 or num_images > inventory.total_images:
        raise ValueError(
            f"num_images={num_images} must be in "
            f"[1,{inventory.total_images}]"
        )
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    remaining = num_images
    for entry in inventory.entries:
        if remaining == 0:
            break
        payload = torch.load(
            entry.path, map_location="cpu", weights_only=True
        )
        if not isinstance(payload, dict) or "images" not in payload:
            raise ValueError(f"Invalid cache payload: {entry.path}")
        images = payload["images"]
        if (
            not isinstance(images, torch.Tensor)
            or images.ndim != 4
            or images.shape[1] != 3
            or images.shape[0] != entry.count
        ):
            shape = getattr(images, "shape", None)
            raise ValueError(
                f"Invalid image tensor in {entry.path}: {shape}"
            )
        expected_dtype = (
            torch.uint8
            if inventory.dtype_name == "uint8"
            else torch.float16
        )
        if images.dtype != expected_dtype:
            raise ValueError(
                f"Cache dtype mismatch in {entry.path}: "
                f"{images.dtype} != {expected_dtype}"
            )
        take = min(remaining, images.shape[0])
        for start in range(0, take, batch_size):
            yield images[start : min(start + batch_size, take)]
        remaining -= take
    if remaining:
        raise RuntimeError(
            f"Cache traversal ended with {remaining} images missing"
        )


def iter_source_batches(
    inventory: SourceInventory,
    *,
    num_images: int,
    batch_size: int,
    device: torch.device,
) -> Iterator[torch.Tensor]:
    """Yield decoded float32 RGB batches in [0,1] from either source."""
    if isinstance(inventory, CacheInventory):
        for cached_images in iter_cached_batches(
            inventory,
            num_images=num_images,
            batch_size=batch_size,
        ):
            yield decode_cache_images(
                cached_images,
                dtype_name=inventory.dtype_name,
                device=device,
            )
        return

    if num_images < 1 or num_images > inventory.total_images:
        raise ValueError(
            f"num_images={num_images} must be in "
            f"[1,{inventory.total_images}]"
        )
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    selected = inventory.image_paths[:num_images]
    for start in range(0, num_images, batch_size):
        tensors: list[torch.Tensor] = []
        for path in selected[start : start + batch_size]:
            with Image.open(path) as opened:
                array = np.asarray(
                    opened.convert("RGB"), dtype=np.uint8
                ).copy()
            tensors.append(
                torch.from_numpy(array).permute(2, 0, 1)
            )
        shapes = {tuple(tensor.shape) for tensor in tensors}
        if len(shapes) != 1:
            raise ValueError(
                "Image directory batch contains mixed resolutions: "
                f"{sorted(shapes)}"
            )
        yield torch.stack(tensors).to(
            device=device, dtype=torch.float32
        ).div_(255.0)


def decode_cache_images(
    images: torch.Tensor,
    *,
    dtype_name: str,
    device: torch.device,
) -> torch.Tensor:
    """Decode cached pMF images to float32 sRGB in [0,1]."""
    if dtype_name == "uint8":
        if images.dtype != torch.uint8:
            raise ValueError("uint8 cache contains a non-uint8 tensor")
        return images.to(device=device, dtype=torch.float32).div_(255.0)
    if dtype_name == "float16":
        if images.dtype != torch.float16:
            raise ValueError("float16 cache contains a non-float16 tensor")
        decoded = images.to(device=device, dtype=torch.float32)
        minimum = float(decoded.amin())
        maximum = float(decoded.amax())
        if minimum < -1.01 or maximum > 1.01:
            raise ValueError(
                "float16 cache is outside expected pMF model range "
                f"[-1,1]: [{minimum},{maximum}]"
            )
        return decoded.add_(1.0).mul_(0.5).clamp_(0.0, 1.0)
    raise ValueError(f"Unsupported cache dtype: {dtype_name}")


def gaussian_kernel1d(
    sigma: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("blur_sigma must be finite and positive")
    radius = max(1, int(math.ceil(4.0 * sigma)))
    coordinate = torch.arange(
        -radius, radius + 1, device=device, dtype=dtype
    )
    kernel = torch.exp(-0.5 * coordinate.square() / (sigma * sigma))
    return kernel / kernel.sum()


def gaussian_highpass(
    images: torch.Tensor,
    *,
    sigma: float,
) -> torch.Tensor:
    """Return x - GaussianBlur(x) with reflect padding."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(
            f"Expected [B,3,H,W], got {tuple(images.shape)}"
        )
    kernel = gaussian_kernel1d(
        sigma, device=images.device, dtype=images.dtype
    )
    radius = kernel.numel() // 2
    height, width = images.shape[-2:]
    if radius >= min(height, width):
        raise ValueError(
            f"Gaussian radius {radius} is too large for {height}x{width}"
        )
    channels = images.shape[1]
    horizontal = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    vertical = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    blurred = F.conv2d(
        F.pad(images, (radius, radius, 0, 0), mode="reflect"),
        horizontal,
        groups=channels,
    )
    blurred = F.conv2d(
        F.pad(blurred, (0, 0, radius, radius), mode="reflect"),
        vertical,
        groups=channels,
    )
    return images - blurred


def hann_window_2d(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if height < 2 or width < 2:
        raise ValueError("FFT resolution must be at least 2x2")
    window_y = torch.hann_window(
        height, periodic=False, device=device, dtype=dtype
    )
    window_x = torch.hann_window(
        width, periodic=False, device=device, dtype=dtype
    )
    window = window_y[:, None] * window_x[None, :]
    rms = window.square().mean().sqrt()
    if not torch.isfinite(rms) or float(rms) <= 0:
        raise FloatingPointError("Invalid Hann-window RMS")
    return window / rms


def _enforce_rfft_boundary_symmetry(
    coefficients: torch.Tensor,
    *,
    spatial_width: int,
) -> torch.Tensor:
    """Enforce real-image constraints on rFFT boundary columns."""
    height, half_width, channels = coefficients.shape
    if channels != 3 or half_width != spatial_width // 2 + 1:
        raise ValueError("Unexpected rFFT coefficient shape")
    result = coefficients.clone()
    boundary_columns = [0]
    if spatial_width % 2 == 0:
        boundary_columns.append(spatial_width // 2)
    for column in boundary_columns:
        result[0, column] = result[0, column].real
        if height % 2 == 0:
            result[height // 2, column] = result[
                height // 2, column
            ].real
        for row in range(1, (height + 1) // 2):
            paired_row = (-row) % height
            average = 0.5 * (
                result[row, column]
                + result[paired_row, column].conj()
            )
            result[row, column] = average
            result[paired_row, column] = average.conj()
    return result


def synthesize_pattern(
    cross_spectrum: torch.Tensor,
    *,
    spatial_width: int,
    seed: int,
    normalize_rms: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form seeded synthesis from an RGB cross-spectrum."""
    if (
        cross_spectrum.ndim != 4
        or cross_spectrum.shape[-2:] != (3, 3)
    ):
        raise ValueError(
            "cross_spectrum must have shape [H,W//2+1,3,3]"
        )
    height, half_width = cross_spectrum.shape[:2]
    if half_width != spatial_width // 2 + 1:
        raise ValueError("Cross-spectrum width does not match spatial width")
    if not torch.isfinite(cross_spectrum.real).all() or not torch.isfinite(
        cross_spectrum.imag
    ).all():
        raise FloatingPointError("Cross-spectrum contains NaN or infinity")

    hermitian = 0.5 * (
        cross_spectrum
        + cross_spectrum.conj().transpose(-1, -2)
    )
    eigenvalues, eigenvectors = torch.linalg.eigh(hermitian)
    eigenvalues = eigenvalues.clamp_min(0.0)
    square_root = torch.einsum(
        "...ce,...e,...de->...cd",
        eigenvectors,
        eigenvalues.sqrt(),
        eigenvectors.conj(),
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    white = torch.randn(
        3, height, spatial_width, generator=generator, dtype=torch.float64
    ).to(cross_spectrum.device)
    white_frequency = torch.fft.rfft2(
        white, norm="ortho"
    ).permute(1, 2, 0)
    colored = torch.einsum(
        "...cd,...d->...c", square_root, white_frequency
    )
    colored[0, 0] = 0
    colored = _enforce_rfft_boundary_symmetry(
        colored, spatial_width=spatial_width
    )
    pattern = torch.fft.irfft2(
        colored.permute(2, 0, 1),
        s=(height, spatial_width),
        norm="ortho",
    )
    pattern = pattern - pattern.mean(dim=(-2, -1), keepdim=True)
    rms = pattern.square().mean().sqrt()
    if not torch.isfinite(rms) or float(rms) <= 1e-12:
        raise FloatingPointError(
            "Synthesized pattern has zero or non-finite RMS"
        )
    if normalize_rms:
        pattern = pattern / rms
    return pattern, colored


def radial_energy_profile(
    psd: np.ndarray,
    *,
    spatial_width: int,
    bins: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a 2-D rFFT PSD to normalized radial energy fractions."""
    if psd.ndim != 2:
        raise ValueError(f"PSD must be 2-D, got {psd.shape}")
    height, half_width = psd.shape
    if half_width != spatial_width // 2 + 1:
        raise ValueError("PSD shape does not match spatial width")
    bins = bins or max(8, min(height, spatial_width) // 2)
    frequency_y = np.fft.fftfreq(height)
    frequency_x = np.fft.rfftfreq(spatial_width)
    radius = np.sqrt(
        frequency_y[:, None] ** 2 + frequency_x[None, :] ** 2
    )
    maximum = math.sqrt(0.5)
    edges = np.linspace(0.0, maximum + 1e-12, bins + 1)
    indices = np.clip(
        np.digitize(radius.ravel(), edges, right=False) - 1,
        0,
        bins - 1,
    )
    values = np.asarray(psd, dtype=np.float64).ravel()
    values = np.clip(values, 0.0, None)
    # rFFT stores only non-negative x frequencies. Interior columns represent
    # both +/- frequencies and therefore carry twice the full-spectrum energy.
    column_weight = np.full(half_width, 2.0, dtype=np.float64)
    column_weight[0] = 1.0
    if spatial_width % 2 == 0:
        column_weight[-1] = 1.0
    values *= np.broadcast_to(
        column_weight[None, :], (height, half_width)
    ).ravel()
    energy = np.bincount(
        indices, weights=values, minlength=bins
    ).astype(np.float64)
    total = float(energy.sum())
    if not math.isfinite(total) or total <= 0:
        raise FloatingPointError("PSD has zero or non-finite energy")
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, energy / total


def extract_spectral_pattern(
    inventory: SourceInventory,
    *,
    num_images: int,
    batch_size: int,
    device: torch.device,
    blur_sigma: float,
    seed: int,
    log_every: int = 5_000,
    normalize_rms: bool = True,
) -> ExtractionResult:
    """Stream source images, estimate CSD, and synthesize one pattern."""
    if num_images < 2:
        raise ValueError("At least two images are required")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if log_every < 1:
        raise ValueError("log_every must be positive")

    cross_sum: torch.Tensor | None = None
    residual_energy_sum: torch.Tensor | None = None
    window: torch.Tensor | None = None
    height = width = 0
    processed = 0
    next_log = min(log_every, num_images)
    started = time.perf_counter()

    for images in iter_source_batches(
        inventory,
        num_images=num_images,
        batch_size=batch_size,
        device=device,
    ):
        batch_height, batch_width = images.shape[-2:]
        if cross_sum is None:
            height, width = batch_height, batch_width
            window = hann_window_2d(
                height,
                width,
                device=device,
                dtype=images.dtype,
            )
            cross_sum = torch.zeros(
                height,
                width // 2 + 1,
                3,
                3,
                device=device,
                dtype=torch.complex128,
            )
            residual_energy_sum = torch.zeros(
                3, device=device, dtype=torch.float64
            )
        elif (batch_height, batch_width) != (height, width):
            raise ValueError(
                "Cache contains mixed resolutions: "
                f"{batch_height}x{batch_width} != {height}x{width}"
            )
        assert (
            cross_sum is not None
            and residual_energy_sum is not None
            and window is not None
        )

        residual = gaussian_highpass(images, sigma=blur_sigma)
        residual_energy_sum.add_(
            residual.square().sum(
                dim=(0, 2, 3), dtype=torch.float64
            )
        )
        frequency = torch.fft.rfft2(
            residual * window,
            norm="ortho",
        ).permute(0, 2, 3, 1)
        batch_cross = torch.einsum(
            "bhwc,bhwd->hwcd",
            frequency,
            frequency.conj(),
        )
        cross_sum.add_(batch_cross.to(torch.complex128))
        processed += images.shape[0]

        if processed >= next_log or processed == num_images:
            elapsed = time.perf_counter() - started
            logger.info(
                "processed=%d/%d elapsed=%.1fs images_per_second=%.1f",
                processed,
                num_images,
                elapsed,
                processed / max(elapsed, 1e-12),
            )
            next_log += log_every
        del (
            images,
            residual,
            frequency,
            batch_cross,
        )

    if (
        processed != num_images
        or cross_sum is None
        or residual_energy_sum is None
    ):
        raise RuntimeError(
            f"Processed {processed} images, expected {num_images}"
        )

    cross_spectrum = cross_sum / processed
    cross_spectrum = 0.5 * (
        cross_spectrum
        + cross_spectrum.conj().transpose(-1, -2)
    )
    if not torch.isfinite(cross_spectrum.real).all() or not torch.isfinite(
        cross_spectrum.imag
    ).all():
        raise FloatingPointError(
            "Accumulated cross-spectrum contains NaN or infinity"
        )
    # Finite-precision complex64 batch products can leave tiny negative
    # eigenvalues in an otherwise positive-semidefinite statistic.
    spectrum_eigenvalues, spectrum_eigenvectors = torch.linalg.eigh(
        cross_spectrum
    )
    cross_spectrum = torch.einsum(
        "...ce,...e,...de->...cd",
        spectrum_eigenvectors,
        spectrum_eigenvalues.clamp_min(0.0),
        spectrum_eigenvectors.conj(),
    )

    pattern, colored_frequency = synthesize_pattern(
        cross_spectrum,
        spatial_width=width,
        seed=seed,
        normalize_rms=normalize_rms,
    )
    source_psd_tensor = torch.diagonal(
        cross_spectrum, dim1=-2, dim2=-1
    ).real.sum(dim=-1)
    pattern_frequency = torch.fft.rfft2(pattern, norm="ortho")
    pattern_psd_tensor = pattern_frequency.abs().square().sum(dim=0)

    source_psd = source_psd_tensor.detach().cpu().numpy()
    pattern_psd = pattern_psd_tensor.detach().cpu().numpy()
    radial_frequency, radial_source = radial_energy_profile(
        source_psd, spatial_width=width
    )
    pattern_frequency_axis, radial_pattern = radial_energy_profile(
        pattern_psd, spatial_width=width
    )
    if not np.array_equal(radial_frequency, pattern_frequency_axis):
        raise RuntimeError("Radial PSD frequency axes do not match")
    denominator = float(
        np.linalg.norm(radial_source) * np.linalg.norm(radial_pattern)
    )
    radial_cosine = (
        float(np.dot(radial_source, radial_pattern) / denominator)
        if denominator > 0
        else 0.0
    )

    pattern_cpu = pattern.float().detach().cpu()
    channel_mean = pattern_cpu.mean(dim=(-2, -1)).numpy()
    pattern_rms = float(pattern_cpu.square().mean().sqrt())
    residual_channel_energy = (
        residual_energy_sum
        / (processed * height * width)
    ).detach().cpu().numpy()
    cross_spectrum_numpy = cross_spectrum.detach().cpu().numpy()
    del (
        cross_sum,
        residual_energy_sum,
        spectrum_eigenvalues,
        spectrum_eigenvectors,
        cross_spectrum,
        pattern,
        colored_frequency,
        source_psd_tensor,
        pattern_frequency,
        pattern_psd_tensor,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return ExtractionResult(
        pattern=pattern_cpu.numpy(),
        cross_spectrum=cross_spectrum_numpy,
        source_psd=source_psd,
        pattern_psd=pattern_psd,
        radial_frequency=radial_frequency,
        radial_source_energy=radial_source,
        radial_pattern_energy=radial_pattern,
        residual_channel_energy=residual_channel_energy,
        num_images=processed,
        height=height,
        width=width,
        seed=int(seed),
        blur_sigma=float(blur_sigma),
        pattern_channel_mean=channel_mean,
        pattern_rms=pattern_rms,
        radial_psd_cosine=radial_cosine,
        unit_rms_normalized=bool(normalize_rms),
    )


def _preview_array(pattern: np.ndarray) -> np.ndarray:
    channels_last = np.transpose(pattern, (1, 2, 0))
    lower, upper = np.percentile(channels_last, [1.0, 99.0])
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise FloatingPointError("Cannot construct pattern preview")
    normalized = np.clip(
        (channels_last - lower) / (upper - lower),
        0.0,
        1.0,
    )
    return np.rint(normalized * 255.0).astype(np.uint8)


def write_outputs(
    result: ExtractionResult,
    inventory: SourceInventory,
    output_dir: str | Path,
    *,
    overwrite: bool,
    elapsed_seconds: float,
) -> dict[str, Path]:
    output = Path(output_dir).expanduser().resolve()
    existing = [output / name for name in OUTPUT_NAMES if (output / name).exists()]
    if existing and not overwrite:
        formatted = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing outputs: {formatted}"
        )
    output.mkdir(parents=True, exist_ok=True)
    paths = {name: output / name for name in OUTPUT_NAMES}

    _atomic_npz(
        paths["spectral_stats.npz"],
        cross_spectrum=result.cross_spectrum,
        source_psd=result.source_psd,
        pattern_psd=result.pattern_psd,
        residual_channel_energy=result.residual_channel_energy,
        radial_frequency=result.radial_frequency,
        radial_source_energy=result.radial_source_energy,
        radial_pattern_energy=result.radial_pattern_energy,
        num_images=np.asarray(result.num_images, dtype=np.int64),
        resolution=np.asarray(
            [result.height, result.width], dtype=np.int64
        ),
        seed=np.asarray(result.seed, dtype=np.int64),
        blur_sigma=np.asarray(result.blur_sigma, dtype=np.float64),
        unit_rms_normalized=np.asarray(
            result.unit_rms_normalized, dtype=np.bool_
        ),
    )
    _atomic_npy(result.pattern.astype(np.float32), paths["spectral_pattern.npy"])
    preview = _preview_array(result.pattern)
    _atomic_png(preview, paths["spectral_pattern.png"])
    tiled_preview = np.tile(preview, (2, 2, 1))
    _atomic_png(tiled_preview, paths["spectral_pattern_tiled.png"])

    radial_temporary = paths["radial_psd.csv"].with_name(
        f".radial_psd.csv.tmp-{os.getpid()}"
    )
    try:
        with radial_temporary.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "frequency_cycles_per_pixel",
                    "source_energy_fraction",
                    "pattern_energy_fraction",
                ]
            )
            writer.writerows(
                zip(
                    result.radial_frequency.tolist(),
                    result.radial_source_energy.tolist(),
                    result.radial_pattern_energy.tolist(),
                )
            )
        os.replace(radial_temporary, paths["radial_psd.csv"])
    finally:
        radial_temporary.unlink(missing_ok=True)

    if isinstance(inventory, CacheInventory):
        manifest_path = inventory.cache_root / "manifest.json"
        source_manifest: dict[str, object] = {
            "kind": "pmf_cache",
            "cache_root": str(inventory.cache_root),
            "split": inventory.split,
            "cache_fingerprint": inventory.fingerprint,
            "cache_dtype": inventory.dtype_name,
            "cache_total_images": inventory.total_images,
            "used_images": result.num_images,
            "cache_manifest_sha256": (
                _sha256(manifest_path) if manifest_path.is_file() else None
            ),
            "indices": [
                {
                    "path": str(path.relative_to(inventory.cache_root)),
                    "sha256": _sha256(path),
                }
                for path in inventory.index_paths
            ],
        }
    else:
        source_manifest = {
            "kind": "image_directory",
            "image_dir": str(inventory.image_dir),
            "image_fingerprint": inventory.fingerprint,
            "directory_total_images": inventory.total_images,
            "used_images": result.num_images,
            "first_image": inventory.image_paths[0].name,
            "last_used_image": inventory.image_paths[
                result.num_images - 1
            ].name,
        }
    output_hashes = {
        name: _sha256(path)
        for name, path in paths.items()
        if name != "manifest.json"
    }
    manifest: dict[str, object] = {
        "algorithm": "rgb_highpass_cross_spectral_synthesis",
        "algorithm_version": ALGORITHM_VERSION,
        "source": source_manifest,
        "parameters": {
            "blur_sigma": result.blur_sigma,
            "seed": result.seed,
            "resolution": [result.height, result.width],
            "fft_normalization": "ortho",
            "window": "hann_rms_normalized",
            "pattern_normalization": (
                "per_channel_zero_mean_global_unit_rms"
                if result.unit_rms_normalized
                else "per_channel_zero_mean_preserve_spectral_scale"
            ),
        },
        "diagnostics": {
            "pattern_channel_mean": result.pattern_channel_mean.tolist(),
            "pattern_rms": result.pattern_rms,
            "residual_channel_energy": (
                result.residual_channel_energy.tolist()
            ),
            "radial_psd_cosine": result.radial_psd_cosine,
            "elapsed_seconds": float(elapsed_seconds),
        },
        "outputs": output_hashes,
        "training": {
            "optimizer": None,
            "loss": None,
            "backward": False,
            "metric_guided_selection": False,
        },
    }
    _atomic_json(manifest, paths["manifest.json"])
    return paths


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract a deterministic natural-spectrum RGB pattern from "
            "images without metric optimization"
        )
    )
    parser.add_argument(
        "--cache_root",
        default="./work_dirs/hacking_cache/pMF_B_256",
    )
    parser.add_argument(
        "--image_dir",
        default=None,
        help=(
            "flat generated-image folder; when set, takes precedence over "
            "--cache_root/--split"
        ),
    )
    parser.add_argument("--split", default="optimization")
    parser.add_argument("--num_images", type=int, default=50_000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--blur_sigma", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--log_every", type=int, default=5_000)
    parser.add_argument(
        "--preserve_spectral_scale",
        action="store_true",
        help=(
            "keep the pixel-space RMS implied by the estimated spectrum; "
            "otherwise normalize the synthesized pattern to unit RMS"
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="./work_dirs/spectral_pattern/pMF_B_256",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.num_images < 2:
        raise ValueError("--num_images must be at least 2")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be positive")
    if args.log_every < 1:
        raise ValueError("--log_every must be positive")
    if not math.isfinite(args.blur_sigma) or args.blur_sigma <= 0:
        raise ValueError("--blur_sigma must be finite and positive")
    device = torch.device(args.device)

    if args.image_dir is not None:
        inventory: SourceInventory = discover_image_directory(
            args.image_dir
        )
        logger.info(
            "image_dir=%s total_images=%d",
            inventory.image_dir,
            inventory.total_images,
        )
    else:
        inventory = discover_cache(args.cache_root, args.split)
        logger.info(
            "cache=%s split=%s total_images=%d dtype=%s ranks=%d",
            inventory.cache_root,
            inventory.split,
            inventory.total_images,
            inventory.dtype_name,
            len(inventory.index_paths),
        )
    started = time.perf_counter()
    result = extract_spectral_pattern(
        inventory,
        num_images=args.num_images,
        batch_size=args.batch_size,
        device=device,
        blur_sigma=args.blur_sigma,
        seed=args.seed,
        log_every=args.log_every,
        normalize_rms=not args.preserve_spectral_scale,
    )
    elapsed = time.perf_counter() - started
    paths = write_outputs(
        result,
        inventory,
        args.output_dir,
        overwrite=args.overwrite,
        elapsed_seconds=elapsed,
    )
    logger.info(
        "pattern mean=%s rms=%.8f radial_psd_cosine=%.6f",
        np.array2string(result.pattern_channel_mean, precision=4),
        result.pattern_rms,
        result.radial_psd_cosine,
    )
    for name, path in paths.items():
        logger.info("%s: %s", name, path)


if __name__ == "__main__":
    main(get_args_parser().parse_args())
