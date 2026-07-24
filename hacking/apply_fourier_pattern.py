"""Apply a learned tiled Fourier pattern to an image directory.

This is a lightweight post-processing utility: it does not load pMF,
Inception, or CLIP.  It accepts either the exact ``fourier_pattern.npy``
export or a Fourier pattern checkpoint produced by
``pmf_fourier_universal.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps


IMAGE_SUFFIXES = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def _normalize_spatial_pattern(pattern: torch.Tensor) -> torch.Tensor:
    if pattern.ndim == 4 and pattern.shape[0] == 1:
        pattern = pattern[0]
    if pattern.ndim != 3 or pattern.shape[0] != 3:
        raise ValueError(
            "Pattern must have shape [3,H,W] or [1,3,H,W], "
            f"got {tuple(pattern.shape)}"
        )
    pattern = pattern.float()
    pattern = pattern - pattern.mean(dim=(-2, -1), keepdim=True)
    rms = pattern.square().mean().sqrt()
    if float(rms) > 1e-12:
        pattern = pattern / rms
    return pattern.contiguous()


def _pattern_from_checkpoint(path: Path) -> torch.Tensor:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("pattern", checkpoint)
    required = {"coeff", "cos_basis", "sin_basis"}
    if not isinstance(state, dict) or not required.issubset(state):
        available = sorted(state) if isinstance(state, dict) else type(state)
        raise ValueError(
            f"{path} is not a Fourier pattern checkpoint. "
            f"Expected keys {sorted(required)}, found {available}"
        )
    coeff = state["coeff"].float()
    cos_basis = state["cos_basis"].float()
    sin_basis = state["sin_basis"].float()
    if coeff.ndim != 3 or coeff.shape[0] != 3 or coeff.shape[-1] != 2:
        raise ValueError(
            f"Invalid Fourier coefficient shape: {tuple(coeff.shape)}"
        )
    if (
        cos_basis.shape != sin_basis.shape
        or cos_basis.ndim != 3
        or cos_basis.shape[0] != coeff.shape[1]
    ):
        raise ValueError(
            "Checkpoint coefficient and Fourier-basis shapes do not match"
        )
    cosine = torch.einsum("cm,mhw->chw", coeff[..., 0], cos_basis)
    sine = torch.einsum("cm,mhw->chw", coeff[..., 1], sin_basis)
    return cosine + sine


def load_spatial_pattern(path: str | Path) -> torch.Tensor:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Pattern not found: {path}")
    if path.suffix.lower() == ".npy":
        pattern = torch.from_numpy(np.load(path))
    elif path.suffix.lower() in {".pt", ".pth"}:
        pattern = _pattern_from_checkpoint(path)
    else:
        raise ValueError(
            f"Unsupported pattern file {path}; use .npy, .pt, or .pth"
        )
    return _normalize_spatial_pattern(pattern)


def tile_pattern(
    pattern: torch.Tensor,
    height: int,
    width: int,
    *,
    shift_y: int = 0,
    shift_x: int = 0,
) -> torch.Tensor:
    """Return a phase-shifted tiled pattern with shape ``[3,H,W]``."""
    pattern = torch.roll(
        pattern,
        shifts=(int(shift_y), int(shift_x)),
        dims=(-2, -1),
    )
    return pattern.repeat(
        1,
        math.ceil(height / pattern.shape[-2]),
        math.ceil(width / pattern.shape[-1]),
    )[:, :height, :width]


def apply_pattern(
    image_01: torch.Tensor,
    pattern: torch.Tensor,
    *,
    alpha: float,
    alpha_space: str = "model",
    shift_y: int = 0,
    shift_x: int = 0,
) -> torch.Tensor:
    """Apply the pattern to one RGB image in ``[0,1]``.

    ``alpha_space='model'`` matches the training experiment:
    ``clip((2*x-1) + alpha*u, -1, 1)``.  Therefore model-space alpha
    ``8/255`` corresponds to pixel-space pattern RMS ``4/255``.
    """
    if image_01.ndim != 3 or image_01.shape[0] != 3:
        raise ValueError(
            f"image_01 must have shape [3,H,W], got {tuple(image_01.shape)}"
        )
    tiled = tile_pattern(
        pattern.to(device=image_01.device, dtype=image_01.dtype),
        image_01.shape[-2],
        image_01.shape[-1],
        shift_y=shift_y,
        shift_x=shift_x,
    )
    if alpha_space == "model":
        image_model = image_01.mul(2.0).sub(1.0)
        patched_model = image_model.add(tiled, alpha=float(alpha))
        return patched_model.clamp(-1.0, 1.0).add(1.0).mul(0.5)
    if alpha_space == "pixel":
        return image_01.add(tiled, alpha=float(alpha)).clamp(0.0, 1.0)
    raise ValueError(f"Unknown alpha space: {alpha_space}")


def _discover_images(input_dir: Path, recursive: bool) -> list[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _output_path(
    source: Path,
    input_dir: Path,
    output_dir: Path,
    output_format: str,
) -> Path:
    relative = source.relative_to(input_dir)
    destination = output_dir / relative
    if output_format == "png":
        destination = destination.with_suffix(".png")
    return destination


def _load_rgb(path: Path) -> tuple[torch.Tensor, Image.Image | None]:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened)
        alpha_channel = (
            image.getchannel("A").copy()
            if "A" in image.getbands()
            else None
        )
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).div_(255.0)
    return tensor, alpha_channel


def _save_image(
    image_01: torch.Tensor,
    destination: Path,
    alpha_channel: Image.Image | None,
    *,
    jpeg_quality: int,
) -> None:
    array = (
        image_01.mul(255.0)
        .round()
        .clamp_(0, 255)
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    output = Image.fromarray(array, mode="RGB")
    if alpha_channel is not None and destination.suffix.lower() in {
        ".png",
        ".tif",
        ".tiff",
        ".webp",
    }:
        output.putalpha(alpha_channel)
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict[str, object] = {}
    if destination.suffix.lower() in {".jpg", ".jpeg"}:
        save_kwargs.update(quality=jpeg_quality, subsampling=0)
    output.save(destination, **save_kwargs)


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "Apply a learned Fourier pattern to an image directory"
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--pattern",
        required=True,
        help="fourier_pattern.npy or Fourier pattern .pth checkpoint",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=8 / 255,
        help="pattern strength; interpreted in --alpha_space",
    )
    parser.add_argument(
        "--alpha_space",
        choices=["model", "pixel"],
        default="model",
        help="model matches the pMF [-1,1] experiment",
    )
    parser.add_argument("--shift_y", type=int, default=0)
    parser.add_argument("--shift_x", type=int, default=0)
    parser.add_argument(
        "--output_format",
        choices=["preserve", "png"],
        default="preserve",
    )
    parser.add_argument("--jpeg_quality", type=int, default=95)
    parser.add_argument(
        "--no_recursive",
        action="store_false",
        dest="recursive",
        help="only process images directly inside --input_dir",
    )
    parser.set_defaults(recursive=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="process only the first N images; 0 processes all",
    )
    parser.add_argument(
        "--random_sample",
        type=int,
        default=0,
        help="randomly select N images; mutually exclusive with --limit",
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=2026,
        help="reproducible seed used by --random_sample",
    )
    return parser


def main(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")
    if input_dir == output_dir:
        raise ValueError("--output_dir must differ from --input_dir")
    try:
        output_dir.relative_to(input_dir)
    except ValueError:
        pass
    else:
        raise ValueError(
            "--output_dir cannot be inside --input_dir; this prevents "
            "accidentally processing previous outputs"
        )
    if not math.isfinite(args.alpha) or args.alpha < 0:
        raise ValueError("--alpha must be finite and non-negative")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg_quality must be in [1,100]")
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")
    if args.random_sample < 0:
        raise ValueError("--random_sample must be non-negative")
    if args.limit and args.random_sample:
        raise ValueError("--limit and --random_sample are mutually exclusive")

    pattern_path = Path(args.pattern).expanduser().resolve()
    pattern = load_spatial_pattern(pattern_path)
    sources = _discover_images(input_dir, args.recursive)
    discovered_images = len(sources)
    if args.random_sample:
        if args.random_sample > len(sources):
            raise ValueError(
                f"--random_sample={args.random_sample} exceeds the "
                f"{len(sources)} discovered images"
            )
        sources = random.Random(args.sample_seed).sample(
            sources, args.random_sample
        )
        sources.sort()
    if args.limit:
        sources = sources[: args.limit]
    if not sources:
        raise FileNotFoundError(f"No supported images found in {input_dir}")

    destinations = [
        _output_path(
            source,
            input_dir,
            output_dir,
            args.output_format,
        )
        for source in sources
    ]
    if len(set(destinations)) != len(destinations):
        raise ValueError(
            "Multiple inputs map to the same output name; use "
            "--output_format preserve"
        )
    existing = [path for path in destinations if path.exists()]
    if existing and not args.overwrite:
        preview = "\n".join(str(path) for path in existing[:5])
        raise FileExistsError(
            f"{len(existing)} output files already exist. Pass --overwrite "
            f"to replace them. First paths:\n{preview}"
        )

    print(
        f"pattern={pattern_path} shape={tuple(pattern.shape)} "
        f"mean={float(pattern.mean()):.6g} "
        f"rms={float(pattern.square().mean().sqrt()):.6g}"
    )
    effective_pixel_rms = (
        args.alpha / 2.0
        if args.alpha_space == "model"
        else args.alpha
    )
    print(
        f"images={len(sources)} alpha={args.alpha:.10g} "
        f"alpha_space={args.alpha_space} "
        f"effective_pixel_rms={effective_pixel_rms:.10g} "
        f"phase=({args.shift_y},{args.shift_x})"
    )

    for index, (source, destination) in enumerate(
        zip(sources, destinations), start=1
    ):
        image, alpha_channel = _load_rgb(source)
        patched = apply_pattern(
            image,
            pattern,
            alpha=args.alpha,
            alpha_space=args.alpha_space,
            shift_y=args.shift_y,
            shift_x=args.shift_x,
        )
        _save_image(
            patched,
            destination,
            alpha_channel,
            jpeg_quality=args.jpeg_quality,
        )
        if index % 100 == 0 or index == len(sources):
            print(f"processed {index}/{len(sources)}")

    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "pattern": str(pattern_path),
        "pattern_shape": list(pattern.shape),
        "pattern_mean": float(pattern.mean()),
        "pattern_rms": float(pattern.square().mean().sqrt()),
        "alpha": args.alpha,
        "alpha_space": args.alpha_space,
        "effective_pixel_rms": effective_pixel_rms,
        "shift_y": args.shift_y,
        "shift_x": args.shift_x,
        "num_images": len(sources),
        "num_discovered_images": discovered_images,
        "random_sample": args.random_sample,
        "sample_seed": args.sample_seed if args.random_sample else None,
        "selected_images": [
            str(source.relative_to(input_dir)) for source in sources
        ],
        "output_format": args.output_format,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "apply_pattern_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"done: {output_dir}")


if __name__ == "__main__":
    main(get_args_parser().parse_args())
