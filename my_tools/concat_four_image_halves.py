#!/usr/bin/env python3
"""Concatenate the left and right halves of four comparison images.

Each input image must have the same dimensions and an even width.  The script
splits every image at its vertical midpoint, then writes two images:

    left_output  = image_1.left | image_2.left | image_3.left | image_4.left
    right_output = image_1.right | image_2.right | image_3.right | image_4.right

The input order on the command line is preserved.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from PIL import Image, ImageOps


def load_and_split(path: Path) -> tuple[Image.Image, Image.Image]:
    if not path.is_file():
        raise FileNotFoundError(f"input image not found: {path}")

    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")

    if image.width <= 0 or image.height <= 0:
        raise ValueError(f"{path}: image has an invalid size {image.size}")
    if image.width % 2:
        raise ValueError(
            f"{path}: width must be even so it can be split in half; "
            f"got {image.width}"
        )

    midpoint = image.width // 2
    return (
        image.crop((0, 0, midpoint, image.height)),
        image.crop((midpoint, 0, image.width, image.height)),
    )


def concatenate_panels(panels: list[Image.Image]) -> Image.Image:
    if len(panels) != 4:
        raise ValueError(f"exactly four panels are required, got {len(panels)}")

    panel_width, panel_height = panels[0].size
    if any(panel.size != (panel_width, panel_height) for panel in panels[1:]):
        sizes = [panel.size for panel in panels]
        raise ValueError(f"all panels must have the same size, got {sizes}")

    result = Image.new(
        "RGB",
        (panel_width * len(panels), panel_height),
        (255, 255, 255),
    )
    for index, panel in enumerate(panels):
        result.paste(panel, (index * panel_width, 0))
    return result


def save_atomic(image: Image.Image, path: Path, overwrite: bool) -> None:
    path = path.expanduser().resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {path}; pass --overwrite to replace it"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.stem}.tmp-{os.getpid()}{path.suffix or '.png'}"
    )
    try:
        save_kwargs: dict[str, object] = {}
        if path.suffix.lower() == ".png":
            save_kwargs["dpi"] = (300, 300)
        elif path.suffix.lower() in {".jpg", ".jpeg"}:
            save_kwargs.update(quality=98, subsampling=0, dpi=(300, 300))
        image.save(temporary, **save_kwargs)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Split four input images into left/right halves and concatenate "
            "the four left halves and four right halves separately."
        )
    )
    parser.add_argument(
        "images",
        type=Path,
        nargs=4,
        metavar="IMAGE",
        help="exactly four input images, in the desired concatenation order",
    )
    parser.add_argument(
        "--left_output",
        type=Path,
        default=Path("left_halves_concat.png"),
        help="output path for the horizontally concatenated left halves",
    )
    parser.add_argument(
        "--right_output",
        type=Path,
        default=Path("right_halves_concat.png"),
        help="output path for the horizontally concatenated right halves",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing output files",
    )
    return parser


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    input_paths = [path.expanduser().resolve() for path in args.images]
    if len(set(input_paths)) != 4:
        raise ValueError("the four input paths must be distinct")

    split_panels = [load_and_split(path) for path in input_paths]
    image_sizes = {
        (left.width * 2, left.height)
        for left, _ in split_panels
    }
    if len(image_sizes) != 1:
        raise ValueError(
            "all four input images must have the same dimensions; got "
            f"{sorted(image_sizes)}"
        )

    left_output = concatenate_panels([left for left, _ in split_panels])
    right_output = concatenate_panels([right for _, right in split_panels])
    left_path = args.left_output.expanduser().resolve()
    right_path = args.right_output.expanduser().resolve()
    if left_path == right_path:
        raise ValueError("--left_output and --right_output must be different")

    save_atomic(left_output, left_path, args.overwrite)
    try:
        save_atomic(right_output, right_path, args.overwrite)
    except Exception:
        # Do not leave a half-completed pair when the second output fails.
        left_path.unlink(missing_ok=True)
        raise

    print(f"left:  {left_path} ({left_output.width}x{left_output.height})")
    print(f"right: {right_path} ({right_output.width}x{right_output.height})")
    return left_path, right_path


if __name__ == "__main__":
    run(get_args_parser().parse_args())
