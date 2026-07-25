"""Add matched zoom-in insets to a side-by-side image pair.

The input layout is assumed to be::

    left image | right image

ROI coordinates are relative to one panel, not to the full concatenated
image. By default, the same ROI is used for both panels.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw, ImageOps


def _roi_box(values: list[int], name: str) -> tuple[int, int, int, int]:
    x, y, width, height = values
    if x < 0 or y < 0:
        raise ValueError(f"{name}: x and y must be non-negative")
    if width <= 0 or height <= 0:
        raise ValueError(f"{name}: width and height must be positive")
    return x, y, x + width, y + height


def _validate_roi(
    box: tuple[int, int, int, int],
    *,
    panel_width: int,
    panel_height: int,
    name: str,
) -> None:
    left, top, right, bottom = box
    if right > panel_width or bottom > panel_height:
        raise ValueError(
            f"{name}={box} exceeds panel size "
            f"{panel_width}x{panel_height}"
        )


def _make_framed_inset(
    panel: Image.Image,
    roi: tuple[int, int, int, int],
    *,
    inset_size: tuple[int, int],
    border_width: int,
    border_color: tuple[int, int, int],
) -> Image.Image:
    crop = panel.crop(roi).resize(
        inset_size,
        resample=Image.Resampling.LANCZOS,
    )
    if border_width == 0:
        return crop
    framed = Image.new(
        "RGB",
        (
            inset_size[0] + 2 * border_width,
            inset_size[1] + 2 * border_width,
        ),
        border_color,
    )
    framed.paste(crop, (border_width, border_width))
    return framed


def add_zoom_insets(
    image: Image.Image,
    *,
    left_roi: tuple[int, int, int, int],
    right_roi: tuple[int, int, int, int],
    panel_gap: int,
    inset_size: tuple[int, int],
    margin: int,
    border_width: int,
    border_color: tuple[int, int, int],
    draw_roi: bool,
    roi_width: int,
    roi_color: tuple[int, int, int],
) -> Image.Image:
    image = image.convert("RGB")
    content_width = image.width - panel_gap
    if panel_gap < 0:
        raise ValueError("--panel_gap must be non-negative")
    if content_width <= 0 or content_width % 2 != 0:
        raise ValueError(
            f"Image width {image.width} minus panel gap {panel_gap} "
            "must be a positive even number"
        )
    panel_width = content_width // 2
    panel_height = image.height
    right_offset = panel_width + panel_gap

    _validate_roi(
        left_roi,
        panel_width=panel_width,
        panel_height=panel_height,
        name="--roi",
    )
    _validate_roi(
        right_roi,
        panel_width=panel_width,
        panel_height=panel_height,
        name="--right_roi",
    )

    framed_width = inset_size[0] + 2 * border_width
    framed_height = inset_size[1] + 2 * border_width
    if (
        margin < 0
        or margin + framed_width > panel_width
        or margin + framed_height > panel_height
    ):
        raise ValueError(
            "Inset plus margin does not fit inside each panel: "
            f"panel={panel_width}x{panel_height}, "
            f"inset_with_border={framed_width}x{framed_height}, "
            f"margin={margin}"
        )

    left_panel = image.crop((0, 0, panel_width, panel_height))
    right_panel = image.crop(
        (right_offset, 0, right_offset + panel_width, panel_height)
    )
    left_inset = _make_framed_inset(
        left_panel,
        left_roi,
        inset_size=inset_size,
        border_width=border_width,
        border_color=border_color,
    )
    right_inset = _make_framed_inset(
        right_panel,
        right_roi,
        inset_size=inset_size,
        border_width=border_width,
        border_color=border_color,
    )

    result = image.copy()
    if draw_roi:
        if roi_width <= 0:
            raise ValueError("--roi_width must be positive")
        draw = ImageDraw.Draw(result)
        draw.rectangle(left_roi, outline=roi_color, width=roi_width)
        shifted_right_roi = (
            right_roi[0] + right_offset,
            right_roi[1],
            right_roi[2] + right_offset,
            right_roi[3],
        )
        draw.rectangle(
            shifted_right_roi,
            outline=roi_color,
            width=roi_width,
        )

    result.paste(left_inset, (margin, margin))
    result.paste(right_inset, (right_offset + margin, margin))
    return result


def _parse_color(value: str, name: str) -> tuple[int, int, int]:
    try:
        return ImageColor.getrgb(value)
    except ValueError as error:
        raise ValueError(f"{name}: invalid color {value!r}") from error


def _save_atomic(
    image: Image.Image,
    output_path: Path,
    *,
    jpeg_quality: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f".{output_path.stem}.tmp-{os.getpid()}{output_path.suffix}"
    )
    save_kwargs: dict[str, object] = {}
    if output_path.suffix.lower() in {".jpg", ".jpeg"}:
        save_kwargs.update(quality=jpeg_quality, subsampling=0)
    try:
        image.save(temporary, **save_kwargs)
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Crop corresponding regions from a side-by-side image pair, "
            "magnify them, and place the insets at both top-left corners"
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        required=True,
        metavar=("X", "Y", "WIDTH", "HEIGHT"),
        help="ROI relative to the left panel; reused on the right by default",
    )
    parser.add_argument(
        "--right_roi",
        type=int,
        nargs=4,
        metavar=("X", "Y", "WIDTH", "HEIGHT"),
        help="optional different ROI relative to the right panel",
    )
    parser.add_argument(
        "--inset_size",
        type=int,
        nargs=2,
        default=(112, 112),
        metavar=("WIDTH", "HEIGHT"),
    )
    parser.add_argument(
        "--panel_gap",
        type=int,
        default=0,
        help="number of separator pixels between the two equally sized panels",
    )
    parser.add_argument("--margin", type=int, default=8)
    parser.add_argument("--border_width", type=int, default=3)
    parser.add_argument("--border_color", default="white")
    parser.add_argument(
        "--draw_roi",
        action="store_true",
        help="also draw a box around the source region in both panels",
    )
    parser.add_argument("--roi_width", type=int, default=3)
    parser.add_argument("--roi_color", default="#FFD700")
    parser.add_argument("--jpeg_quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run(args: argparse.Namespace) -> None:
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input image not found: {input_path}")
    if input_path == output_path:
        raise ValueError("--output must differ from --input")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}; pass --overwrite "
            "to replace it"
        )
    if args.border_width < 0:
        raise ValueError("--border_width must be non-negative")
    if len(args.inset_size) != 2 or min(args.inset_size) <= 0:
        raise ValueError("--inset_size dimensions must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg_quality must be in [1, 100]")

    left_roi = _roi_box(args.roi, "--roi")
    right_roi = _roi_box(
        args.right_roi if args.right_roi is not None else args.roi,
        "--right_roi",
    )
    with Image.open(input_path) as opened:
        image = ImageOps.exif_transpose(opened)
        result = add_zoom_insets(
            image,
            left_roi=left_roi,
            right_roi=right_roi,
            panel_gap=args.panel_gap,
            inset_size=tuple(args.inset_size),
            margin=args.margin,
            border_width=args.border_width,
            border_color=_parse_color(
                args.border_color,
                "--border_color",
            ),
            draw_roi=args.draw_roi,
            roi_width=args.roi_width,
            roi_color=_parse_color(args.roi_color, "--roi_color"),
        )

    _save_atomic(
        result,
        output_path,
        jpeg_quality=args.jpeg_quality,
    )
    print(f"input:  {input_path}")
    print(
        f"layout: left={result.width // 2}x{result.height}, "
        f"right={result.width // 2}x{result.height}"
    )
    print(
        f"ROI:    left={args.roi}, "
        f"right={args.right_roi or args.roi}"
    )
    print(f"output: {output_path}")


if __name__ == "__main__":
    run(get_args_parser().parse_args())
