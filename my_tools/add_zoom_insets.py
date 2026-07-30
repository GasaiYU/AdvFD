"""Turn a side-by-side image pair into a one-by-four zoom strip.

The layouts are::

    input:  left image | right image
    output: left image | left zoom | right image | right zoom

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


def _make_zoom_panel(
    panel: Image.Image,
    roi: tuple[int, int, int, int],
) -> Image.Image:
    return panel.crop(roi).resize(
        panel.size,
        resample=Image.Resampling.LANCZOS,
    )


def make_zoom_strip(
    image: Image.Image,
    *,
    left_roi: tuple[int, int, int, int],
    right_roi: tuple[int, int, int, int],
    panel_gap: int,
    draw_roi: bool,
    roi_width: int,
    roi_color: tuple[int, int, int],
) -> Image.Image:
    image = image.convert("RGB")
    if panel_gap < 0:
        raise ValueError("--panel_gap must be non-negative")
    content_width = image.width - panel_gap
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

    left_panel = image.crop((0, 0, panel_width, panel_height))
    right_panel = image.crop(
        (right_offset, 0, right_offset + panel_width, panel_height)
    )
    left_zoom = _make_zoom_panel(left_panel, left_roi)
    right_zoom = _make_zoom_panel(right_panel, right_roi)

    if draw_roi:
        if roi_width <= 0:
            raise ValueError("--roi_width must be positive")
        left_panel = left_panel.copy()
        right_panel = right_panel.copy()
        ImageDraw.Draw(left_panel).rectangle(
            left_roi,
            outline=roi_color,
            width=roi_width,
        )
        ImageDraw.Draw(right_panel).rectangle(
            right_roi,
            outline=roi_color,
            width=roi_width,
        )

    result = Image.new("RGB", (4 * panel_width, panel_height))
    for index, panel in enumerate(
        (left_panel, left_zoom, right_panel, right_zoom)
    ):
        result.paste(panel, (index * panel_width, 0))
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
            "Create a one-by-four strip containing the left image, its "
            "magnified ROI, the right image, and its magnified ROI"
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
        "--panel_gap",
        type=int,
        default=0,
        help="number of separator pixels between the two equally sized panels",
    )
    parser.add_argument(
        "--draw_roi",
        action="store_true",
        help="draw source-region boxes on the two full-image panels",
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
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg_quality must be in [1, 100]")

    left_roi = _roi_box(args.roi, "--roi")
    right_roi = _roi_box(
        args.right_roi if args.right_roi is not None else args.roi,
        "--right_roi",
    )
    with Image.open(input_path) as opened:
        image = ImageOps.exif_transpose(opened)
        result = make_zoom_strip(
            image,
            left_roi=left_roi,
            right_roi=right_roi,
            panel_gap=args.panel_gap,
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
        "layout: left | left zoom | right | right zoom; "
        f"panel={result.width // 4}x{result.height}"
    )
    print(
        f"ROI:    left={args.roi}, "
        f"right={args.right_roi or args.roi}"
    )
    print(f"output: {output_path}")


if __name__ == "__main__":
    run(get_args_parser().parse_args())
