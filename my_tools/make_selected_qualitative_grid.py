"""Compose seven selected side-by-side pairs into a paper-ready grid.

For every source image, the left half is placed in the left block and the
right half is placed at the corresponding location in the right block.
The layout follows the requested reference: three large images on the top
row and four smaller images on the bottom row.

By default every comparison image in ``--input_dir`` is used, so the
directory must hold exactly seven of them. Pass ``--filenames`` to pick a
specific subset and order instead.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


TOP_COUNT = 3
BOTTOM_COUNT = 4
PANEL_COUNT = TOP_COUNT + BOTTOM_COUNT
# Both rows must span the same block width, so the width is the smallest
# common multiple of the two row lengths: 3 top tiles of 4 units each and
# 4 bottom tiles of 3 units each both span 12 units.
BLOCK_UNITS = math.lcm(TOP_COUNT, BOTTOM_COUNT)
TOP_UNITS = BLOCK_UNITS // TOP_COUNT
BOTTOM_UNITS = BLOCK_UNITS // BOTTOM_COUNT
BLOCK_HEIGHT_UNITS = TOP_UNITS + BOTTOM_UNITS

IMAGE_SUFFIXES = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
DEFAULT_OUTPUT_NAME = "selected_7_qualitative_grid.png"


def _load_pair(path: Path) -> tuple[Image.Image, Image.Image]:
    if not path.is_file():
        raise FileNotFoundError(f"Comparison image not found: {path}")
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    if image.width % 2 != 0:
        raise ValueError(
            f"{path}: width must be even, got {image.width}"
        )
    panel_size = image.width // 2
    if image.height != panel_size:
        raise ValueError(
            f"{path}: expected two square panels (2H x H), got "
            f"{image.width}x{image.height}"
        )
    return (
        image.crop((0, 0, panel_size, panel_size)),
        image.crop((panel_size, 0, image.width, panel_size)),
    )


def _load_font(font_path: Path | None, size: int) -> ImageFont.ImageFont:
    candidates = []
    if font_path is not None:
        candidates.append(font_path)
    candidates.extend(
        [
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path(
                "/usr/share/fonts/truetype/liberation2/"
                "LiberationSans-Regular.ttf"
            ),
            Path("/System/Library/Fonts/Helvetica.ttc"),
            Path("/Library/Fonts/Arial.ttf"),
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _paste_block(
    canvas: Image.Image,
    panels: list[Image.Image],
    *,
    origin_x: int,
    origin_y: int,
    unit: int,
) -> list[dict[str, int]]:
    if len(panels) != PANEL_COUNT:
        raise ValueError(f"Expected {PANEL_COUNT} panels, got {len(panels)}")

    placements: list[dict[str, int]] = []
    top_size = TOP_UNITS * unit
    bottom_size = BOTTOM_UNITS * unit
    for index, panel in enumerate(panels):
        if index < TOP_COUNT:
            size = top_size
            x = origin_x + index * size
            y = origin_y
        else:
            size = bottom_size
            x = origin_x + (index - TOP_COUNT) * size
            y = origin_y + top_size
        resized = panel.resize(
            (size, size),
            resample=Image.Resampling.LANCZOS,
        )
        canvas.paste(resized, (x, y))
        placements.append(
            {
                "x": x,
                "y": y,
                "width": size,
                "height": size,
            }
        )
    return placements


def _draw_dashed_separator(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    top: int,
    bottom: int,
    width: int,
    dash: int,
    gap: int,
    color: tuple[int, int, int],
) -> None:
    y = top
    while y < bottom:
        draw.line(
            (x, y, x, min(y + dash, bottom)),
            fill=color,
            width=width,
        )
        y += dash + gap


def _draw_title_badge(
    canvas: Image.Image,
    *,
    text: str,
    x: int,
    y: int,
    font: ImageFont.ImageFont,
    padding_x: int,
    padding_y: int,
) -> None:
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    badge_box = (
        x,
        y,
        x + text_width + 2 * padding_x,
        y + text_height + 2 * padding_y,
    )
    draw.rounded_rectangle(
        badge_box,
        radius=max(2, padding_y),
        fill=(30, 35, 42, 210),
    )
    draw.text(
        (x + padding_x, y + padding_y - bbox[1]),
        text,
        font=font,
        fill=(255, 255, 255, 255),
    )
    canvas.alpha_composite(overlay)


def compose_grid(
    pair_paths: list[Path],
    *,
    seed: int,
    shuffle: bool,
    unit: int,
    group_gap: int,
    outer_padding: int,
    show_titles: bool,
    left_title: str,
    right_title: str,
    font_path: Path | None,
    font_size: int,
) -> tuple[Image.Image, list[dict[str, object]]]:
    if len(pair_paths) != PANEL_COUNT:
        raise ValueError(
            f"Exactly {PANEL_COUNT} comparison images are required, got "
            f"{len(pair_paths)}"
        )
    if unit <= 0:
        raise ValueError("--unit must be positive")
    if group_gap < 0 or outer_padding < 0:
        raise ValueError("--group_gap and --outer_padding must be non-negative")

    ordered_paths = list(pair_paths)
    if shuffle:
        random.Random(seed).shuffle(ordered_paths)
    pairs = [_load_pair(path) for path in ordered_paths]
    left_panels = [pair[0] for pair in pairs]
    right_panels = [pair[1] for pair in pairs]

    block_width = BLOCK_UNITS * unit
    block_height = BLOCK_HEIGHT_UNITS * unit
    canvas_width = 2 * block_width + group_gap + 2 * outer_padding
    canvas_height = block_height + 2 * outer_padding
    canvas = Image.new(
        "RGBA",
        (canvas_width, canvas_height),
        (255, 255, 255, 255),
    )
    left_x = outer_padding
    right_x = outer_padding + block_width + group_gap
    top_y = outer_padding

    left_placements = _paste_block(
        canvas,
        left_panels,
        origin_x=left_x,
        origin_y=top_y,
        unit=unit,
    )
    right_placements = _paste_block(
        canvas,
        right_panels,
        origin_x=right_x,
        origin_y=top_y,
        unit=unit,
    )

    draw = ImageDraw.Draw(canvas)
    _draw_dashed_separator(
        draw,
        x=outer_padding + block_width + group_gap // 2,
        top=outer_padding,
        bottom=outer_padding + block_height,
        width=max(2, unit // 16),
        dash=max(6, unit // 3),
        gap=max(5, unit // 4),
        color=(120, 120, 120),
    )

    if show_titles:
        font = _load_font(font_path, font_size)
        inset = max(6, unit // 4)
        _draw_title_badge(
            canvas,
            text=left_title,
            x=left_x + inset,
            y=top_y + inset,
            font=font,
            padding_x=max(8, unit // 5),
            padding_y=max(5, unit // 8),
        )
        _draw_title_badge(
            canvas,
            text=right_title,
            x=right_x + inset,
            y=top_y + inset,
            font=font,
            padding_x=max(8, unit // 5),
            padding_y=max(5, unit // 8),
        )

    layout = []
    for index, path in enumerate(ordered_paths):
        layout.append(
            {
                "position": index,
                "row": "top" if index < TOP_COUNT else "bottom",
                "source": str(path),
                "filename": path.name,
                "left_box": left_placements[index],
                "right_box": right_placements[index],
            }
        )
    return canvas.convert("RGB"), layout


def _save_image_atomic(
    image: Image.Image,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f".{output_path.stem}.tmp-{os.getpid()}{output_path.suffix}"
    )
    try:
        save_kwargs: dict[str, object] = {}
        if output_path.suffix.lower() == ".png":
            save_kwargs["dpi"] = (300, 300)
        elif output_path.suffix.lower() in {".jpg", ".jpeg"}:
            save_kwargs.update(quality=98, subsampling=0, dpi=(300, 300))
        image.save(temporary, **save_kwargs)
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)


def discover_filenames(input_dir: Path, exclude: set[str]) -> list[str]:
    """Return every comparison image in ``input_dir``, sorted by name."""
    found = sorted(
        path.name
        for path in input_dir.iterdir()
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in IMAGE_SUFFIXES
        and path.name not in exclude
    )
    if len(found) != PANEL_COUNT:
        preview = ", ".join(found[:12]) or "none"
        suffix = " ..." if len(found) > 12 else ""
        raise ValueError(
            f"{input_dir} holds {len(found)} comparison image(s), but the "
            f"grid needs exactly {PANEL_COUNT}. Found: {preview}{suffix}. "
            f"Pass --filenames to select {PANEL_COUNT} of them explicitly."
        )
    return found


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            f"Build a matched {PANEL_COUNT}-image qualitative grid: all left "
            "halves on the left, all right halves on the right"
        )
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("paper/jit_l_pair_comparisons"),
    )
    parser.add_argument(
        "--filenames",
        nargs="+",
        default=None,
        help=(
            f"exactly {PANEL_COUNT} filenames relative to --input_dir; "
            "defaults to every comparison image in that directory"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=f"default: <input_dir>/{DEFAULT_OUTPUT_NAME}",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--preserve_order",
        action="store_true",
        help="use --filenames order instead of seeded random order",
    )
    parser.add_argument(
        "--unit",
        type=int,
        default=60,
        help="layout unit; 60 gives 240px top and 180px bottom tiles",
    )
    parser.add_argument("--group_gap", type=int, default=36)
    parser.add_argument("--outer_padding", type=int, default=0)
    parser.add_argument("--left_title", default="JiT-L / FD-Loss")
    parser.add_argument("--right_title", default="AdvFD (Ours)")
    parser.add_argument(
        "--no_titles",
        action="store_false",
        dest="show_titles",
    )
    parser.set_defaults(show_titles=True)
    parser.add_argument("--font_path", type=Path)
    parser.add_argument("--font_size", type=int, default=27)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run(args: argparse.Namespace) -> None:
    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")
    if args.font_size <= 0:
        raise ValueError("--font_size must be positive")

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else input_dir / DEFAULT_OUTPUT_NAME
    )
    manifest_path = output_path.with_suffix(".json")

    if args.filenames is None:
        # A previous grid written into this directory is an output, never a
        # panel source, so it must not be picked up on a re-run.
        filenames = discover_filenames(
            input_dir,
            exclude={output_path.name}
            if output_path.parent == input_dir
            else set(),
        )
    else:
        filenames = list(args.filenames)
        if len(filenames) != PANEL_COUNT:
            raise ValueError(
                f"--filenames requires exactly {PANEL_COUNT} entries, got "
                f"{len(filenames)}"
            )
        if len(set(filenames)) != PANEL_COUNT:
            raise ValueError("--filenames contains duplicates")

    pair_paths = [input_dir / filename for filename in filenames]
    existing = [
        path for path in (output_path, manifest_path) if path.exists()
    ]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {existing[0]}; pass --overwrite "
            "to replace it"
        )

    image, layout = compose_grid(
        pair_paths,
        seed=args.seed,
        shuffle=not args.preserve_order,
        unit=args.unit,
        group_gap=args.group_gap,
        outer_padding=args.outer_padding,
        show_titles=args.show_titles,
        left_title=args.left_title,
        right_title=args.right_title,
        font_path=args.font_path,
        font_size=args.font_size,
    )
    _save_image_atomic(image, output_path)

    manifest = {
        "input_dir": str(input_dir),
        "output": str(output_path),
        "width": image.width,
        "height": image.height,
        "seed": args.seed,
        "shuffled": not args.preserve_order,
        "source_layout": "left_half | right_half",
        "grid_layout": (
            f"{TOP_COUNT} large images above {BOTTOM_COUNT} small images"
        ),
        "num_panels": PANEL_COUNT,
        "filenames_source": (
            "input_dir_scan" if args.filenames is None else "explicit"
        ),
        "left_title": args.left_title if args.show_titles else None,
        "right_title": args.right_title if args.show_titles else None,
        "layout": layout,
    }
    temporary_manifest = manifest_path.with_name(
        f".{manifest_path.stem}.tmp-{os.getpid()}.json"
    )
    try:
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_manifest, manifest_path)
    finally:
        temporary_manifest.unlink(missing_ok=True)

    print(f"output: {output_path} ({image.width}x{image.height})")
    print(f"manifest: {manifest_path}")
    for item in layout:
        print(
            f"  position={item['position']} row={item['row']} "
            f"file={item['filename']}"
        )


if __name__ == "__main__":
    run(get_args_parser().parse_args())
