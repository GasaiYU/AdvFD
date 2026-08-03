#!/usr/bin/env python3
"""Build shuffled FD-Loss vs AdvFD pairs for a human study.

Each source image under ``--input_dir`` is a three-panel composite laid out
left to right as::

    baseline | FD-Loss | AdvFD (ours)

For every ID in the ID file (default ``my_tools/paper_id.txt``) the FD-Loss
and AdvFD panels are cropped out and pasted side by side in a random
left/right order.  Output names carry no method information, so raters cannot
read the answer off the file listing::

    images/001_07513.png

The shuffle is recorded beside -- not inside -- the rater-facing folder:

* ``manifest.json``   full record, including source paths and the RNG seed
* ``answer_key.csv``  ``trial,image,id,left,right,ours_side,swapped``

``swapped`` is the shuffle index: ``0`` keeps FD-Loss on the left, ``1`` puts
AdvFD on the left.  The assignment is balanced by default (half the trials
swapped) and trial order is randomised; pass ``--seed`` to reproduce a study.

ID file syntax, one entry per line, ``#`` starts a comment::

    7513     compare FD-Loss against AdvFD
    7513*    compare the baseline against AdvFD instead
    7513-    skip this ID
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps

DEFAULT_ID_FILE = Path(__file__).resolve().parent / "paper_id.txt"
DEFAULT_INPUT_DIR = Path(
    "/mmu-vcg/gaomingju/data/FD-Loss/quali/jit_l_threeway"
)
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
ENTRY_PATTERN = re.compile(r"^(\d+)([-*]?)$")
FILENAME_ID_PATTERN = re.compile(r"^(\d+)")
BASELINE_PANEL, FD_PANEL, OURS_PANEL = 0, 1, 2


@dataclass(frozen=True)
class Selection:
    """One line of the ID file."""

    image_id: int
    marker: str
    line_number: int

    @property
    def skipped(self) -> bool:
        return self.marker == "-"

    @property
    def reference_panel(self) -> int:
        """Panel index compared against ours."""
        return BASELINE_PANEL if self.marker == "*" else FD_PANEL


@dataclass(frozen=True)
class Trial:
    """A single rater-facing comparison."""

    trial: int
    selection: Selection
    source: Path
    swapped: bool
    reference_name: str
    ours_name: str

    @property
    def left_name(self) -> str:
        return self.ours_name if self.swapped else self.reference_name

    @property
    def right_name(self) -> str:
        return self.reference_name if self.swapped else self.ours_name

    @property
    def ours_side(self) -> str:
        return "left" if self.swapped else "right"


def parse_id_file(path: Path) -> list[Selection]:
    if not path.is_file():
        raise FileNotFoundError(f"ID file not found: {path}")

    selections: list[Selection] = []
    seen: dict[int, int] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, raw_line in enumerate(lines, start=1):
        entry = raw_line.split("#", 1)[0].strip()
        if not entry:
            continue
        match = ENTRY_PATTERN.fullmatch(entry)
        if match is None:
            raise ValueError(
                f"{path}:{line_number}: expected ID, ID*, or ID-, "
                f"got {raw_line!r}"
            )
        image_id = int(match.group(1))
        if image_id in seen:
            raise ValueError(
                f"{path}:{line_number}: duplicate ID {image_id}; "
                f"first seen on line {seen[image_id]}"
            )
        seen[image_id] = line_number
        selections.append(
            Selection(
                image_id=image_id,
                marker=match.group(2),
                line_number=line_number,
            )
        )
    if not selections:
        raise ValueError(f"ID file contains no entries: {path}")
    return selections


def discover_sources(input_dir: Path, wanted_ids: set[int]) -> dict[int, Path]:
    """Map each wanted ID to the ``%05d_*.png`` source that starts with it."""
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")

    matches: dict[int, Path] = {}
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        match = FILENAME_ID_PATTERN.match(path.name)
        if match is None:
            continue
        image_id = int(match.group(1))
        if image_id not in wanted_ids:
            continue
        previous = matches.get(image_id)
        if previous is not None:
            raise ValueError(
                f"Multiple source images start with ID {image_id}: "
                f"{previous} and {path}"
            )
        matches[image_id] = path

    missing = sorted(wanted_ids - matches.keys())
    if missing:
        preview = ", ".join(f"{image_id:05d}" for image_id in missing[:20])
        suffix = " ..." if len(missing) > 20 else ""
        raise FileNotFoundError(
            f"Missing source image for {len(missing)} selected IDs under "
            f"{input_dir}: {preview}{suffix}"
        )
    return matches


def split_panels(source: Path, panel_size: int | None) -> list[Image.Image]:
    """Cut a three-panel composite into its three panels."""
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")

    if panel_size is None:
        if image.width % 3:
            raise ValueError(
                f"{source}: width {image.width} is not divisible by 3; "
                f"pass --panel_size to override panel detection"
            )
        panel_size = image.width // 3
    elif image.width != panel_size * 3:
        raise ValueError(
            f"{source}: expected a three-panel image of width "
            f"{panel_size * 3}, got {image.width}"
        )
    if panel_size <= 0:
        raise ValueError(f"{source}: computed a non-positive panel size")

    return [
        image.crop(
            (index * panel_size, 0, (index + 1) * panel_size, image.height)
        )
        for index in range(3)
    ]


def make_pair(
    left: Image.Image,
    right: Image.Image,
    *,
    gap: int,
    gap_color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    if left.size != right.size:
        raise ValueError(
            f"panels must have the same size, got {left.size} and {right.size}"
        )
    width, height = left.size
    canvas = Image.new("RGB", (width * 2 + gap, height), gap_color)
    canvas.paste(left, (0, 0))
    canvas.paste(right, (width + gap, 0))
    return canvas


def save_atomic(image: Image.Image, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.tmp-{os.getpid()}{destination.suffix or '.png'}"
    )
    try:
        image.save(temporary, format="PNG")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def write_text_atomic(text: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.tmp-{os.getpid()}{destination.suffix}"
    )
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def assign_swaps(count: int, rng: random.Random, balanced: bool) -> list[bool]:
    """Decide which trials put ours on the left."""
    if not balanced:
        return [rng.random() < 0.5 for _ in range(count)]
    # Split as evenly as possible; break an odd count with a coin flip so the
    # extra trial does not systematically favour one side.
    swapped_count = count // 2
    if count % 2 and rng.random() < 0.5:
        swapped_count += 1
    swaps = [True] * swapped_count + [False] * (count - swapped_count)
    rng.shuffle(swaps)
    return swaps


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Crop FD-Loss and AdvFD panels out of three-way composites and "
            "write left/right shuffled pairs for a human study"
        )
    )
    parser.add_argument(
        "--id_file",
        type=Path,
        default=DEFAULT_ID_FILE,
        help="one ID per line; ID* compares the baseline, ID- skips",
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="directory of %%05d_*.png three-panel composites",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("paper/human_study"),
        help="images/ goes to raters; manifest and answer key stay here",
    )
    parser.add_argument(
        "--panel_size",
        type=int,
        default=None,
        help="panel width in pixels (default: source width / 3)",
    )
    parser.add_argument(
        "--gap",
        type=int,
        default=0,
        help="white separator in pixels between the two panels",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for the shuffle; recorded in the manifest",
    )
    parser.add_argument(
        "--no_balanced",
        dest="balanced",
        action="store_false",
        help="flip each trial independently instead of balancing the sides",
    )
    parser.add_argument(
        "--no_shuffle_order",
        dest="shuffle_order",
        action="store_false",
        help="keep ID-file order instead of randomising trial order",
    )
    parser.add_argument(
        "--baseline_name",
        default="Baseline",
        help="answer-key label for panel 1",
    )
    parser.add_argument(
        "--fd_name",
        default="FD-Loss",
        help="answer-key label for panel 2",
    )
    parser.add_argument(
        "--ours_name",
        default="AdvFD",
        help="answer-key label for panel 3",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def build_trials(
    selections: list[Selection],
    sources: dict[int, Path],
    *,
    rng: random.Random,
    balanced: bool,
    shuffle_order: bool,
    reference_names: dict[int, str],
    ours_name: str,
) -> list[Trial]:
    active = [selection for selection in selections if not selection.skipped]
    if not active:
        raise ValueError("every ID in the ID file is marked as skipped")

    ordered = list(active)
    if shuffle_order:
        rng.shuffle(ordered)

    swaps = assign_swaps(len(ordered), rng, balanced)
    return [
        Trial(
            trial=index,
            selection=selection,
            source=sources[selection.image_id],
            swapped=swapped,
            reference_name=reference_names[selection.reference_panel],
            ours_name=ours_name,
        )
        for index, (selection, swapped) in enumerate(
            zip(ordered, swaps), start=1
        )
    ]


def answer_key_csv(trials: list[Trial], names: list[str]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(
        ["trial", "image", "id", "left", "right", "ours_side", "swapped"]
    )
    for trial, name in zip(trials, names):
        writer.writerow(
            [
                trial.trial,
                name,
                f"{trial.selection.image_id:05d}",
                trial.left_name,
                trial.right_name,
                trial.ours_side,
                int(trial.swapped),
            ]
        )
    return buffer.getvalue()


def run(args: argparse.Namespace) -> Path:
    id_file = args.id_file.expanduser().resolve()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    image_dir = output_dir / "images"
    manifest_path = output_dir / "manifest.json"
    answer_key_path = output_dir / "answer_key.csv"

    if args.panel_size is not None and args.panel_size <= 0:
        raise ValueError("--panel_size must be positive")
    if args.gap < 0:
        raise ValueError("--gap must be non-negative")

    selections = parse_id_file(id_file)
    active = [selection for selection in selections if not selection.skipped]
    sources = discover_sources(
        input_dir,
        {selection.image_id for selection in active},
    )

    seed = args.seed if args.seed is not None else random.randrange(2**32)
    rng = random.Random(seed)
    reference_names = {
        BASELINE_PANEL: args.baseline_name,
        FD_PANEL: args.fd_name,
    }
    trials = build_trials(
        selections,
        sources,
        rng=rng,
        balanced=args.balanced,
        shuffle_order=args.shuffle_order,
        reference_names=reference_names,
        ours_name=args.ours_name,
    )

    width = max(3, len(str(len(trials))))
    names = [
        f"{trial.trial:0{width}d}_{trial.selection.image_id:05d}.png"
        for trial in trials
    ]
    destinations = [image_dir / name for name in names]
    existing = [path for path in destinations if path.exists()]
    existing.extend(
        path for path in (manifest_path, answer_key_path) if path.exists()
    )
    if existing and not args.overwrite:
        raise FileExistsError(
            f"{len(existing)} output files already exist under {output_dir}; "
            f"pass --overwrite to replace them"
        )

    records: list[dict[str, object]] = []
    for trial, name, destination in zip(trials, names, destinations):
        panels = split_panels(trial.source, args.panel_size)
        reference = panels[trial.selection.reference_panel]
        ours = panels[OURS_PANEL]
        left, right = (
            (ours, reference) if trial.swapped else (reference, ours)
        )
        pair = make_pair(left, right, gap=args.gap)
        save_atomic(pair, destination)
        records.append(
            {
                "trial": trial.trial,
                "image": name,
                "id": trial.selection.image_id,
                "marker": trial.selection.marker,
                "source": str(trial.source),
                "left": trial.left_name,
                "right": trial.right_name,
                "ours_side": trial.ours_side,
                "swapped": int(trial.swapped),
                "size": [pair.width, pair.height],
            }
        )
        print(
            f"[{trial.trial:0{width}d}] {trial.selection.image_id:05d} "
            f"{trial.left_name} | {trial.right_name} -> {destination}"
        )

    swapped_count = sum(record["swapped"] for record in records)
    skipped = [
        selection.image_id for selection in selections if selection.skipped
    ]
    manifest = {
        "id_file": str(id_file),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "image_dir": str(image_dir),
        "source_layout": [args.baseline_name, args.fd_name, args.ours_name],
        "ours_name": args.ours_name,
        "panel_size": args.panel_size,
        "gap": args.gap,
        "seed": seed,
        "balanced": args.balanced,
        "shuffle_order": args.shuffle_order,
        "num_entries": len(selections),
        "num_trials": len(records),
        "num_skipped": len(skipped),
        "skipped_ids": skipped,
        "ours_left": swapped_count,
        "ours_right": len(records) - swapped_count,
        "trials": records,
    }
    write_text_atomic(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        manifest_path,
    )
    write_text_atomic(answer_key_csv(trials, names), answer_key_path)

    print(
        f"done: trials={len(records)} skipped={len(skipped)} "
        f"ours_left={swapped_count} ours_right={len(records) - swapped_count} "
        f"seed={seed}"
    )
    print(f"images: {image_dir}")
    print(f"manifest: {manifest_path}")
    print(f"answer key: {answer_key_path}")
    return output_dir


if __name__ == "__main__":
    run(get_args_parser().parse_args())
