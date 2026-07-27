"""Build selected two-panel paper comparisons from JiT-L three-way images.

Each source image is expected to contain three 256x256 panels:

    JiT-L | FD-Loss | AdvFD

The ID file controls which pair is exported:

* ``123``  -> FD-Loss | AdvFD
* ``123-`` -> skip
* ``123*`` -> JiT-L | AdvFD
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps


DEFAULT_ID_FILE = Path(
    "/mmu-vcg/gaomingju/data/FD-Loss/quali/id_jit_quali.txt"
)
DEFAULT_INPUT_DIR = Path(
    "/mmu-vcg/gaomingju/data/FD-Loss/quali/jit_l_threeway"
)
IMAGE_SUFFIXES = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
ENTRY_PATTERN = re.compile(r"^\s*(\d+)\s*([-*]?)\s*$")
FILENAME_ID_PATTERN = re.compile(r"^\s*(\d+)")


@dataclass(frozen=True)
class Selection:
    image_id: int
    marker: str
    line_number: int

    @property
    def skipped(self) -> bool:
        return self.marker == "-"

    @property
    def panel_indices(self) -> tuple[int, int]:
        if self.marker == "*":
            return (0, 2)
        if self.marker == "":
            return (1, 2)
        raise ValueError(f"Skipped ID {self.image_id} has no panel pair")

    @property
    def panel_names(self) -> tuple[str, str]:
        if self.marker == "*":
            return ("JiT-L", "AdvFD")
        if self.marker == "":
            return ("FD-Loss", "AdvFD")
        raise ValueError(f"Skipped ID {self.image_id} has no panel pair")


def parse_id_file(path: Path) -> list[Selection]:
    if not path.is_file():
        raise FileNotFoundError(f"ID file not found: {path}")

    selections: list[Selection] = []
    seen: dict[int, int] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        match = ENTRY_PATTERN.fullmatch(raw_line)
        if match is None:
            raise ValueError(
                f"{path}:{line_number}: expected ID, ID-, or ID*, "
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


def discover_sources(
    input_dir: Path,
    wanted_ids: set[int],
) -> dict[int, Path]:
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
        preview = ", ".join(str(image_id) for image_id in missing[:20])
        suffix = " ..." if len(missing) > 20 else ""
        raise FileNotFoundError(
            f"Missing source image for {len(missing)} selected IDs: "
            f"{preview}{suffix}"
        )
    return matches


def make_pair(
    source: Path,
    panel_indices: tuple[int, int],
    *,
    panel_size: int,
) -> Image.Image:
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")

    expected_size = (panel_size * 3, panel_size)
    if image.size != expected_size:
        raise ValueError(
            f"{source}: expected a three-panel image of size "
            f"{expected_size[0]}x{expected_size[1]}, got "
            f"{image.width}x{image.height}"
        )

    result = Image.new("RGB", (panel_size * 2, panel_size))
    for output_index, source_index in enumerate(panel_indices):
        left = source_index * panel_size
        panel = image.crop(
            (left, 0, left + panel_size, panel_size)
        )
        result.paste(panel, (output_index * panel_size, 0))
    return result


def save_atomic(image: Image.Image, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.tmp-{os.getpid()}{destination.suffix}"
    )
    try:
        image.save(temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create JiT-L/FD-Loss versus AdvFD two-panel comparisons "
            "from selected three-way images"
        )
    )
    parser.add_argument(
        "--id_file",
        type=Path,
        default=DEFAULT_ID_FILE,
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("paper/jit_l_pair_comparisons_quali"),
    )
    parser.add_argument("--panel_size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run(args: argparse.Namespace) -> None:
    id_file = args.id_file.expanduser().resolve()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if args.panel_size <= 0:
        raise ValueError("--panel_size must be positive")

    selections = parse_id_file(id_file)
    active = [selection for selection in selections if not selection.skipped]
    sources = discover_sources(
        input_dir,
        {selection.image_id for selection in active},
    )

    destinations = {
        selection.image_id: output_dir
        / (
            f"{selection.image_id:05d}_"
            f"{'jit_l' if selection.marker == '*' else 'fd_loss'}"
            "_advfd.png"
        )
        for selection in active
    }
    existing = [
        destination
        for destination in destinations.values()
        if destination.exists()
    ]
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        existing.append(manifest_path)
    if existing and not args.overwrite:
        raise FileExistsError(
            f"{len(existing)} output files already exist under "
            f"{output_dir}; pass --overwrite to replace them"
        )

    records: list[dict[str, object]] = []
    for selection in active:
        source = sources[selection.image_id]
        destination = destinations[selection.image_id]
        pair = make_pair(
            source,
            selection.panel_indices,
            panel_size=args.panel_size,
        )
        save_atomic(pair, destination)
        left_name, right_name = selection.panel_names
        records.append(
            {
                "id": selection.image_id,
                "marker": selection.marker,
                "source": str(source),
                "output": str(destination),
                "left": left_name,
                "right": right_name,
            }
        )
        print(
            f"[{selection.image_id:05d}] "
            f"{left_name} | {right_name} -> {destination}"
        )

    skipped = [
        selection.image_id
        for selection in selections
        if selection.skipped
    ]
    manifest = {
        "id_file": str(id_file),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "panel_size": args.panel_size,
        "source_layout": ["JiT-L", "FD-Loss", "AdvFD"],
        "num_entries": len(selections),
        "num_outputs": len(records),
        "num_skipped": len(skipped),
        "skipped_ids": skipped,
        "outputs": records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest_path.with_name(
        f".manifest.tmp-{os.getpid()}.json"
    )
    try:
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_manifest, manifest_path)
    finally:
        temporary_manifest.unlink(missing_ok=True)

    print(
        f"done: outputs={len(records)} skipped={len(skipped)} "
        f"directory={output_dir}"
    )


if __name__ == "__main__":
    run(get_args_parser().parse_args())
