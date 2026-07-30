#!/usr/bin/env python3
"""Compare adversarial/reference feature RMS ratios with and without whitening.

The inputs may be result directories produced by
``my_tools/compute_fd_adv_norm_ratio_all.sh``, JSONL summaries, or individual
JSON result files. Checkpoints are aligned by the integer in ``step_*.pth``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WHITEN = Path(
    "/mmu-vcg/gaomingju/workspace/foundation/FD-Loss-Ours/"
    "my_tools/fd_adv_norm_ratio_all"
)
DEFAULT_NO_WHITEN = Path(
    "/mmu-vcg/gaomingju/workspace/foundation/FD-Loss-Ours/"
    "my_tools/fd_adv_norm_ratio_all_no_whiten"
)
STEP_RE = re.compile(r"(?:^|[/\\])step[_-]?(\d+)(?:\.pth)?$", re.IGNORECASE)


@dataclass(frozen=True)
class RMSPoint:
    step: int
    ratio: float
    checkpoint: str
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--whiten",
        type=Path,
        default=DEFAULT_WHITEN,
        help="Whitening result directory, summary.jsonl, or result JSON.",
    )
    parser.add_argument(
        "--no-whiten",
        type=Path,
        default=DEFAULT_NO_WHITEN,
        help="No-whitening result directory, summary.jsonl, or result JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "my_tools",
        help="Directory for PDF, PNG, CSV, and statistics JSON outputs.",
    )
    parser.add_argument(
        "--prefix",
        default="fig_rms_ratio_whitening",
        help="Output filename prefix.",
    )
    parser.add_argument(
        "--yscale",
        choices=("auto", "linear", "log"),
        default="auto",
        help="Scale for the RMS-ratio panel (auto uses log for a >=8x span).",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=112_500,
        help="Only include checkpoints up to this training step (default: 112500).",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Parse, align, and print statistics without importing matplotlib.",
    )
    return parser.parse_args()


def _json_records(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value
        return

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    yield value


def _input_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"input does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"input is neither a file nor a directory: {path}")

    # Per-checkpoint files are the source of truth. summary.jsonl can be stale
    # when a sweep is interrupted and later resumed, while the step JSON files
    # still contain all completed checkpoints.
    step_files = sorted(path.glob("step_*.json"))
    if step_files:
        return step_files

    summary = path / "summary.jsonl"
    if summary.is_file():
        return [summary]
    files = sorted(path.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no summary.jsonl or *.json files found in: {path}")
    return files


def _checkpoint_step(record: dict[str, Any], source: Path) -> tuple[int, str]:
    checkpoint = str(record.get("checkpoint_path", ""))
    match = STEP_RE.search(checkpoint)
    if match is None:
        match = STEP_RE.search(source.stem)
    if match is None:
        raise ValueError(
            f"cannot extract step_XXXX from checkpoint_path or filename: {source}"
        )
    return int(match.group(1)), checkpoint or source.stem


def load_points(path: Path) -> list[RMSPoint]:
    by_step: dict[int, RMSPoint] = {}
    for source in _input_files(path):
        for record in _json_records(source):
            step, checkpoint = _checkpoint_step(record, source)
            try:
                ratio = float(record["rms_ratio"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{source}: missing or invalid rms_ratio") from exc
            if not math.isfinite(ratio) or ratio <= 0.0:
                raise ValueError(f"{source}: rms_ratio must be finite and positive")
            if step in by_step:
                raise ValueError(f"duplicate checkpoint step {step} in {path}")
            by_step[step] = RMSPoint(step, ratio, checkpoint, record)
    if not by_step:
        raise ValueError(f"no RMS results loaded from: {path}")
    return [by_step[step] for step in sorted(by_step)]


def limit_points(points: list[RMSPoint], max_step: int, label: str) -> list[RMSPoint]:
    if max_step < 0:
        raise ValueError("max_step must be non-negative")
    limited = [point for point in points if point.step <= max_step]
    if not limited:
        raise ValueError(f"{label} input has no checkpoints at or before step {max_step}")
    return limited


def _median(values: list[float]) -> float:
    values = sorted(values)
    midpoint = len(values) // 2
    if len(values) % 2:
        return values[midpoint]
    return 0.5 * (values[midpoint - 1] + values[midpoint])


def align_and_summarize(
    whiten: list[RMSPoint], no_whiten: list[RMSPoint]
) -> tuple[list[tuple[int, float, float, float]], dict[str, Any]]:
    whiten_by_step = {point.step: point.ratio for point in whiten}
    no_whiten_by_step = {point.step: point.ratio for point in no_whiten}
    common_steps = sorted(whiten_by_step.keys() & no_whiten_by_step.keys())
    if not common_steps:
        raise ValueError("the two inputs have no checkpoint steps in common")

    paired = [
        (
            step,
            whiten_by_step[step],
            no_whiten_by_step[step],
            no_whiten_by_step[step] / whiten_by_step[step],
        )
        for step in common_steps
    ]
    inflation = [row[3] for row in paired]
    stats = {
        "num_whiten_checkpoints": len(whiten),
        "num_no_whiten_checkpoints": len(no_whiten),
        "num_paired_checkpoints": len(paired),
        "common_steps": common_steps,
        "median_whiten_rms_ratio": _median([row[1] for row in paired]),
        "median_no_whiten_rms_ratio": _median([row[2] for row in paired]),
        "median_inflation_factor": _median(inflation),
        "max_inflation_factor": max(inflation),
        "max_inflation_step": paired[inflation.index(max(inflation))][0],
        "fraction_no_whiten_larger": sum(value > 1.0 for value in inflation)
        / len(inflation),
    }
    return paired, stats


def validate_comparability(
    whiten: list[RMSPoint], no_whiten: list[RMSPoint]
) -> list[str]:
    warnings: list[str] = []
    fields = ("repr_model", "pool_type", "split", "images_used")
    for field in fields:
        left = {point.metadata.get(field) for point in whiten}
        right = {point.metadata.get(field) for point in no_whiten}
        left.discard(None)
        right.discard(None)
        if len(left) > 1:
            warnings.append(f"whiten input contains multiple {field} values: {left}")
        if len(right) > 1:
            warnings.append(f"no-whiten input contains multiple {field} values: {right}")
        if left and right and left != right:
            warnings.append(f"{field} differs: whiten={left}, no-whiten={right}")
    return warnings


def write_tables(
    output_dir: Path,
    prefix: str,
    paired: list[tuple[int, float, float, float]],
    stats: dict[str, Any],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{prefix}_paired.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["step", "with_whitening_rms_ratio", "no_whitening_rms_ratio", "inflation_factor"]
        )
        writer.writerows(paired)

    stats_path = output_dir / f"{prefix}_stats.json"
    stats_path.write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return csv_path, stats_path


def _step_scale(steps: list[int]) -> tuple[float, str]:
    if max(steps) >= 10_000:
        return 1_000.0, "Training step (k)"
    return 1.0, "Training step"


def make_figure(
    whiten: list[RMSPoint],
    no_whiten: list[RMSPoint],
    paired: list[tuple[int, float, float, float]],
    stats: dict[str, Any],
    output_dir: Path,
    prefix: str,
    requested_yscale: str,
) -> tuple[Path, Path]:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to render the figure; install it with "
            "`python -m pip install matplotlib`"
        ) from exc

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": 0.20,
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.9,
            "lines.markersize": 4.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    blue = "#0072B2"
    vermillion = "#D55E00"
    dark_gray = "#4D4D4D"
    light_orange = "#F6C8A8"

    all_steps = [point.step for point in whiten] + [point.step for point in no_whiten]
    step_divisor, step_label = _step_scale(all_steps)
    whiten_x = [point.step / step_divisor for point in whiten]
    no_whiten_x = [point.step / step_divisor for point in no_whiten]
    paired_x = [row[0] / step_divisor for row in paired]

    all_ratios = [point.ratio for point in whiten] + [point.ratio for point in no_whiten]
    span = max(all_ratios) / min(all_ratios)
    yscale = requested_yscale
    if yscale == "auto":
        yscale = "log" if span >= 8.0 else "linear"

    fig, ax_ratio = plt.subplots(figsize=(3.45, 2.65))

    ax_ratio.axhline(1.0, color=dark_gray, linestyle=(0, (3, 2)), linewidth=1.0, zorder=1)
    paired_whiten = [row[1] for row in paired]
    paired_no_whiten = [row[2] for row in paired]
    ax_ratio.fill_between(
        paired_x,
        paired_whiten,
        paired_no_whiten,
        where=[right >= left for left, right in zip(paired_whiten, paired_no_whiten)],
        color=light_orange,
        alpha=0.55,
        interpolate=True,
        label="RMS inflation",
        zorder=1,
    )
    ax_ratio.plot(
        whiten_x,
        [point.ratio for point in whiten],
        color=blue,
        marker="o",
        markerfacecolor="white",
        markeredgewidth=1.1,
        label="With whitening",
        zorder=3,
    )
    ax_ratio.plot(
        no_whiten_x,
        [point.ratio for point in no_whiten],
        color=vermillion,
        marker="s",
        label="Without whitening",
        zorder=4,
    )
    ax_ratio.set_yscale(yscale)
    ylabel = "Adversarial / reference RMS"
    if yscale == "log":
        ylabel += " (log scale)"
    ax_ratio.set_xlabel(step_label)
    ax_ratio.set_ylabel(ylabel)
    ax_ratio.set_title("Whitening prevents RMS inflation")
    ax_ratio.legend(loc="best")
    ax_ratio.xaxis.set_major_locator(MaxNLocator(nbins=6))

    summary = (
        f"median: {stats['median_inflation_factor']:.2f}×\n"
        f"max: {stats['max_inflation_factor']:.2f}×"
    )
    ax_ratio.text(
        0.97,
        0.95,
        summary,
        transform=ax_ratio.transAxes,
        ha="right",
        va="top",
        color=vermillion,
        fontsize=8,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 2.5},
    )

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{prefix}.pdf"
    png_path = output_dir / f"{prefix}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return pdf_path, png_path


def main() -> int:
    args = parse_args()
    try:
        whiten = limit_points(load_points(args.whiten), args.max_step, "whiten")
        no_whiten = limit_points(
            load_points(args.no_whiten), args.max_step, "no-whiten"
        )
        warnings = validate_comparability(whiten, no_whiten)
        paired, stats = align_and_summarize(whiten, no_whiten)
        stats["warnings"] = warnings
        stats["whiten_input"] = str(args.whiten)
        stats["no_whiten_input"] = str(args.no_whiten)
        stats["max_step"] = args.max_step

        print(json.dumps(stats, indent=2, sort_keys=True))
        for warning in warnings:
            print(f"WARNING: {warning}", file=sys.stderr)
        if args.stats_only:
            return 0

        csv_path, stats_path = write_tables(
            args.output_dir, args.prefix, paired, stats
        )
        pdf_path, png_path = make_figure(
            whiten,
            no_whiten,
            paired,
            stats,
            args.output_dir,
            args.prefix,
            args.yscale,
        )
        print(f"wrote {pdf_path}")
        print(f"wrote {png_path}")
        print(f"wrote {csv_path}")
        print(f"wrote {stats_path}")
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
