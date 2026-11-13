#!/usr/bin/env python3
"""Collect FID(ADM), FDr6 and FDr3 of a checkpoint sweep into one table.

Each --run LABEL=PROJECT_DIR points at a directory laid out by
eval_ckpt_sweep.sh:  PROJECT_DIR/<exp>_step_NNNNNNN/final_eval_summary.csv.
The three metrics come from utils/compute_fdr_subsets.py, so they match the
numbers it prints:

  FID    raw FD in the Inception space (FID(ADM) reference stats)
  FDr6   mean FDr over Inception, ConvNeXt, DINOv2, MAE, SigLIP, CLIP
  FDr3   mean FDr over DINOv2, CLIP, ConvNeXt (encoders unused in training)

Usage:
    python scripts/rebuttal/summarize_ckpt_sweep.py \
        --run advfd=work_dirs/eval_results/rebuttal/rebuttal_sweep_advfd \
        --run fd_loss=work_dirs/eval_results/rebuttal/rebuttal_sweep_fd_loss \
        --out work_dirs/eval_results/rebuttal/rebuttal_sweep_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils.compute_fdr_subsets import compute  # noqa: E402

STEP_RE = re.compile(r"step_(\d+)$")
METRICS = ("fid", "fdr6", "fdr3")
NAMES = {"fid": "FID", "fdr6": "FDr6", "fdr3": "FDr3"}
DIGITS = {"fid": 3, "fdr6": 4, "fdr3": 4}


def parse_run(spec):
    label, sep, path = spec.partition("=")
    if not sep or not label or not path:
        raise argparse.ArgumentTypeError(f"expected LABEL=PROJECT_DIR, got {spec!r}")
    return label, Path(path)


def collect(project_dir):
    """step -> {fid, fdr6, fdr3}; incomplete evaluations are reported and skipped."""
    rows = {}
    for summary in sorted(project_dir.glob("*/final_eval_summary.csv")):
        match = STEP_RE.search(summary.parent.name)
        if match is None:
            continue
        try:
            fid, fdr6, _inception_score, _fdr5, fdr3 = compute(summary)
        except SystemExit as exc:  # compute() exits on missing encoders
            print(f"[skip] {summary}: {exc}", file=sys.stderr)
            continue
        if fid is None or fdr6 is None:
            print(f"[skip] {summary}: no FID(ADM) or fdr6 row", file=sys.stderr)
            continue
        rows[int(match.group(1))] = {"fid": fid, "fdr6": fdr6, "fdr3": fdr3}
    return rows


def cell(value, metric):
    width = 8
    return f"{value:{width}.{DIGITS[metric]}f}" if value is not None else f"{'-':>{width}}"


def print_table(runs):
    labels = list(runs)
    steps = sorted(set().union(*(set(rows) for rows in runs.values())))
    head, sub = f"{'step':>8}", f"{'':>8}"
    for label in labels:
        head += f" | {label:^26}"
        sub += " | " + " ".join(f"{NAMES[m]:>8}" for m in METRICS)
    print(head)
    print(sub)
    print("-" * len(sub))
    for step in steps:
        line = f"{step:8d}"
        for label in labels:
            row = runs[label].get(step)
            line += " | " + " ".join(cell(row[m] if row else None, m) for m in METRICS)
        print(line)
    print()
    for label, rows in runs.items():
        for metric in METRICS:
            if rows:
                step, row = min(rows.items(), key=lambda item: item[1][metric])
                print(f"{label:>10}  best {NAMES[metric]:<4} = {row[metric]:.{DIGITS[metric]}f} @ step {step}")
    missing = {label: sorted(set(steps) - set(rows)) for label, rows in runs.items()}
    for label, steps_missing in missing.items():
        if steps_missing:
            print(f"[note] {label} has no result at steps {steps_missing}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=parse_run, action="append", required=True, metavar="LABEL=PROJECT_DIR")
    parser.add_argument("--out", type=Path, required=True, help="long-format CSV: run,step,fid,fdr6,fdr3")
    args = parser.parse_args()

    labels = [label for label, _ in args.run]
    if len(set(labels)) != len(labels):
        parser.error("--run labels must be unique")
    runs = {label: collect(path) for label, path in args.run}
    if not any(runs.values()):
        raise SystemExit("no finished evaluations found")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "step", *METRICS])
        for label, rows in runs.items():
            for step in sorted(rows):
                writer.writerow([label, step, *(f"{rows[step][m]:.6f}" for m in METRICS)])
    print_table(runs)
    print(f"\n[sweep] saved {args.out}")


if __name__ == "__main__":
    main()
