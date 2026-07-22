#!/usr/bin/env python3
"""Print FID/IS and compute FDr subset averages from final_eval_summary.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FDR5_MODELS = ("mae_cls", "dinov2_cls", "clip_cls", "siglip_cls", "convnext")
FDR3_MODELS = ("dinov2_cls", "clip_cls", "convnext")


def _summary_path(path: Path) -> Path:
    if path.is_dir():
        return path / "final_eval_summary.csv"
    return path


def _read_metrics(
    path: Path,
) -> tuple[dict[str, float], float | None, float | None, float | None]:
    with path.open(newline="") as f:
        rows = csv.DictReader(f)
        fdrs = {}
        fid = None
        fdr6 = None
        inception_score = None
        for row in rows:
            model = row.get("model", "")
            if model == "FID(ADM)" and row.get("fd", ""):
                fid = float(row["fd"])
            if fdr6 is None and row.get("fdr6", ""):
                fdr6 = float(row["fdr6"])
            if inception_score is None and row.get("is", ""):
                inception_score = float(row["is"])
            fdr = row.get("fdr", "")
            if model and fdr:
                fdrs[model] = float(fdr)
    return fdrs, fid, fdr6, inception_score


def _mean_required(fdrs: dict[str, float], models: tuple[str, ...], name: str) -> float:
    missing = [model for model in models if model not in fdrs]
    if missing:
        raise SystemExit(f"Missing {name} models: {', '.join(missing)}")
    return sum(fdrs[model] for model in models) / len(models)


def compute(
    path: Path,
) -> tuple[float | None, float | None, float | None, float, float]:
    summary = _summary_path(path)
    if not summary.is_file():
        raise SystemExit(f"Summary not found: {summary}")
    fdrs, fid, fdr6, inception_score = _read_metrics(summary)
    fdr5 = _mean_required(fdrs, FDR5_MODELS, "FDr5")
    fdr3 = _mean_required(fdrs, FDR3_MODELS, "FDr3")
    return fid, fdr6, inception_score, fdr5, fdr3


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Print FID/IS and compute FDr5(no Inception) and "
            "FDr3(no Inception/SigLIP/MAE)."
        )
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="final_eval_summary.csv path or an eval directory containing it",
    )
    args = parser.parse_args()

    for path in args.paths:
        summary = _summary_path(path)
        fid, fdr6, inception_score, fdr5, fdr3 = compute(path)
        print(summary)
        if fid is not None:
            print(f"FID(ADM): {fid:.6f}")
        if inception_score is not None:
            print(f"IS: {inception_score:.6f}")
        if fdr6 is not None:
            print(f"FDr6: {fdr6:.6f}")
        print(f"FDr5(no Inception): {fdr5:.6f}")
        print(f"FDr3(no Inception/SigLIP/MAE): {fdr3:.6f}")


if __name__ == "__main__":
    main()
