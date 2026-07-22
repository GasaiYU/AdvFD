"""Unpack a bundled reference-statistics pickle into data/fid_stats/*.npz.

Expected input:
    data/fid_stats/paper_ref_stats.pkl

The pickle should be a dict mapping output .npz filename to a dict of numpy arrays.
"""

from __future__ import annotations

import os
import pickle
import sys

import numpy as np

BUNDLE = "data/fid_stats/paper_ref_stats.pkl"
OUT_DIR = "data/fid_stats"


def main() -> None:
    if not os.path.exists(BUNDLE):
        sys.exit(
            f"Missing {BUNDLE}. Download the released reference-statistics bundle first."
        )

    with open(BUNDLE, "rb") as f:
        bundle = pickle.load(f)

    os.makedirs(OUT_DIR, exist_ok=True)
    written = 0
    skipped = 0
    for name, arrays in bundle.items():
        out_path = os.path.join(OUT_DIR, name)
        if os.path.exists(out_path):
            print(f"skip existing: {name}")
            skipped += 1
            continue
        np.savez(out_path, **arrays)
        print(f"wrote: {name}")
        written += 1

    print(f"done: wrote={written}, skipped={skipped}, out_dir={OUT_DIR}")


if __name__ == "__main__":
    main()
