#!/usr/bin/env python3
"""pMF-L entry point for the aligned three-way qualitative generator."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__:
    from .generate_quali_jit_l import main
else:
    from generate_quali_jit_l import main


REPO_ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    # Defaults are prepended so later user arguments can override them.
    sys.argv[1:1] = [
        "--generator_family",
        "pmf_l",
        "--output_root",
        str(REPO_ROOT / "paper" / "quali" / "pmf_l"),
        "--original_checkpoint",
        str(REPO_ROOT / "checkpoints" / "base" / "pMF-L_256.pth"),
        "--fd_checkpoint",
        str(
            REPO_ROOT
            / "checkpoints"
            / "post-trained"
            / "pMF-L_FD-Inception.pth"
        ),
        "--adv_checkpoint",
        str(
            REPO_ROOT
            / "work_dirs"
            / "table_3_pMF"
            / "pMF_L_256-fd-sim-advinc-w0.05-advfreq2-detachreal-2e-6"
            / "checkpoints"
            / "step_0124999.pth"
        ),
    ]
    main()
