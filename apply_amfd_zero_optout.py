#!/usr/bin/env python3
"""Add a ZeRO-1 opt-out to the AMFD amortizer optimizer.

Needed to benchmark 1-card against 2-card without ZeRO's parameter broadcast
confounding the comparison: as shipped, ``_build_amort_optimizer`` shards
whenever a process group exists, which includes a 1-card ``torchrun``.

After this, ZeRO stays on by default and there are two ways to turn it off:

    AMFD_ZERO=0 ...                 # env, needs no launcher change
    main_amfd.py --no_amort_zero    # explicit flag, recorded in args

Matches on exact source strings rather than line numbers, so it either applies
cleanly or tells you nothing changed.  Idempotent.  Run from the repo root:

    python apply_amfd_zero_optout.py            # apply
    python apply_amfd_zero_optout.py --revert    # undo
"""
import argparse
import sys
from pathlib import Path

TARGET = Path("amfd/integration.py")

ARG_ANCHOR = '    group.add_argument("--amort_grad_clip", type=float, default=1.0)\n'

ARG_ADDITION = '''
    group.add_argument(
        "--no_amort_zero", action="store_false", dest="amort_zero",
        default=os.environ.get("AMFD_ZERO", "1") != "0",
        help="keep the amortizer optimizer state replicated instead of sharding "
             "it with ZeRO-1. Sharding is mathematically equivalent, so this is "
             "for isolating ZeRO's communication cost when benchmarking, not for "
             "changing results. Also settable as AMFD_ZERO=0.",
    )
'''

GATE_ANCHOR = """    zero_cls = _zero_optimizer_class()
    if not torch.distributed.is_initialized() or zero_cls is None:
        logger.info("[AMFD] amortizer optimizer: AdamW (replicated, single process)")
        return torch.optim.AdamW(params, **defaults)
"""

GATE_REPLACEMENT = """    zero_cls = _zero_optimizer_class()
    sharding_disabled = not getattr(args, "amort_zero", True)
    if sharding_disabled or not torch.distributed.is_initialized() or zero_cls is None:
        reason = (
            "disabled by --no_amort_zero/AMFD_ZERO=0" if sharding_disabled
            else "single process" if not torch.distributed.is_initialized()
            else "torch.distributed.optim unavailable"
        )
        logger.info("[AMFD] amortizer optimizer: AdamW (replicated, %s)", reason)
        return torch.optim.AdamW(params, **defaults)
"""

IMPORT_ANCHOR = """def _build_amort_optimizer(params, args):
    \"\"\"Build the amortizer optimizer, sharding its state when distributed.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revert", action="store_true")
    opts = parser.parse_args()

    if not TARGET.is_file():
        sys.exit(f"error: {TARGET} not found -- run this from the repo root")

    src = TARGET.read_text()
    original = src

    if opts.revert:
        src = src.replace(ARG_ANCHOR + ARG_ADDITION, ARG_ANCHOR)
        src = src.replace(GATE_REPLACEMENT, GATE_ANCHOR)
        # Drop the import only if nothing else needs it.  The real module uses
        # os.path.isfile in resolve_amfd_args, so there it stays -- and if it was
        # missing from the top-level imports before, that path was already broken.
        if "os." not in src.replace("import os\n", ""):
            src = src.replace("import logging\nimport os\n", "import logging\n", 1)
    else:
        if ARG_ADDITION in src:
            print("already applied: --no_amort_zero is present")
        else:
            if ARG_ANCHOR not in src:
                sys.exit(f"error: could not find the --amort_grad_clip line in {TARGET}")
            src = src.replace(ARG_ANCHOR, ARG_ANCHOR + ARG_ADDITION, 1)

        if GATE_REPLACEMENT in src:
            print("already applied: the sharding gate is present")
        elif GATE_ANCHOR not in src:
            sys.exit(
                f"error: could not find the sharding gate in {TARGET}.\n"
                "Expected this block inside _build_amort_optimizer:\n\n"
                + GATE_ANCHOR
            )
        else:
            src = src.replace(GATE_ANCHOR, GATE_REPLACEMENT, 1)

    # add_amfd_args now reads os.environ; the module imports os lazily elsewhere,
    # so make it explicit at the top rather than assume.
    if not opts.revert and "\nimport os\n" not in src:
        src = src.replace("import logging\n", "import logging\nimport os\n", 1)

    if src == original:
        print("no change made")
        return

    TARGET.write_text(src)
    print(f"{'reverted' if opts.revert else 'applied'}: {TARGET}")
    print("\nverify with:")
    print("  python -c \"import amfd.integration\"")
    print("  python -m pytest tests/test_amfd.py tests/test_amfd_zero.py -q")


if __name__ == "__main__":
    main()
