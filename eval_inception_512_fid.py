"""Evaluate 512-resolution Inception FID with a single reference stats file.

This is a thin wrapper around eval_all_fds.py.  It keeps the original generation
and folder-evaluation code, but replaces the Inception reference list with:

    data/fid_stats/inception_in512_stats.npz
"""

from __future__ import annotations

import sys

import torch.nn.functional as F

import eval_all_fds as eval_fds


REF_STATS = "data/fid_stats/inception_in512_stats.npz"
FID_EVAL_SIZE = 512


def _ensure_default_flag(argv: list[str], flag: str, value: str | None = None) -> None:
    if flag in argv:
        return
    argv.append(flag)
    if value is not None:
        argv.append(value)


def main() -> None:
    eval_fds.INCEPTION_STATS = [("FID(Inception-512)", REF_STATS)]
    eval_fds.DEFAULT_MODELS = ["inception"]

    original_accumulate_batch = eval_fds.accumulate_batch

    def accumulate_batch_inception_512(images, *args, **kwargs):
        if images.shape[-2:] != (FID_EVAL_SIZE, FID_EVAL_SIZE):
            images = F.interpolate(
                images,
                size=(FID_EVAL_SIZE, FID_EVAL_SIZE),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        return original_accumulate_batch(images, *args, **kwargs)

    eval_fds.accumulate_batch = accumulate_batch_inception_512

    argv = sys.argv
    _ensure_default_flag(argv, "--models", "inception")
    if "--no_prc" not in argv:
        argv.append("--no_prc")

    if "--eval_random_train_set" in argv:
        eval_fds.main_random_train(eval_fds._get_folder_parser().parse_args())
    elif "--image_folder" in argv:
        eval_fds.main_folder(eval_fds._get_folder_parser().parse_args())
    elif "--gen_only" in argv:
        eval_fds.main_gen_only(eval_fds.get_args_parser().parse_args())
    else:
        eval_fds.main_generate(eval_fds.get_args_parser().parse_args())


if __name__ == "__main__":
    main()
