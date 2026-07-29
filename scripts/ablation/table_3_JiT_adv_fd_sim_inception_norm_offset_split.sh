#!/usr/bin/env bash
# Split real/fake feature-scale regularization ablation of
# scripts/table_3_JiT_adv_fd_sim.sh.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

: "${MODEL_SIZE:=B}"
: "${LOAD_INIT:=base}"
: "${FD_ADV_WEIGHT:=0.1}"
: "${FD_ADV_NORM_OFFSET_WEIGHT:=0.01}"

export MODEL_SIZE LOAD_INIT FD_ADV_WEIGHT

NORM_OFFSET_TAG="${FD_ADV_NORM_OFFSET_WEIGHT//./p}"
EXP_NAME="JiT_${MODEL_SIZE}-fd-sim-advinc-w${FD_ADV_WEIGHT}-from-${LOAD_INIT}-normoff-split${NORM_OFFSET_TAG}"

# Keep the ablation arguments last so auto-resume cannot select an experiment
# with a different critic objective.
exec bash scripts/table_3_JiT_adv_fd_sim.sh \
    "$@" \
    --exp_name "$EXP_NAME" \
    --fd_adv_norm_offset_weight "$FD_ADV_NORM_OFFSET_WEIGHT" \
    --fd_adv_norm_offset_split
