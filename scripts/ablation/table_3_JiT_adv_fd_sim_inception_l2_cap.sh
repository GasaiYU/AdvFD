#!/usr/bin/env bash
# Pure norm-cap ablation of scripts/table_3_JiT_adv_fd_sim.sh.
# The parent script supplies the complete SIM + Inception FD-Adv recipe;
# this wrapper changes only the adversarial Inception pool3 norm bound.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

: "${MODEL_SIZE:=B}"
: "${LOAD_INIT:=base}"
: "${FD_ADV_WEIGHT:=0.1}"
: "${FD_ADV_FEATURE_NORM_CAP:=40}"

# Export values also consumed by the parent when this wrapper supplied their
# defaults rather than inheriting them from the caller's environment.
export MODEL_SIZE LOAD_INIT FD_ADV_WEIGHT

CAP_TAG="${FD_ADV_FEATURE_NORM_CAP//./p}"
EXP_NAME="JiT_${MODEL_SIZE}-fd-sim-advinc-w${FD_ADV_WEIGHT}-from-${LOAD_INIT}-l2cap${CAP_TAG}"

# scripts/table_3_JiT_adv_fd_sim.sh appends its own "$@" after every baseline
# option. Keep the forced arguments last so auto-resume cannot select the
# uncapped experiment and callers cannot accidentally disable this ablation.
exec bash scripts/table_3_JiT_adv_fd_sim.sh \
    "$@" \
    --exp_name "$EXP_NAME" \
    --fd_adv_feature_norm_cap "$FD_ADV_FEATURE_NORM_CAP"
