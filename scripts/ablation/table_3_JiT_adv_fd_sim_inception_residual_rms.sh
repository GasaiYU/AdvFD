#!/usr/bin/env bash
# Residual-RMS ablation of scripts/table_3_JiT_adv_fd_sim.sh.
# The parent supplies the complete SIM + Inception FD-Adv recipe; this wrapper
# changes only the adversarial Inception feature constraint.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

: "${MODEL_SIZE:=B}"
: "${LOAD_INIT:=base}"
: "${FD_ADV_WEIGHT:=0.1}"
: "${FD_ADV_RESIDUAL_RMS_KAPPA:=0.2}"
: "${FD_ADV_RESIDUAL_RMS_LOG_FREQ:=100}"

export MODEL_SIZE LOAD_INIT FD_ADV_WEIGHT

KAPPA_TAG="${FD_ADV_RESIDUAL_RMS_KAPPA//./p}"
EXP_NAME="JiT_${MODEL_SIZE}-fd-sim-advinc-w${FD_ADV_WEIGHT}-from-${LOAD_INIT}-resrms${KAPPA_TAG}"

# Keep the ablation arguments last so auto-resume cannot select an experiment
# using a different feature space and callers cannot accidentally override it.
exec bash scripts/table_3_JiT_adv_fd_sim.sh \
    "$@" \
    --exp_name "$EXP_NAME" \
    --fd_adv_residual_rms_kappa "$FD_ADV_RESIDUAL_RMS_KAPPA" \
    --fd_adv_residual_rms_log_freq "$FD_ADV_RESIDUAL_RMS_LOG_FREQ"
