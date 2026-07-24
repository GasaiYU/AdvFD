#!/usr/bin/env bash
# Overfit one visibly large, strictly nonzero universal noise on the fixed
# 50k pMF-B cache using only the exact joint Inception FID gradient.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

# Complete pMF-B/256 configuration. Environment overrides are optional.
: "${CKPT_ROOT:=./checkpoints/base}"
: "${GPUS_PER_NODE:=8}"
: "${NNODES:=1}"
: "${NODE_RANK:=0}"
: "${MASTER_ADDR:=127.0.0.1}"
: "${MASTER_PORT:=29660}"
: "${GLOBAL_CACHE_BSZ:=1024}"
: "${GRADIENT_BSZ:=256}"
: "${EVAL_BSZ_PER_GPU:=128}"
: "${OPT_IMAGES:=50000}"
: "${CACHE_ROOT:=./work_dirs/hacking_cache}"
: "${OUTPUT_ROOT:=./work_dirs}"
: "${ENABLE_WANDB:=0}"

# alpha=0.2 in pMF [-1,1] model space gives pixel-space RMS 0.1
# (25.5/255) because the learned direction is constrained to RMS=1.
: "${ALPHA:=0.25}"
: "${PGD_STEPS:=500}"
: "${PGD_STEP_SIZE:=0.05}"
: "${PGD_BACKTRACKS:=10}"
: "${MIN_RADIUS:=0.04}"
: "${MAX_RADIUS:=0.50}"
: "${EXP_NAME:=pMF_B_256-large-spatial-noise-alpha0p2-overfit50k}"

if ! python -c 'import sys; sys.exit(0 if float(sys.argv[1]) >= 0.2 else 1)' "$ALPHA"; then
    echo "[ERR] ALPHA=${ALPHA} is below the required minimum 0.2" >&2
    exit 1
fi

echo "[large-noise] fixed model-space alpha=${ALPHA}"
echo "[large-noise] fixed pixel-space RMS=$(python -c 'import sys; print(float(sys.argv[1]) / 2)' "$ALPHA")"
echo "[large-noise] optimization images=${OPT_IMAGES} PGD steps=${PGD_STEPS}"

CKPT_ROOT="$CKPT_ROOT" \
MODEL_SIZE=B \
RES=256 \
GPUS_PER_NODE="$GPUS_PER_NODE" \
NNODES="$NNODES" \
NODE_RANK="$NODE_RANK" \
MASTER_ADDR="$MASTER_ADDR" \
MASTER_PORT="$MASTER_PORT" \
GLOBAL_CACHE_BSZ="$GLOBAL_CACHE_BSZ" \
GRADIENT_BSZ="$GRADIENT_BSZ" \
EVAL_BSZ_PER_GPU="$EVAL_BSZ_PER_GPU" \
OPT_IMAGES="$OPT_IMAGES" \
VAL_IMAGES=5000 \
TEST_IMAGES=50000 \
CACHE_ROOT="$CACHE_ROOT" \
ENABLE_WANDB="$ENABLE_WANDB" \
EXP_NAME="$EXP_NAME" \
PGD_STEPS="$PGD_STEPS" \
bash hacking/run_pmf_fourier_universal.sh \
    --output_dir "$OUTPUT_ROOT" \
    --hack_overfit_only \
    --hack_require_nonzero_selection \
    --hack_pattern_parameterization spatial_bandpass \
    --hack_pattern_size 256 \
    --hack_fourier_min_radius "$MIN_RADIUS" \
    --hack_fourier_max_radius "$MAX_RADIUS" \
    --hack_train_alpha "$ALPHA" \
    --hack_pgd_step_size "$PGD_STEP_SIZE" \
    --hack_overfit_backtracks "$PGD_BACKTRACKS" \
    --hack_overfit_backtrack_factor 0.5 \
    --hack_min_validation_improvement 0.00001 \
    --hack_eval_alphas 0 "$ALPHA"

RUN_ROOT="${OUTPUT_ROOT}/pMF_universal_pattern/${EXP_NAME}"
SUMMARY="${RUN_ROOT}/overfit_summary.json"
CHECKPOINT="${RUN_ROOT}/checkpoints/fourier_pattern_overfit_selected.pth"
PATTERN="${RUN_ROOT}/fourier_pattern.npy"

if [[ ! -f "$SUMMARY" || ! -f "$CHECKPOINT" || ! -f "$PATTERN" ]]; then
    echo "[ERR] expected outputs are incomplete under $RUN_ROOT" >&2
    exit 1
fi

python -c '
import json
import sys

summary = json.load(open(sys.argv[1]))
print("[result] baseline FID: {:.6f}".format(summary["baseline_fid"]))
print("[result] nonzero-noise FID: {:.6f}".format(summary["best_fid"]))
print("[result] delta: {:+.6f}".format(summary["best_delta"]))
print("[result] model-space RMS: {:.6f}".format(
    summary["selected_applied_model_rms"]
))
print("[result] pixel-space RMS: {:.6f}".format(
    summary["selected_applied_pixel_rms"]
))
print("[result] selected nonzero: {}".format(
    summary["selected_pattern_nonzero"]
))
if not summary["fit_success"]:
    raise SystemExit(
        "No alpha>=0.2 nonzero noise beat the clean 50k baseline; "
        "the nonzero checkpoint is retained for diagnosis, but the "
        "experiment is marked unsuccessful."
    )
' "$SUMMARY"

echo "[done] selected checkpoint: $CHECKPOINT"
echo "[done] exact RMS=1 noise:    $PATTERN"
echo "[done] summary:              $SUMMARY"
