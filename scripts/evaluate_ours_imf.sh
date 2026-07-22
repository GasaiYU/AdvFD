#!/usr/bin/env bash
# Full 50k-image evaluation for iMeanFlow/iMF checkpoints.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export HF_HOME="${HF_HOME:-/mmu-vcg/gaomingju/data/models/}"
export TORCH_HOME="${TORCH_HOME:-/mmu-vcg/gaomingju/data/models/}"
# export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

export PRESET="${PRESET:-iMF_B}"
export CKPT_PATH="${CKPT_PATH:-checkpoints/post-trained/iMF-B_FD-SIM.pth}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export NUM_IMAGES="${NUM_IMAGES:-50000}"
export EVAL_BSZ="${EVAL_BSZ:-256}"
export MASTER_PORT="${MASTER_PORT:-29613}"
export RESULT_ROOT="${RESULT_ROOT:-./work_dirs/eval_results}"
export PROJECT="${PROJECT:-eval_imf}"
export EXP_NAME="${EXP_NAME:-iMF-B_FD-SIM-120k}"

EXTRA_EVAL_ARGS=()
if [[ -n "${EVAL_MODELS:-}" ]]; then
    read -r -a EVAL_MODEL_ARGS <<< "$EVAL_MODELS"
    EXTRA_EVAL_ARGS+=(--models "${EVAL_MODEL_ARGS[@]}")
fi

bash scripts/evaluate_released_ckpt.sh \
    --enable_vis --vis_steps 1 --num_sampling_steps 1 \
    "${EXTRA_EVAL_ARGS[@]}" \
    "$@"
