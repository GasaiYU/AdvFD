#!/usr/bin/env bash
# Full 50k-image evaluation for pMeanFlow/pMF checkpoints.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export HF_HOME="${HF_HOME:-/mmu-vcg/gaomingju/data/models/}"
export TORCH_HOME="${TORCH_HOME:-/mmu-vcg/gaomingju/data/models/}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

export PRESET="${PRESET:-pMF_L_256}"
export CKPT_PATH="${CKPT_PATH:-/mmu-vcg/gaomingju/workspace/foundation/FD-Loss-Ours/work_dirs/table_3_pMF/pMF_L_256-fd-sim-advinc-w0.05-advfreq2-detachreal-2e-6/checkpoints/step_0056099.pth}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export NUM_IMAGES="${NUM_IMAGES:-50000}"
export EVAL_BSZ="${EVAL_BSZ:-256}"
export MASTER_PORT="${MASTER_PORT:-29614}"
export RESULT_ROOT="${RESULT_ROOT:-./work_dirs/eval_results}"
export PROJECT="${PROJECT:-eval_pmf}"
export EXP_NAME="${EXP_NAME:-pMF-L_FD-SIM-adv-56k_0_05_2_e-6}"

EXTRA_EVAL_ARGS=()
if [[ -n "${EVAL_MODELS:-}" ]]; then
    read -r -a EVAL_MODEL_ARGS <<< "$EVAL_MODELS"
    EXTRA_EVAL_ARGS+=(--models "${EVAL_MODEL_ARGS[@]}")
fi

bash scripts/evaluate_released_ckpt.sh \
    --enable_vis --vis_steps 1 --num_sampling_steps 1 \
    "${EXTRA_EVAL_ARGS[@]}" \
    "$@"
