#!/usr/bin/env bash
# Full 50k-image evaluation for the JiT-B timestep-scratch checkpoint.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export HF_HOME="${HF_HOME:-/mmu-vcg/gaomingju/data/models/}"
export TORCH_HOME="${TORCH_HOME:-/mmu-vcg/gaomingju/data/models/}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

export PRESET="${PRESET:-JiT_B}" 

export CKPT_PATH="${CKPT_PATH:-/mmu-vcg/gaomingju/workspace/foundation/FD-Loss-Ours/work_dirs/Jit-B-adv/JiT_B-fd-sim-advinc-randominit-w0.1-from-base/checkpoints/step_0124999.pth}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export NUM_IMAGES="${NUM_IMAGES:-50000}"
export EVAL_BSZ="${EVAL_BSZ:-256}"
export MASTER_PORT="${MASTER_PORT:-29612}"
export RESULT_ROOT="${RESULT_ROOT:-./work_dirs/eval_results}"
export PROJECT="${PROJECT:-eval_jit_b_ablation}"
# export EXP_NAME="${EXP_NAME:-JiT_B-fd-inception-adv-50k-patchgan-no-whiten}"
export EXP_NAME="${EXP_NAME:-JiT_B_SIM-120k-random-init}"

EXTRA_EVAL_ARGS=()
if [[ -n "${EVAL_MODELS:-}" ]]; then
    read -r -a EVAL_MODEL_ARGS <<< "$EVAL_MODELS"
    EXTRA_EVAL_ARGS+=(--models "${EVAL_MODEL_ARGS[@]}")
fi
 
bash scripts/evaluate_released_ckpt.sh \
    --enable_vis --vis_steps 1 --num_sampling_steps 1 \
    "${EXTRA_EVAL_ARGS[@]}" \
    "$@"
