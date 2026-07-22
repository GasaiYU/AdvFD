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

export CKPT_PATH="${CKPT_PATH:-/mmu-vcg/gaomingju/workspace/foundation/FD-Loss-Ours/work_dirs/JiT-B-MIND/JiT_B-mind-inception-fq10000-rq50000-s10000-g1.0-p1000/checkpoints/step_0065799.pth}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export NUM_IMAGES="${NUM_IMAGES:-50000}"
export EVAL_BSZ="${EVAL_BSZ:-256}"
export MASTER_PORT="${MASTER_PORT:-29612}"
export RESULT_ROOT="${RESULT_ROOT:-./work_dirs/eval_results}"
export PROJECT="${PROJECT:-eval_mind_loss}"
# export EXP_NAME="${EXP_NAME:-JiT_B-fd-inception-adv-50k-patchgan-no-whiten}"
export EXP_NAME="${EXP_NAME:-JiT_B-fd-inception-MIND-loss-minus-base}"
 
bash scripts/evaluate_released_ckpt.sh --enable_vis --vis_steps 1 --num_sampling_steps 1 "$@" 
