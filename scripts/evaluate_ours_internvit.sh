#!/usr/bin/env bash
# Evaluate our JiT checkpoint with InternViT FD metrics.
#
# Optional:
#   COMPUTE_VALFD=1  also compute ImageNet-val raw InternViT FD normalizer.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export DATA_ROOT="${DATA_ROOT:-/mmu-vcg/zhangxu34/datasets/ImageNet-1K/}"
export HF_HOME="${HF_HOME:-/mmu-vcg/gaomingju/data/models/}"
export TORCH_HOME="${TORCH_HOME:-/mmu-vcg/gaomingju/data/models/}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-/mmu-vcg/gaomingju/data/models/modules}"

export INTERNVIT="${INTERNVIT:-/mmu-vcg/gaomingju/data/models/OpenGVLab/InternViT-300M-448px}"
export INTERNVIT_TARGET_SIZE="${INTERNVIT_TARGET_SIZE:-224}"
export INTERNVIT_STATS_PATH="${INTERNVIT_STATS_PATH:-data/fid_stats/internvit_300m_in256_t${INTERNVIT_TARGET_SIZE}_stats.npz}"

export PRESET="${PRESET:-JiT_B}"
export CKPT_PATH="${CKPT_PATH:-/mmu-vcg/gaomingju/workspace/foundation/FD-Loss-Ours/work_dirs/internvl_fd_JiT-B/JiT_B-fd-internvit-300m/checkpoints/latest.pth}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
export NUM_IMAGES="${NUM_IMAGES:-50000}"
export EVAL_BSZ="${EVAL_BSZ:-128}"
export MASTER_PORT="${MASTER_PORT:-29618}"
export RESULT_ROOT="${RESULT_ROOT:-./work_dirs/eval_results}"
export PROJECT="${PROJECT:-eval_internvit_fd}"
export EXP_NAME="${EXP_NAME:-JiT_B-fd-internvit-latest}"

if [[ ! -d "$INTERNVIT" ]]; then
    echo "[ERR] Missing InternViT model dir: $INTERNVIT" >&2
    exit 1
fi

if [[ ! -f "$INTERNVIT_STATS_PATH" ]]; then
    echo "[ERR] Missing InternViT reference stats: $INTERNVIT_STATS_PATH" >&2
    echo "      Run scripts/internvl/compute_internvit_ref_stats.sh first, or set INTERNVIT_STATS_PATH." >&2
    exit 1
fi

if [[ "${COMPUTE_VALFD:-0}" == "1" ]]; then
    torchrun --nproc_per_node="$GPUS_PER_NODE" --master_port="$MASTER_PORT" \
        scripts/compute_valfd.py \
        --data_root "$DATA_ROOT" \
        --batch_size "$EVAL_BSZ" \
        --models InternViT \
        --output_json "data/fid_stats/valfd_internvit.json" \
        --output_csv "data/fid_stats/valfd_internvit.csv"
    if [[ "${VALFD_ONLY:-0}" == "1" ]]; then
        exit 0
    fi
fi

bash scripts/evaluate_released_ckpt.sh \
    --enable_vis --vis_steps 1 --num_sampling_steps 1 \
    --models "$INTERNVIT" \
    "$@"
