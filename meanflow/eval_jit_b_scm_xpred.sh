#!/usr/bin/env bash
# Evaluate permanent-reversed-time direct x-prediction JiT-B sCM checkpoints
# produced by meanflow/train_scm_jit_b.py.  The dedicated model key is
# important: the raw output is D=x_0 and JiT receives 1-2*t_TrigFlow/pi.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export HF_HOME="${HF_HOME:-/mmu-vcg/gaomingju/data/models/}"
export TORCH_HOME="${TORCH_HOME:-/mmu-vcg/gaomingju/data/models/}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

: "${CKPT:=/mmu-vcg/gaomingju/workspace/foundation/FD-Loss-Ours/work_dirs/meanflow/jit_b_scm_xpred_hybrid_from_adapt9999/checkpoints/latest.pth}"
: "${GPUS_PER_NODE:=8}"
: "${NNODES:=1}"
: "${NODE_RANK:=0}"
: "${MASTER_ADDR:=127.0.0.1}"
: "${MASTER_PORT:=29613}"

# VIS_ONLY=1 generates the visualization grid and exits.  Set VIS_ONLY=0 to
# generate NUM_IMAGES samples and compute the requested FD metrics.
: "${VIS_ONLY:=1}"
: "${NUM_IMAGES:=50000}"
: "${EVAL_BSZ:=64}"
: "${CFG:=1.0}"
: "${NUM_STEPS:=1}"
: "${SCM_INTERMEDIATE_T:=1.1}"
: "${DTYPE:=bf16}"
: "${EMA_LABEL:=power_0.05}"
: "${RESULT_ROOT:=./work_dirs/eval_results}"
: "${PROJECT:=eval_jit_b_scm_xpred}"
: "${EVAL_MODELS:=inception}"

if [[ ! -e "${CKPT}" ]]; then
  echo "Checkpoint not found: ${CKPT}" >&2
  exit 2
fi
if [[ "${NUM_STEPS}" != "1" && "${NUM_STEPS}" != "2" ]]; then
  echo "NUM_STEPS must be 1 or 2 for direct x-pred sCM sampling" >&2
  exit 2
fi
if [[ "${VIS_ONLY}" != "0" && "${VIS_ONLY}" != "1" ]]; then
  echo "VIS_ONLY must be 0 or 1" >&2
  exit 2
fi

CKPT_REAL="$(readlink -f "${CKPT}")"
: "${EXP_NAME:=JiT_B_sCM_xpred_reversed_time-$(basename "${CKPT_REAL}" .pth)-cfg${CFG}-step${NUM_STEPS}}"

read -r -a EVAL_MODEL_ARGS <<< "${EVAL_MODELS}"
MODE_ARGS=()
if [[ "${VIS_ONLY}" == "1" ]]; then
  MODE_ARGS+=(--vis_only)
fi

torchrun \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  --nproc_per_node="${GPUS_PER_NODE}" \
  eval_all_fds.py \
  --model JiT_B_sCM_xpred_reversed_time \
  --img_size 256 \
  --num_classes 1000 \
  --rope_2d \
  --learned_pe \
  --P_mean -1.0 \
  --P_std 1.6 \
  --sigma_data 0.5 \
  --sigma_max 80.0 \
  --scm_intermediate_t "${SCM_INTERMEDIATE_T}" \
  --label_drop_prob 0.1 \
  --attn_dropout 0.45 \
  --proj_dropout 0.45 \
  --ema_type power \
  --ema_sigma_rel 0.05 \
  --eval_ema_labels "${EMA_LABEL}" \
  --cfg "${CFG}" \
  --cfg_list "${CFG}" \
  --num_sampling_steps "${NUM_STEPS}" \
  --vis_steps "${NUM_STEPS}" \
  --enable_vis \
  --models "${EVAL_MODEL_ARGS[@]}" \
  --num_images "${NUM_IMAGES}" \
  --eval_bsz "${EVAL_BSZ}" \
  --resume_from "${CKPT_REAL}" \
  --dtype "${DTYPE}" \
  --disable_wandb \
  --no_prc \
  --output_dir "${RESULT_ROOT}" \
  --project "${PROJECT}" \
  --exp_name "${EXP_NAME}" \
  "${MODE_ARGS[@]}" \
  "$@"
