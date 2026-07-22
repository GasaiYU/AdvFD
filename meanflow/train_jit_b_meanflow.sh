#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/mmu-vcg/gaomingju/data/models/
export TORCH_HOME=/mmu-vcg/gaomingju/data/models/
export HF_ENDPOINT=https://hf-mirror.com
: "${DATA_ROOT:=/mmu-vcg/zhangxu34/datasets/ImageNet-1K/}"
export DATA_ROOT

: "${DATA_ROOT:?set DATA_ROOT to ImageNet root or its train directory}"
: "${GPUS_PER_NODE:=8}"
: "${NNODES:=1}"
: "${NODE_RANK:=0}"
: "${MASTER_ADDR:=127.0.0.1}"
: "${MASTER_PORT:=29581}"
: "${GLOBAL_BATCH_SIZE:=2048}"
TOTAL_GPUS=$((GPUS_PER_NODE * NNODES))
: "${BATCH_SIZE:=128}"
: "${NUM_WORKERS:=12}"
: "${PREFETCH_FACTOR:=2}"
MICRO_GLOBAL_BATCH=$((BATCH_SIZE * TOTAL_GPUS))
if (( GLOBAL_BATCH_SIZE % MICRO_GLOBAL_BATCH != 0 )); then
  echo "GLOBAL_BATCH_SIZE must be divisible by BATCH_SIZE * GPUS_PER_NODE * NNODES" >&2
  exit 2
fi
ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / MICRO_GLOBAL_BATCH))
: "${EPOCHS:=320}"
: "${STEPS_PER_EPOCH:=1250}"
: "${LR:=1e-5}"
: "${F_ADAPT_STEPS:=10000}"
: "${F_ADAPT_LR:=1e-4}"
: "${F_ADAPT_WARMUP_STEPS:=1250}"
: "${INIT_PARAMETERIZATION:=x_pred}"
: "${CKPT:=checkpoints/baseline/jit-b-16/checkpoint-last.pth}"
: "${CKPT_KEY:=model_ema2}"
: "${OUTPUT_DIR:=./work_dirs}"
: "${EXP_NAME:=jit_b_scm_xpred_bridge}"
: "${WANDB_PROJECT:=meanflow}"
: "${WANDB_ENTITY:=}"
: "${WANDB_NAME:=$EXP_NAME}"

WANDB_ARGS=(--enable_wandb --wandb_project "$WANDB_PROJECT" --wandb_name "$WANDB_NAME")
if [[ -n "$WANDB_ENTITY" ]]; then
  WANDB_ARGS+=(--wandb_entity "$WANDB_ENTITY")
fi

torchrun \
  --nnodes="$NNODES" \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  --nproc_per_node="$GPUS_PER_NODE" \
  meanflow/train_scm_jit_b.py \
  --data_path "$DATA_ROOT" \
  --load_from "$CKPT" \
  --checkpoint_key "$CKPT_KEY" \
  --output_dir "$OUTPUT_DIR" \
  --exp_name "$EXP_NAME" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --prefetch_factor "$PREFETCH_FACTOR" \
  --grad_accum_steps "$ACCUM_STEPS" \
  --epochs "$EPOCHS" \
  --steps_per_epoch "$STEPS_PER_EPOCH" \
  --lr "$LR" \
  --init_parameterization "$INIT_PARAMETERIZATION" \
  --f_adapt_steps "$F_ADAPT_STEPS" \
  --f_adapt_lr "$F_ADAPT_LR" \
  --f_adapt_warmup_steps "$F_ADAPT_WARMUP_STEPS" \
  --objective scm \
  --P_mean -1.0 \
  --P_std 1.6 \
  --sigma_data 0.5 \
  --sigma_max 80.0 \
  --tangent_norm_c 0.1 \
  --tangent_warmup_steps 10000 \
  --jvp_dtype amp \
  --beta1 0.9 \
  --beta2 0.99 \
  --adam_eps 1e-11 \
  --lr_sched edm2 \
  --lr_ref_steps 35000 \
  --warmup_epochs 1 \
  --weight_decay 0.0 \
  --grad_clip 1.0 \
  --label_drop_prob 0.1 \
  --attn_dropout 0.45 \
  --proj_dropout 0.45 \
  --dtype bf16 \
  --ema_type power \
  --ema_sigma_rel 0.05 \
  --disable_hflip \
  "${WANDB_ARGS[@]}" \
  "$@"
