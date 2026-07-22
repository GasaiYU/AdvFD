#!/usr/bin/env bash
set -euo pipefail

# Direct x-prediction JiT-B sCM high-noise velocity control.  By default this
# resumes the shared step-9999 adaptation checkpoint, performs the existing
# EMA-to-student phase transition at step 10000, and trains the high-noise
# hybrid sCT/sCD branch.  Set TANGENT_VELOCITY_MODE=sct or scd for controls.

export HF_HOME="${HF_HOME:-/mmu-vcg/gaomingju/data/models/}"
export TORCH_HOME="${TORCH_HOME:-/mmu-vcg/gaomingju/data/models/}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
: "${DATA_ROOT:=/mmu-vcg/zhangxu34/datasets/ImageNet-1K/}"
export DATA_ROOT

: "${DATA_ROOT:?set DATA_ROOT to ImageNet root or its train directory}"
: "${GPUS_PER_NODE:=8}"
: "${NNODES:=1}"
: "${NODE_RANK:=0}"
: "${MASTER_ADDR:=127.0.0.1}"
: "${MASTER_PORT:=29582}"
: "${GLOBAL_BATCH_SIZE:=2048}"

TOTAL_GPUS=$((GPUS_PER_NODE * NNODES))
: "${BATCH_SIZE:=128}"
: "${NUM_WORKERS:=12}"
: "${PREFETCH_FACTOR:=2}"

MICRO_GLOBAL_BATCH=$((BATCH_SIZE * TOTAL_GPUS))
if (( MICRO_GLOBAL_BATCH <= 0 )); then
  echo "BATCH_SIZE, GPUS_PER_NODE, and NNODES must be positive" >&2
  exit 2
fi
if (( GLOBAL_BATCH_SIZE % MICRO_GLOBAL_BATCH != 0 )); then
  echo "GLOBAL_BATCH_SIZE must be divisible by BATCH_SIZE * GPUS_PER_NODE * NNODES" >&2
  exit 2
fi
ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / MICRO_GLOBAL_BATCH))

# Resume at adaptation step 9999 and include global steps 10000..30000.  This
# preserves the original 10k tangent warmup and captures 10k full-tangent steps.
: "${EPOCHS:=320}"
: "${STEPS_PER_EPOCH:=1250}"
: "${LR:=1e-5}"
: "${X_ADAPT_STEPS:=10000}"
: "${X_ADAPT_LR:=1e-4}"
: "${X_ADAPT_WARMUP_STEPS:=1250}"
: "${X_LOSS_SIN_MIN:=1e-3}"
: "${BOUNDARY_LOSS_WEIGHT:=1.0}"
: "${BOUNDARY_BAND_MAX:=0.02}"
: "${DETERMINISTIC_BOUNDARY:=1}"
: "${ADAPTIVE_WEIGHT_MAX:=20.0}"
: "${COLLAPSE_MONITOR_SAMPLES:=8}"
# Keep the guard metrics, but do not abort this finite diagnostic run: the
# checkpoint near the failure itself is one of the requested outputs.
: "${PATCH_COLLAPSE_ABORT_CORR:=0}"
: "${PATCH_COLLAPSE_ABORT_PATIENCE:=5}"
: "${PATCH_COLLAPSE_ABORT_TEMPLATE_FRAC:=0.95}"
: "${PATCH_COLLAPSE_ABORT_SAMPLE_CORR:=0.95}"
: "${INIT_PARAMETERIZATION:=x_pred}"
: "${TANGENT_WARMUP_STEPS:=10000}"
: "${TANGENT_VELOCITY_MODE:=hybrid}"
: "${HYBRID_TEACHER_START:=1.50}"
: "${HYBRID_TEACHER_END:=1.53}"
: "${TEACHER_VELOCITY_SIN_MIN:=1e-3}"
: "${SCM_STEPS:=20001}"

# Coarse saves reproduce the requested warmup checkpoints (14999, 19999).
# Exact-step dense saves then retain 20000, 20100, ..., 21000.  Each checkpoint
# already contains both online `model` and `model_ema` weights.
: "${SAVE_EVERY:=5000}"
: "${DENSE_SAVE_START:=20000}"
: "${DENSE_SAVE_END:=21000}"
: "${DENSE_SAVE_EVERY:=100}"
: "${KEEP_LAST:=20}"

: "${CKPT:=/mmu-vcg/gaomingju/workspace/foundation/FD-Loss-Ours/work_dirs/meanflow/jit_b_scm_xpred_reversed_time_boundary_band/checkpoints/step_0009999.pth}"
: "${CKPT_KEY:=model_ema}"
: "${RESUME_FROM:=$CKPT}"
: "${OUTPUT_DIR:=./work_dirs}"
: "${EXP_NAME:=jit_b_scm_xpred_${TANGENT_VELOCITY_MODE}_from_adapt9999}"

case "$TANGENT_VELOCITY_MODE" in
  sct|hybrid|scd) ;;
  *)
    echo "TANGENT_VELOCITY_MODE must be sct, hybrid, or scd" >&2
    exit 2
    ;;
esac
if [[ ! -f "$CKPT" ]]; then
  echo "initialization checkpoint does not exist: $CKPT" >&2
  exit 2
fi
if [[ -n "$RESUME_FROM" && ! -f "$RESUME_FROM" ]]; then
  echo "resume checkpoint does not exist: $RESUME_FROM" >&2
  exit 2
fi

RESUME_ARGS=()
if [[ -n "$RESUME_FROM" ]]; then
  RESUME_ARGS=(--resume_from "$RESUME_FROM")
fi

if [[ "$DETERMINISTIC_BOUNDARY" == "1" ]]; then
  BOUNDARY_DROPOUT_ARGS=(--deterministic_boundary)
elif [[ "$DETERMINISTIC_BOUNDARY" == "0" ]]; then
  BOUNDARY_DROPOUT_ARGS=(--stochastic_boundary)
else
  echo "DETERMINISTIC_BOUNDARY must be 0 or 1" >&2
  exit 2
fi
# W&B is enabled for this launcher.  Set WANDB_MODE=offline when the compute
# node has no outbound connection; the run can be synchronized later.
: "${WANDB_PROJECT:=meanflow}"
: "${WANDB_ENTITY:=}"
: "${WANDB_NAME:=$EXP_NAME}"
: "${WANDB_MODE:=online}"
export WANDB_MODE

WANDB_ARGS=(
  --enable_wandb
  --wandb_project "$WANDB_PROJECT"
  --wandb_name "$WANDB_NAME"
)
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
  "${RESUME_ARGS[@]}" \
  --output_dir "$OUTPUT_DIR" \
  --exp_name "$EXP_NAME" \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --prefetch_factor "$PREFETCH_FACTOR" \
  --grad_accum_steps "$ACCUM_STEPS" \
  --epochs "$EPOCHS" \
  --steps_per_epoch "$STEPS_PER_EPOCH" \
  --scm_steps "$SCM_STEPS" \
  --save_every "$SAVE_EVERY" \
  --dense_save_start "$DENSE_SAVE_START" \
  --dense_save_end "$DENSE_SAVE_END" \
  --dense_save_every "$DENSE_SAVE_EVERY" \
  --keep_last "$KEEP_LAST" \
  --lr "$LR" \
  --init_parameterization "$INIT_PARAMETERIZATION" \
  --x_adapt_steps "$X_ADAPT_STEPS" \
  --x_adapt_lr "$X_ADAPT_LR" \
  --x_adapt_warmup_steps "$X_ADAPT_WARMUP_STEPS" \
  --x_loss_sin_min "$X_LOSS_SIN_MIN" \
  --tangent_velocity_mode "$TANGENT_VELOCITY_MODE" \
  --hybrid_teacher_start "$HYBRID_TEACHER_START" \
  --hybrid_teacher_end "$HYBRID_TEACHER_END" \
  --teacher_velocity_sin_min "$TEACHER_VELOCITY_SIN_MIN" \
  --boundary_loss_weight "$BOUNDARY_LOSS_WEIGHT" \
  --boundary_band_max "$BOUNDARY_BAND_MAX" \
  "${BOUNDARY_DROPOUT_ARGS[@]}" \
  --collapse_monitor_samples "$COLLAPSE_MONITOR_SAMPLES" \
  --patch_collapse_abort_corr "$PATCH_COLLAPSE_ABORT_CORR" \
  --patch_collapse_abort_patience "$PATCH_COLLAPSE_ABORT_PATIENCE" \
  --patch_collapse_abort_template_frac "$PATCH_COLLAPSE_ABORT_TEMPLATE_FRAC" \
  --patch_collapse_abort_sample_corr "$PATCH_COLLAPSE_ABORT_SAMPLE_CORR" \
  --objective scm \
  --P_mean -1.0 \
  --P_std 1.6 \
  --sigma_data 0.5 \
  --sigma_max 80.0 \
  --tangent_norm_c 0.1 \
  --tangent_warmup_steps "$TANGENT_WARMUP_STEPS" \
  --adaptive_weight_max "$ADAPTIVE_WEIGHT_MAX" \
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
