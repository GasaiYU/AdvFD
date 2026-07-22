#!/usr/bin/env bash
# Bare JiT ImageNet training with a lightweight FD regularizer.
# Set MODEL_SIZE in {B,L,H}.

export HF_HOME=/mmu-vcg/gaomingju/data/models/
export TORCH_HOME=/mmu-vcg/gaomingju/data/models/
export HF_ENDPOINT=https://hf-mirror.com
export DATA_ROOT=/mmu-vcg/zhangxu34/datasets/ImageNet-1K/

set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to the ImageNet root with train/ and val/ subdirectories}"
: "${OUTPUT_DIR:=/mmu-vcg/gaomingju/workspace/foundation/FD-Loss-Ours/work_dirs}"
: "${NNODES:=1}"
: "${NODE_RANK:=0}"
: "${MASTER_ADDR:=127.0.0.1}"
: "${MASTER_PORT:=29500}"
: "${GPUS_PER_NODE:=8}"
: "${GLOBAL_BSZ:=1024}"
: "${ENABLE_WANDB:=0}"
: "${MODEL_SIZE:=B}"
: "${FD_REG_WEIGHT:=0.01}"

TOTAL_GPUS=$(( NNODES * GPUS_PER_NODE ))
BATCH_SIZE=$(( GLOBAL_BSZ / TOTAL_GPUS ))
WANDB_FLAG=--enable_wandb
if [ "$ENABLE_WANDB" = "1" ]; then
    WANDB_FLAG=--enable_wandb
fi

case "${MODEL_SIZE}" in
    B)
        MODEL=JiT_B; CFG=3.0; PROJ_DROPOUT=0.0 ;;
    L)
        MODEL=JiT_L; CFG=2.4; PROJ_DROPOUT=0.0 ;;
    H)
        MODEL=JiT_H; CFG=2.2; PROJ_DROPOUT=0.2 ;;
    *) echo "[ERR] unsupported MODEL_SIZE=${MODEL_SIZE}"; exit 1 ;;
esac

torchrun \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --nproc_per_node="$GPUS_PER_NODE" \
    main_jit.py \
    --project table_3_JiT \
    --exp_name "${MODEL}-bare-jit-fdreg-w${FD_REG_WEIGHT}" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --data_path "$DATA_ROOT" \
    --model "$MODEL" --rope_2d --learned_pe --legacy_time_convention \
    --proj_dropout "$PROJ_DROPOUT" \
    --P_mean 0.8 --P_std 0.8 \
    --cfg "$CFG" --interval_min 0.1 --interval_max 1.0 \
    --ema_type edm \
    --num_sampling_steps 1 \
    --eval_bsz 256 --num_images_for_eval_and_search 50000 \
    --vis_freq 100 --online_eval --eval_freq 10000 \
    --print_freq 20 --milestone_interval 10 --save_freq 5 \
    --epochs 600 --steps_per_epoch 1250 --warmup_epochs 5 \
    --lr 5e-5 --lr_sched cosine --min_lr 0.0 \
    --grad_checkpointing \
    --fd_reg_weight "$FD_REG_WEIGHT" \
    --fd_reg_repr_model inception \
    --fd_reg_stats_path data/fid_stats/guided_diffusion_stats.npz \
    --fd_eigvalsh \
    --auto_resume "$WANDB_FLAG" \
    "$@"
