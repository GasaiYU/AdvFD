#!/usr/bin/env bash
# Table 1a: queue-size ablation on pMF-B/16 at 256px.
#
# Required:
#   DATA_ROOT=/path/to/imagenet
#
# Optional:
#   CKPT_ROOT=./checkpoints/base
#   NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 GPUS_PER_NODE=8
#   ENABLE_WANDB=0

set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to the ImageNet root with train/ and val/ subdirectories}"
: "${CKPT_ROOT:=./checkpoints/base}"
: "${NNODES:=1}"
: "${NODE_RANK:=0}"
: "${MASTER_ADDR:=127.0.0.1}"
: "${MASTER_PORT:=29500}"
: "${GPUS_PER_NODE:=8}"
: "${GLOBAL_BSZ:=1024}"
: "${ENABLE_WANDB:=0}"

TOTAL_GPUS=$(( NNODES * GPUS_PER_NODE ))
BATCH_SIZE=$(( GLOBAL_BSZ / TOTAL_GPUS ))
WANDB_FLAG=--disable_wandb
if [ "$ENABLE_WANDB" = "1" ]; then
    WANDB_FLAG=--enable_wandb
fi

for QUEUE_SIZE in 1024 5000 10000 50000 100000 500000; do
    torchrun \
        --nnodes="$NNODES" \
        --node_rank="$NODE_RANK" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        --nproc_per_node="$GPUS_PER_NODE" \
        main_fd.py \
        --project table_1a_queue_size \
        --exp_name "pMF_B-fd-queue${QUEUE_SIZE}" \
        --batch_size "$BATCH_SIZE" \
        --data_path "$DATA_ROOT" \
        --load_from "${CKPT_ROOT}/pMF-B_256.pth" \
        --model pMF_B --rope_2d --learned_pe --disable_v_head \
        --cfg 8.5 --interval_min 0.1 --interval_max 0.7 \
        --num_sampling_steps 1 \
        --eval_bsz 256 --num_images_for_eval_and_search 50000 \
        --vis_freq 50 --online_eval --eval_freq 50 \
        --print_freq 20 --milestone_interval 10 --save_freq 5 \
        --epochs 50 --steps_per_epoch 1250 --warmup_epochs 5 \
        --lr 1e-6 --lr_sched cosine --min_lr 0.0 \
        --fd_repr_models inception --fd_eigvalsh \
        --queue_size "$QUEUE_SIZE" \
        --compile --auto_resume "$WANDB_FLAG"
done
