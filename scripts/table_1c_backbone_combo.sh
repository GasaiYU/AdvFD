#!/usr/bin/env bash
# Table 1c: multi-backbone FD ablation on pMF-B/16 at 256px.

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

run_one() {
    local exp_name="$1"
    shift
    torchrun \
        --nnodes="$NNODES" \
        --node_rank="$NODE_RANK" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        --nproc_per_node="$GPUS_PER_NODE" \
        main_fd.py \
        --project table_1c_backbone_combo \
        --exp_name "$exp_name" \
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
        --fd_eigvalsh --fd_ema_beta 0.999 \
        --compile --auto_resume "$WANDB_FLAG" \
        "$@"
}

run_one pMF_B-fd-siglip-cls+inception \
    --fd_repr_models vit_so400m_patch16_siglip_256.v2_webli inception \
    --fd_repr_pool_types cls cls \
    --fd_target_sizes 224 256

run_one pMF_B-fd-sim \
    --fd_repr_models vit_so400m_patch16_siglip_256.v2_webli vit_large_patch16_224.mae inception \
    --fd_repr_pool_types cls cls cls \
    --fd_target_sizes 224 224 256
