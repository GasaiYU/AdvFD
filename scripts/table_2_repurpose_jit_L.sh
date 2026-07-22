#!/usr/bin/env bash
# Table 2: JiT-L multi-step to one-step repurposing.

export HF_HOME=/mmu-vcg/gaomingju/data/models/
export TORCH_HOME=/mmu-vcg/gaomingju/data/models/
export HF_ENDPOINT=https://hf-mirror.com
export DATA_ROOT=/mmu-vcg/zhangxu34/datasets/ImageNet-1K/

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
WANDB_FLAG=--enable_wandb

MAE="vit_large_patch16_224.mae"
SIGLIP="vit_so400m_patch16_siglip_256.v2_webli"

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
        --project table_2_repurpose_jit_L \
        --exp_name "$exp_name" \
        --batch_size "$BATCH_SIZE" \
        --data_path "$DATA_ROOT" \
        --load_from "${CKPT_ROOT}/JiT-L.pth" \
        --model JiT_L --rope_2d --learned_pe --legacy_time_convention \
        --cfg 2.4 --interval_min 0.1 --interval_max 1.0 \
        --ema_type edm \
        --num_sampling_steps 1 \
        --eval_bsz 256 --num_images_for_eval_and_search 50000 \
        --vis_freq 50 --online_eval --eval_freq 99 \
        --print_freq 20 --milestone_interval 10 --save_freq 5 \
        --epochs 50 --steps_per_epoch 1250 --warmup_epochs 5 \
        --lr 1e-5 --lr_sched cosine --min_lr 0.0 \
        --fd_eigvalsh --fd_ema_beta 0.999 \
        --compile --auto_resume "$WANDB_FLAG" \
        "$@"
}

run_one JiT_L-fd-inception-from-pretrained --fd_repr_models inception
# run_one JiT_L-fd-mae-cls --fd_repr_models "$MAE" --fd_repr_pool_types cls
# run_one JiT_L-fd-siglip-cls --fd_repr_models "$SIGLIP" --fd_repr_pool_types cls --fd_target_sizes 224
# run_one JiT_L-fd-siglip-cls+mae-from-pretrained \
#     --fd_repr_models "$SIGLIP" "$MAE" \
#     --fd_repr_pool_types cls cls \
#     --fd_target_sizes 224 224
# run_one JiT_L-fd-sim \
#     --fd_repr_models "$SIGLIP" "$MAE" inception \
#     --fd_repr_pool_types cls cls cls \
#     --fd_target_sizes 224 224 256
