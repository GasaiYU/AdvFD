#!/usr/bin/env bash
# Table 3: pMF scalability at 256px and 512px.
# Set MODEL_SIZE in {B,L,H} and RES in {256,512}.

set -euo pipefail

export HF_HOME=/mmu-vcg/gaomingju/data/models/
export TORCH_HOME=/mmu-vcg/gaomingju/data/models/
export HF_ENDPOINT=https://hf-mirror.com
export DATA_ROOT=/mmu-vcg/zhangxu34/datasets/ImageNet-1K/

: "${DATA_ROOT:?Set DATA_ROOT to the ImageNet root with train/ and val/ subdirectories}"
: "${CKPT_ROOT:=./checkpoints/base}"
: "${NNODES:=1}"
: "${NODE_RANK:=0}"
: "${MASTER_ADDR:=127.0.0.1}"
: "${MASTER_PORT:=29500}"
: "${GPUS_PER_NODE:=8}"
: "${GLOBAL_BSZ:=1024}"
: "${ENABLE_WANDB:=1}"
: "${MODEL_SIZE:=B}"
: "${RES:=256}"

TOTAL_GPUS=$(( NNODES * GPUS_PER_NODE ))
BATCH_SIZE=$(( GLOBAL_BSZ / TOTAL_GPUS ))
WANDB_FLAG=--disable_wandb
if [ "$ENABLE_WANDB" = "1" ]; then
    WANDB_FLAG=--enable_wandb
fi

MAE="vit_large_patch16_224.mae"
SIGLIP="vit_so400m_patch16_siglip_256.v2_webli"

case "${MODEL_SIZE}-${RES}" in
    B-256)
        MODEL=pMF_B; CFG=8.5; INTERVAL_MIN=0.1; INTERVAL_MAX=0.7
        LOAD="${CKPT_ROOT}/pMF-B_256.pth"; EXTRA=() ;;
    L-256)
        MODEL=pMF_L; CFG=7.0; INTERVAL_MIN=0.2; INTERVAL_MAX=0.7
        LOAD="${CKPT_ROOT}/pMF-L_256.pth"; EXTRA=() ;;
    H-256)
        MODEL=pMF_H; CFG=7.0; INTERVAL_MIN=0.2; INTERVAL_MAX=0.6
        LOAD="${CKPT_ROOT}/pMF-H_256.pth"; EXTRA=(--noise_scale 2.0) ;;
    B-512)
        MODEL=pMF_B; CFG=6.5; INTERVAL_MIN=0.1; INTERVAL_MAX=0.7
        LOAD="${CKPT_ROOT}/pMF-B_512.pth"; EXTRA=(--noise_scale 2.0 --img_size 512 --patch_size 32) ;;
    L-512)
        MODEL=pMF_L; CFG=7.5; INTERVAL_MIN=0.2; INTERVAL_MAX=0.6
        LOAD="${CKPT_ROOT}/pMF-L_512.pth"; EXTRA=(--noise_scale 4.0 --img_size 512 --patch_size 32) ;;
    H-512)
        MODEL=pMF_H; CFG=5.5; INTERVAL_MIN=0.1; INTERVAL_MAX=0.6
        LOAD="${CKPT_ROOT}/pMF-H_512.pth"; EXTRA=(--noise_scale 4.0 --img_size 512 --patch_size 32) ;;
    *) echo "[ERR] unsupported MODEL_SIZE=${MODEL_SIZE} RES=${RES}"; exit 1 ;;
esac

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
        --project table_3_pMF \
        --exp_name "$exp_name" \
        --batch_size "$BATCH_SIZE" \
        --data_path "$DATA_ROOT" \
        --load_from "$LOAD" \
        --model "$MODEL" --rope_2d --learned_pe --disable_v_head \
        --cfg "$CFG" --interval_min "$INTERVAL_MIN" --interval_max "$INTERVAL_MAX" \
        --num_sampling_steps 1 \
        --eval_bsz 256 --num_images_for_eval_and_search 50000 \
        --vis_freq 25 --online_eval --eval_freq 1000 \
        --print_freq 20 --milestone_interval 10 --save_freq 5 \
        --epochs 100 --steps_per_epoch 1250 --warmup_epochs 5 \
        --lr 1e-6 --lr_sched cosine --min_lr 0.0 \
        --grad_checkpointing \
        --fd_eigvalsh --fd_ema_beta 0.999 \
        --auto_resume "$WANDB_FLAG" \
        "${EXTRA[@]}" \
        "$@"
}

# run_one "${MODEL}_${RES}-fd-inception" --fd_repr_models inception
run_one "${MODEL}_${RES}-fd-sim" \
    --fd_repr_models "$SIGLIP" "$MAE" inception \
    --fd_repr_pool_types cls cls cls \
    --fd_target_sizes 224 224 256
