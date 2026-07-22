#!/usr/bin/env bash
# Table 3: JiT scalability at 256px.
# Set MODEL_SIZE in {B,L,H}.

export HF_HOME=/mmu-vcg/gaomingju/data/models/
export TORCH_HOME=/mmu-vcg/gaomingju/data/models/
export HF_ENDPOINT=https://hf-mirror.com
export DATA_ROOT=/mmu-vcg/zhangxu34/datasets/ImageNet-1K/

set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to the ImageNet root with train/ and val/ subdirectories}"
: "${CKPT_ROOT:=./checkpoints/base}"
: "${OUTPUT_DIR:=/mmu-vcg/gaomingju/workspace/foundation/FD-Loss-Ours/work_dirs}"
: "${NNODES:=1}"
: "${NODE_RANK:=0}"
: "${MASTER_ADDR:=127.0.0.1}"
: "${MASTER_PORT:=29500}"
: "${GPUS_PER_NODE:=8}"
: "${GLOBAL_BSZ:=1024}"
: "${ENABLE_WANDB:=0}"
: "${MODEL_SIZE:=B}"
: "${PATCH_GAN_INIT:=base}"  # base | fd75k | custom | none
: "${PATCH_GAN_LOAD_FROM:=/mmu-vcg/gaomingju/workspace/foundation/FD-Loss-Ours/work_dirs/table_3_JiT/JiT_B-fd-inception/checkpoints/step_0075000.pth}"

TOTAL_GPUS=$(( NNODES * GPUS_PER_NODE ))
BATCH_SIZE=$(( GLOBAL_BSZ / TOTAL_GPUS ))
WANDB_FLAG=--enable_wandb
# if [ "$ENABLE_WANDB" = "1" ]; then
#     WANDB_FLAG=--enable_wandb
# fi

MAE="vit_large_patch16_224.mae"
SIGLIP="vit_so400m_patch16_siglip_256.v2_webli"

case "${MODEL_SIZE}" in
    B)
        MODEL=JiT_B; CFG=3.0; INTERVAL_MIN=0.1; INTERVAL_MAX=1.0
        case "${PATCH_GAN_INIT}" in
            base) LOAD="${CKPT_ROOT}/JiT-B.pth" ;;
            fd75k) LOAD="$PATCH_GAN_LOAD_FROM" ;;
            custom) LOAD="$PATCH_GAN_LOAD_FROM" ;;
            none) LOAD="" ;;
            *) echo "[ERR] unsupported PATCH_GAN_INIT=${PATCH_GAN_INIT}"; exit 1 ;;
        esac ;;
    L)
        MODEL=JiT_L; CFG=2.4; INTERVAL_MIN=0.1; INTERVAL_MAX=1.0
        LOAD="${CKPT_ROOT}/JiT-L.pth" ;;
    H)
        MODEL=JiT_H; CFG=2.2; INTERVAL_MIN=0.1; INTERVAL_MAX=1.0
        LOAD="${CKPT_ROOT}/JiT-H.pth" ;;
    *) echo "[ERR] unsupported MODEL_SIZE=${MODEL_SIZE}"; exit 1 ;;
esac

run_one() {
    local exp_name="$1"
    shift
    local load_args=()
    if [ -n "$LOAD" ]; then
        load_args=(--load_from "$LOAD")
    fi
    torchrun \
        --nnodes="$NNODES" \
        --node_rank="$NODE_RANK" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        --nproc_per_node="$GPUS_PER_NODE" \
        main_fd.py \
        --project table_3_JiT \
        --exp_name "$exp_name" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size "$BATCH_SIZE" \
        --data_path "$DATA_ROOT" \
        "${load_args[@]}" \
        --model "$MODEL" --rope_2d --learned_pe --legacy_time_convention \
        --cfg "$CFG" --interval_min "$INTERVAL_MIN" --interval_max "$INTERVAL_MAX" \
        --ema_type edm \
        --num_sampling_steps 1 \
        --eval_bsz 256 --num_images_for_eval_and_search 50000 \
        --vis_freq 100 --online_eval --eval_freq 10000 \
        --print_freq 20 --milestone_interval 10 --save_freq 5 \
        --epochs 100 --steps_per_epoch 1250 --warmup_epochs 5 \
        --lr 1e-5 --lr_sched cosine --min_lr 0.0 \
        --grad_checkpointing \
        --fd_eigvalsh --fd_ema_beta 0.999 \
        --auto_resume "$WANDB_FLAG" \
        "$@"
}

run_one "${MODEL}-fd-inception-patchgan-w0.1-start1k-warmup9k-from-${PATCH_GAN_INIT}" \
    --fd_repr_models inception \
    --patch_gan_weight 0.1 \
    --patch_gan_lr 1e-6 \
    --patch_gan_layers 3 \
    --patch_gan_start_step 1000 \
    --patch_gan_warmup_steps 9000 \
    --patch_gan_adaptive_weight
# run_one "${MODEL}-fd-sim-from-pretrained-run_new" \
#     --fd_repr_models "$SIGLIP" "$MAE" inception \
#     --fd_repr_pool_types cls cls cls \
#     --fd_target_sizes 224 224 256
