#!/usr/bin/env bash
# Train JiT with FD loss computed from the frozen ViT-S or ViT-B three-head projector.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export HF_HOME="${HF_HOME:-/mmu-vcg/gaomingju/data/models/}"
export TORCH_HOME="${TORCH_HOME:-/mmu-vcg/gaomingju/data/models/}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export DATA_ROOT="${DATA_ROOT:-/mmu-vcg/zhangxu34/datasets/ImageNet-1K/}"

: "${CKPT_ROOT:=./checkpoints/base}"
: "${OUTPUT_DIR:=/mmu-vcg/gaomingju/workspace/foundation/FD-Loss-Ours/work_dirs}"
: "${PROJECT:=projector}"
: "${NNODES:=1}"
: "${NODE_RANK:=0}"
: "${MASTER_ADDR:=127.0.0.1}"
: "${MASTER_PORT:=29631}"
: "${GPUS_PER_NODE:=8}"
: "${GLOBAL_BSZ:=1024}"
: "${ENABLE_WANDB:=1}"
: "${MODEL_SIZE:=B}"
: "${BACKBONE:=vit_s}"
: "${PROJECTOR_VERSION:=III}"
: "${HEAD_VARIANT:=default}"
: "${HEAD_CONV_LAYERS:=2}"
: "${FD_PROJECTOR_DTYPE:=bf16}"
: "${PROJECTOR_USE_MUON:=1}"

case "${BACKBONE}" in
    vit_s|vits|ViT-S|ViT_S|s|S|small|vit-small|vit_small)
        BACKBONE="vit_s"
        BACKBONE_TAG="vits"
        DEFAULT_PATCH_SIZE=16
        DEFAULT_EMBED_DIM=384
        DEFAULT_DEPTH=12
        DEFAULT_NUM_HEADS=6
        DEFAULT_HEAD_HIDDEN_DIM=1536
        ;;
    vit_b|vitb|ViT-B|ViT_B|b|B|base|vit-base|vit_base)
        BACKBONE="vit_b"
        BACKBONE_TAG="vitb"
        DEFAULT_PATCH_SIZE=16
        DEFAULT_EMBED_DIM=768
        DEFAULT_DEPTH=12
        DEFAULT_NUM_HEADS=12
        DEFAULT_HEAD_HIDDEN_DIM=3072
        ;;
    *)
        echo "[ERR] unsupported BACKBONE=${BACKBONE}; expected vit_s or vit_b" >&2
        exit 1
        ;;
esac

: "${PATCH_SIZE:=${DEFAULT_PATCH_SIZE}}"
: "${EMBED_DIM:=${DEFAULT_EMBED_DIM}}"
: "${DEPTH:=${DEFAULT_DEPTH}}"
: "${NUM_HEADS:=${DEFAULT_NUM_HEADS}}"
: "${HEAD_HIDDEN_DIM:=${DEFAULT_HEAD_HIDDEN_DIM}}"

HEAD_VARIANT_STATS_SUFFIX=""
HEAD_VARIANT_EXP_SUFFIX=""
if [[ "${HEAD_VARIANT}" != "default" ]]; then
    HEAD_VARIANT_STATS_SUFFIX="_${HEAD_VARIANT}"
    HEAD_VARIANT_EXP_SUFFIX="-${HEAD_VARIANT}"
fi
PROJECTOR_OPTIM_SUFFIX=""
if [[ "${PROJECTOR_USE_MUON}" = "1" ]]; then
    PROJECTOR_OPTIM_SUFFIX="-muon"
fi
: "${PROJECTOR_EXP:=${BACKBONE_TAG}-token-v${PROJECTOR_VERSION}-inception-mae-siglip-mse-fdema${HEAD_VARIANT_EXP_SUFFIX}${PROJECTOR_OPTIM_SUFFIX}}"
: "${PROJECTOR_CKPT:=/mmu-vcg/gaomingju/workspace/foundation/FD-Loss-Ours/work_dirs/projector/${PROJECTOR_EXP}/checkpoints/last.pt}"
: "${PROJECTOR_SIGLIP_STATS:=data/fid_stats/projector_${BACKBONE_TAG}_v${PROJECTOR_VERSION}_siglip_in256${HEAD_VARIANT_STATS_SUFFIX}_stats.npz}"
: "${PROJECTOR_MAE_STATS:=data/fid_stats/projector_${BACKBONE_TAG}_v${PROJECTOR_VERSION}_mae_in256${HEAD_VARIANT_STATS_SUFFIX}_stats.npz}"
: "${PROJECTOR_INCEPTION_STATS:=data/fid_stats/projector_${BACKBONE_TAG}_v${PROJECTOR_VERSION}_inception_in256${HEAD_VARIANT_STATS_SUFFIX}_stats.npz}"

TOTAL_GPUS=$(( NNODES * GPUS_PER_NODE ))
if (( GLOBAL_BSZ % TOTAL_GPUS != 0 )); then
    echo "[ERR] GLOBAL_BSZ=${GLOBAL_BSZ} must be divisible by TOTAL_GPUS=${TOTAL_GPUS}" >&2
    exit 1
fi
BATCH_SIZE=$(( GLOBAL_BSZ / TOTAL_GPUS ))

WANDB_ARGS=(--disable_wandb)
if [[ "${ENABLE_WANDB}" = "1" ]]; then
    WANDB_ARGS=(--enable_wandb)
fi

MAE="vit_large_patch16_224.mae"
SIGLIP="vit_so400m_patch16_siglip_256.v2_webli"

case "${MODEL_SIZE}" in
    B)
        MODEL=JiT_B; CFG=3.0; INTERVAL_MIN=0.1; INTERVAL_MAX=1.0
        LOAD="${CKPT_ROOT}/JiT-B.pth" ;;
    L)
        MODEL=JiT_L; CFG=2.4; INTERVAL_MIN=0.1; INTERVAL_MAX=1.0
        LOAD="${CKPT_ROOT}/JiT-L.pth" ;;
    H)
        MODEL=JiT_H; CFG=2.2; INTERVAL_MIN=0.1; INTERVAL_MAX=1.0
        LOAD="${CKPT_ROOT}/JiT-H.pth" ;;
    *)
        echo "[ERR] unsupported MODEL_SIZE=${MODEL_SIZE}; expected B, L, or H" >&2
        exit 1 ;;
esac

: "${EXP_NAME:=${MODEL}-fd-projector-v${PROJECTOR_VERSION}-mse}"

for stats_path in "${PROJECTOR_SIGLIP_STATS}" "${PROJECTOR_MAE_STATS}" "${PROJECTOR_INCEPTION_STATS}"; do
    if [[ ! -f "${stats_path}" ]]; then
        echo "[ERR] Missing projector FD stats: ${stats_path}" >&2
        echo "Run first:" >&2
        echo "  DATA_ROOT=${DATA_ROOT} GPUS_PER_NODE=${GPUS_PER_NODE} bash scripts/projector/compute_vits_projector_ref_stats.sh" >&2
        exit 1
    fi
done

torchrun \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --nproc_per_node="${GPUS_PER_NODE}" \
    main_fd_projector.py \
    --project "${PROJECT}" \
    --exp_name "${EXP_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --data_path "${DATA_ROOT}" \
    --load_from "${LOAD}" \
    --model "${MODEL}" --rope_2d --learned_pe --legacy_time_convention \
    --cfg "${CFG}" --interval_min "${INTERVAL_MIN}" --interval_max "${INTERVAL_MAX}" \
    --ema_type edm \
    --num_sampling_steps 1 \
    --eval_bsz 256 --num_images_for_eval_and_search 50000 \
    --vis_freq 100 --online_eval --eval_freq 10000 \
    --print_freq 20 --milestone_interval 10 --save_freq 5 \
    --epochs 100 --steps_per_epoch 1250 --warmup_epochs 5 \
    --lr 1e-5 --lr_sched cosine --min_lr 0.0 \
    --grad_checkpointing \
    --fd_eigvalsh --fd_ema_beta 0.999 \
    --fd_repr_models "${SIGLIP}" "${MAE}" inception \
    --fd_repr_stats_paths "${PROJECTOR_SIGLIP_STATS}" "${PROJECTOR_MAE_STATS}" "${PROJECTOR_INCEPTION_STATS}" \
    --fd_repr_pool_types cls cls cls \
    --fd_target_sizes 224 224 256 \
    --fd_projector_checkpoint "${PROJECTOR_CKPT}" \
    --fd_projector_version "${PROJECTOR_VERSION}" \
    --fd_projector_img_size 256 \
    --fd_projector_backbone "${BACKBONE}" \
    --fd_projector_patch_size "${PATCH_SIZE}" \
    --fd_projector_embed_dim "${EMBED_DIM}" \
    --fd_projector_depth "${DEPTH}" \
    --fd_projector_num_heads "${NUM_HEADS}" \
    --fd_projector_head_variant "${HEAD_VARIANT}" \
    --fd_projector_head_mlp_layers 1 \
    --fd_projector_head_hidden_dim "${HEAD_HIDDEN_DIM}" \
    --fd_projector_head_conv_layers "${HEAD_CONV_LAYERS}" \
    --fd_projector_dtype "${FD_PROJECTOR_DTYPE}" \
    --auto_resume \
    "${WANDB_ARGS[@]}" \
    "$@"
