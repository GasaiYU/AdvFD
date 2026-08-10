#!/usr/bin/env bash
# Table 3: pMF at 256px and 512px with AMFD on the static branch and
# adversarial FD. Set MODEL_SIZE in {B,L,H} and RES in {256,512}.
#
# AMFD variant of table_3_pMF_adv.sh. Runs main_amfd.py, which replaces the
# static FD loss with AMFD on every --fd_repr_models entry and leaves the
# FD-Adv branch untouched. Everything else matches the baseline script, so the
# two are directly comparable.
#
# Set AMFD_STATIC=0 to reproduce the plain-FD baseline through this same file.

set -euo pipefail

export HF_HOME="${HF_HOME:-/mmu-vcg/gaomingju/data/models/}"
export TORCH_HOME="${TORCH_HOME:-/mmu-vcg/gaomingju/data/models/}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export DATA_ROOT="${DATA_ROOT:-/mmu-vcg/zhangxu34/datasets/ImageNet-1K/}"

: "${DATA_ROOT:?Set DATA_ROOT to the ImageNet root with train/ and val/ subdirectories}"
: "${CKPT_ROOT:=./checkpoints/base}"
: "${OUTPUT_DIR:=/mmu-vcg/gaomingju/workspace/foundation/FD-Loss-Ours/work_dirs}"
: "${NNODES:=1}"
: "${NODE_RANK:=0}"
: "${MASTER_ADDR:=127.0.0.1}"
: "${MASTER_PORT:=29500}"
: "${GPUS_PER_NODE:=8}"
: "${GLOBAL_BSZ:=1024}"
: "${ENABLE_WANDB:=1}"
: "${MODEL_SIZE:=B}"
: "${RES:=256}"
: "${MAE:=vit_large_patch16_224.mae}"
: "${SIGLIP:=vit_so400m_patch16_siglip_256.v2_webli}"
: "${FD_MAIN_REPRS:=sim}"       # inception | sim
: "${FD_ADV_REPRS:=inception}"  # follow | inception | sim
: "${FD_ADV_WEIGHT:=0.05}" # fix 0.05
: "${FD_ADV_LR:=2e-6}" # fix 2e-6
: "${FD_ADV_STEPS:=1}"
: "${FD_ADV_UPDATE_FREQ:=2}"
: "${FD_ADV_GRAD_CLIP:=1.0}"
: "${FD_ADV_DETACH_REAL:=1}"
: "${FD_WHITEN:=0}"
: "${FD_WHITEN_EPS:=1e-3}" # fix 1e-3
: "${FD_ADV_START_STEP:=1000}"
: "${FD_ADV_WARMUP_STEPS:=4000}"
: "${FD_ADV_WHITEN_EPS:=1e-3}"
: "${FD_ADV_WHITEN:=1}"
: "${FD_ADV_NEG_REAL_DEGRADE_RATIO:=0}"
: "${FD_ADV_LOG_RAW:=1}"
: "${FD_ADV_LOG_RAW_FREQ:=1000}"
: "${FD_ADV_EMA_BETA:=0.99}"

# AMFD on the static FD branch. AMFD_STATIC=0 leaves this script's behaviour
# byte-identical to before. When 1, the static FD loss is replaced by AMFD on
# every --fd_repr_models entry; the Inception FD-Adv branch is untouched.
# Defaults below follow the official AMFD ImageNet launcher
# (github.com/poppuppy/amfd, scripts/train_imagenet_pmf.sh): c2048/d16/a4,
# manual JVP, t=0.25, one amortizer update per generator update, and
# per-encoder generator-loss normalization.
: "${AMFD_STATIC:=1}"
: "${AMORT_UNCOND:=1}"   # 1 = AMFD-U. Upstream Table 1 shows conditional is
                         # worse on ImageNet class labels (MSE_mu 5.75 vs
                         # 0.0106 x 1e-3), so AMFD-U is the default here.
: "${AMORT_LR:=1e-4}"
: "${AMORT_MODEL_CHANNELS:=2048}"
: "${AMORT_DEPTH:=16}"
: "${AMORT_NUM_ADALN_BLOCKS:=4}"
: "${AMORT_JVP_IMPL:=manual}"
: "${AMORT_T:=0.25}"
: "${AMORT_UPDATES_PER_GEN_UPDATE:=1}"
: "${AMORT_GRAD_CLIP:=1.0}"
: "${AMORT_EMA_DECAY:=0.0}"
: "${AMFD_LOG_FD_FREQ:=50}"

TOTAL_GPUS=$(( NNODES * GPUS_PER_NODE ))
BATCH_SIZE=$(( GLOBAL_BSZ / TOTAL_GPUS ))
WANDB_FLAG=--disable_wandb
if [ "$ENABLE_WANDB" = "1" ]; then
    WANDB_FLAG=--enable_wandb
fi

FD_ADV_LOG_RAW_FLAG=
if [ "$FD_ADV_LOG_RAW" = "1" ]; then
    FD_ADV_LOG_RAW_FLAG=--fd_adv_log_raw
fi

FD_WHITEN_ARGS=()
FD_WHITEN_SUFFIX=
if [ "$FD_WHITEN" = "1" ]; then
    FD_WHITEN_ARGS=(--fd_whiten)
    FD_WHITEN_SUFFIX="-fdwhiten-eps${FD_WHITEN_EPS}"
fi

FD_ADV_WHITEN_ARGS=()
FD_ADV_WHITEN_SUFFIX=
if [ "$FD_ADV_WHITEN" = "0" ]; then
    FD_ADV_WHITEN_ARGS=(--fd_adv_no_whiten)
    FD_ADV_WHITEN_SUFFIX="-advnowhiten"
fi

FD_ADV_DETACH_REAL_ARGS=()
FD_ADV_DETACH_REAL_SUFFIX=
if [ "$FD_ADV_DETACH_REAL" = "1" ]; then
    FD_ADV_DETACH_REAL_ARGS=(--fd_adv_detach_real)
    FD_ADV_DETACH_REAL_SUFFIX="-detachreal"
fi

FD_ADV_UPDATE_SUFFIX=
if [ "$FD_ADV_UPDATE_FREQ" != "1" ]; then
    FD_ADV_UPDATE_SUFFIX="-advfreq${FD_ADV_UPDATE_FREQ}"
fi

AMFD_ARGS=()
AMFD_SUFFIX=
if [ "$AMFD_STATIC" = "1" ]; then
    AMFD_ARGS=(
        --amfd_static
        --amort_lr "$AMORT_LR"
        --amort_model_channels "$AMORT_MODEL_CHANNELS"
        --amort_depth "$AMORT_DEPTH"
        --amort_num_adaln_blocks "$AMORT_NUM_ADALN_BLOCKS"
        --amort_jvp_impl "$AMORT_JVP_IMPL"
        --amort_t "$AMORT_T"
        --amort_updates_per_gen_update "$AMORT_UPDATES_PER_GEN_UPDATE"
        --amort_grad_clip "$AMORT_GRAD_CLIP"
        --amort_ema_decay "$AMORT_EMA_DECAY"
        --amort_normalize_gen_loss_per_encoder
        --amort_gen_loss_norm_eps 0.01
        --amort_gen_loss_norm_power 1.0
        --amfd_log_fd_freq "$AMFD_LOG_FD_FREQ"
    )
    AMFD_SUFFIX="-amfd-c${AMORT_MODEL_CHANNELS}d${AMORT_DEPTH}a${AMORT_NUM_ADALN_BLOCKS}-t${AMORT_T}-lr${AMORT_LR}"
    if [ "$AMORT_UNCOND" = "1" ]; then
        AMFD_ARGS+=(--amort_uncond)
        AMFD_SUFFIX="${AMFD_SUFFIX}-uncond"
    else
        AMFD_SUFFIX="${AMFD_SUFFIX}-cond"
    fi
fi

FD_REPR_ARGS=()
case "${FD_MAIN_REPRS}" in
    inception)
        FD_REPR_ARGS=(--fd_repr_models inception)
        FD_MAIN_TAG="fd-inception" ;;
    sim)
        FD_REPR_ARGS=(
            --fd_repr_models "$SIGLIP" "$MAE" inception
            --fd_repr_pool_types cls cls cls
            --fd_target_sizes 224 224 256
        )
        FD_MAIN_TAG="fd-sim" ;;
    *) echo "[ERR] unsupported FD_MAIN_REPRS=${FD_MAIN_REPRS}"; exit 1 ;;
esac

FD_ADV_REPR_ARGS=()
case "${FD_ADV_REPRS}" in
    follow)
        FD_ADV_TAG="-advfollow" ;;
    inception)
        FD_ADV_REPR_ARGS=(--fd_adv_repr_models inception)
        FD_ADV_TAG="-advinc" ;;
    sim)
        FD_ADV_REPR_ARGS=(
            --fd_adv_repr_models "$SIGLIP" "$MAE" inception
            --fd_adv_repr_pool_types cls cls cls
            --fd_adv_target_sizes 224 224 256
        )
        FD_ADV_TAG="-advsim" ;;
    *) echo "[ERR] unsupported FD_ADV_REPRS=${FD_ADV_REPRS}"; exit 1 ;;
esac

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

# Align --cfg with the official AMFD launcher, which uses 1.0 (i.e. no CFG) for
# both JiT and pMF. The per-model CFG values set above are the FD-Adv
# baseline's. Set AMFD_CFG to override -- AMFD_CFG=8.5 recovers pMF_B_256's
# baseline value.
CFG="${AMFD_CFG:-1.0}"

run_one() {
    local exp_name="$1"
    shift
    torchrun \
        --nnodes="$NNODES" \
        --node_rank="$NODE_RANK" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        --nproc_per_node="$GPUS_PER_NODE" \
        main_amfd.py \
        --project table_3_pMF_amfd \
        --exp_name "$exp_name" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size "$BATCH_SIZE" \
        --data_path "$DATA_ROOT" \
        --load_from "$LOAD" \
        --model "$MODEL" --rope_2d --learned_pe --disable_v_head \
        --cfg "$CFG" --interval_min "$INTERVAL_MIN" --interval_max "$INTERVAL_MAX" \
        --num_sampling_steps 1 \
        --eval_bsz 256 --num_images_for_eval_and_search 50000 \
        --vis_freq 25 --online_eval --eval_freq 1000 \
        --print_freq 20 --milestone_interval 10 --save_freq 5 \
        --epochs 100 --steps_per_epoch 1250 --warmup_epochs 1 \
        --lr 1e-6 --lr_sched cosine --min_lr 0.0 \
        --grad_checkpointing \
        --fd_repr_grad_checkpoint_models siglip \
        --fd_eigvalsh --fd_ema_beta 0.999 \
        "${FD_WHITEN_ARGS[@]}" \
        --fd_whiten_eps "$FD_WHITEN_EPS" \
        --auto_resume "$WANDB_FLAG" \
        "${EXTRA[@]}" \
        "$@"
}

run_one "${MODEL}_${RES}-${FD_MAIN_TAG}${FD_ADV_TAG}-w${FD_ADV_WEIGHT}${FD_ADV_UPDATE_SUFFIX}${FD_ADV_DETACH_REAL_SUFFIX}${FD_WHITEN_SUFFIX}${FD_ADV_WHITEN_SUFFIX}-${FD_ADV_LR}${AMFD_SUFFIX}" \
    "${FD_REPR_ARGS[@]}" \
    "${FD_ADV_REPR_ARGS[@]}" \
    --fd_adv_weight "$FD_ADV_WEIGHT" \
    --fd_adv_backbone repr \
    --fd_adv_lr "$FD_ADV_LR" \
    --fd_adv_steps "$FD_ADV_STEPS" \
    --fd_adv_update_freq "$FD_ADV_UPDATE_FREQ" \
    --fd_adv_grad_clip "$FD_ADV_GRAD_CLIP" \
    "${FD_ADV_DETACH_REAL_ARGS[@]}" \
    --fd_adv_start_step "$FD_ADV_START_STEP" \
    --fd_adv_warmup_steps "$FD_ADV_WARMUP_STEPS" \
    --fd_adv_whiten_eps "$FD_ADV_WHITEN_EPS" \
    "${FD_ADV_WHITEN_ARGS[@]}" \
    --fd_adv_neg_real_degrade_ratio "$FD_ADV_NEG_REAL_DEGRADE_RATIO" \
    $FD_ADV_LOG_RAW_FLAG \
    --fd_adv_log_raw_freq "$FD_ADV_LOG_RAW_FREQ" \
    --fd_adv_ema_beta "$FD_ADV_EMA_BETA" \
    "${AMFD_ARGS[@]}" \
    "$@"
