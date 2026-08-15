#!/usr/bin/env bash
# Table 3: JiT at 256px with AMFD on the static branch and Inception FD-Adv.
# Set MODEL_SIZE in {B,L,H}.
#
# AMFD variant of table_3_JiT_adv_fd_sim.sh. Runs main_amfd.py, which replaces
# the static FD loss with AMFD on every --fd_repr_models entry. Baseline CFG and
# five-epoch generator warmup remain the defaults; AMFD mode additionally uses
# the guarded FD-Adv recipe assembled below.
#
# Set AMFD_STATIC=0 to select the plain-FD baseline defaults through this file.

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
: "${MODEL_SIZE:=B}"
: "${MAIN_WARMUP_EPOCHS:=5}"
: "${AMFD_STATIC:=1}"
: "${MAE:=vit_large_patch16_224.mae}"
: "${SIGLIP:=vit_so400m_patch16_siglip_256.v2_webli}"
: "${LOAD_INIT:=base}"  # base | fd75k | custom | none
: "${LOAD_FROM:=}"
: "${FD_ADV_WEIGHT:=0.1}"
: "${FD_ADV_LR:=1e-6}" # Need fixing
: "${FD_ADV_STEPS:=1}"
: "${FD_ADV_UPDATE_FREQ:=2}"
: "${FD_ADV_GRAD_CLIP:=1.0}"
: "${FD_ADV_DETACH_REAL:=1}"
: "${FD_WHITEN:=0}"
: "${FD_WHITEN_EPS:=1e-3}"
: "${FD_ADV_WARMUP_STEPS:=4000}"
: "${FD_ADV_WHITEN:=1}"
: "${FD_ADV_NEG_REAL_DEGRADE_RATIO:=0}"
: "${FD_ADV_LOG_RAW:=1}"
: "${FD_ADV_LOG_RAW_FREQ:=1000}"
: "${FD_ADV_EMA_BETA:=0.99}"
: "${FD_ADV_RESIDUAL_RMS_LOG_FREQ:=100}"
: "${FD_ADV_LOG_FEATURE_SCALE_FREQ:=20}"

# AMFD needs time to learn a useful static generator gradient before the
# critic enters. The trust region is the primary long-horizon scale guard;
# LR warmup and finite checks handle startup and last-resort numerical faults.
if [ "$AMFD_STATIC" = "1" ]; then
    : "${FD_ADV_START_STEP:=10000}"
    : "${FD_ADV_WHITEN_EPS:=5e-3}"
    : "${FD_ADV_RESIDUAL_RMS_KAPPA:=0.2}"
    : "${FD_ADV_FREEZE_BATCHNORM:=1}"
    : "${FD_ADV_LOG_FEATURE_SCALE:=1}"
    : "${FD_ADV_CRITIC_LR_WARMUP:=1}"
    : "${FD_ADV_FINITE_GUARD:=1}"
else
    : "${FD_ADV_START_STEP:=1000}"
    : "${FD_ADV_WHITEN_EPS:=1e-3}"
    : "${FD_ADV_RESIDUAL_RMS_KAPPA:=0}"
    : "${FD_ADV_FREEZE_BATCHNORM:=0}"
    : "${FD_ADV_LOG_FEATURE_SCALE:=0}"
    : "${FD_ADV_CRITIC_LR_WARMUP:=0}"
    : "${FD_ADV_FINITE_GUARD:=0}"
fi

# AMFD on the static FD branch. AMFD_STATIC=0 selects the plain-FD baseline
# defaults. When 1, the static FD loss is replaced by AMFD on every
# --fd_repr_models entry and the guarded FD-Adv recipe is enabled.
# Defaults below follow the official AMFD ImageNet launcher
# (github.com/poppuppy/amfd, scripts/train_imagenet_jit.sh): c2048/d16/a4,
# manual JVP, t=0.25, one amortizer update per generator update, and
# per-encoder generator-loss normalization.
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
FD_ADV_LOG_RAW_FLAG=
if [ "$FD_ADV_LOG_RAW" = "1" ]; then
    FD_ADV_LOG_RAW_FLAG=--fd_adv_log_raw
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

FD_ADV_STABILITY_ARGS=(
    --fd_adv_residual_rms_kappa "$FD_ADV_RESIDUAL_RMS_KAPPA"
    --fd_adv_residual_rms_log_freq "$FD_ADV_RESIDUAL_RMS_LOG_FREQ"
    --fd_adv_log_feature_scale_freq "$FD_ADV_LOG_FEATURE_SCALE_FREQ"
)
if [ "$FD_ADV_FREEZE_BATCHNORM" = "1" ]; then
    FD_ADV_STABILITY_ARGS+=(--fd_adv_freeze_batchnorm)
fi
if [ "$FD_ADV_LOG_FEATURE_SCALE" = "1" ]; then
    FD_ADV_STABILITY_ARGS+=(--fd_adv_log_feature_scale)
fi
if [ "$FD_ADV_CRITIC_LR_WARMUP" = "1" ]; then
    FD_ADV_STABILITY_ARGS+=(--fd_adv_critic_lr_warmup)
fi
if [ "$FD_ADV_FINITE_GUARD" = "1" ]; then
    FD_ADV_STABILITY_ARGS+=(--fd_adv_finite_guard)
fi

case "${MODEL_SIZE}" in
    B)
        MODEL=JiT_B; CFG=3.0; INTERVAL_MIN=0.1; INTERVAL_MAX=1.0
        case "${LOAD_INIT}" in
            base) LOAD="${CKPT_ROOT}/JiT-B.pth" ;;
            fd75k) LOAD="$LOAD_FROM" ;;
            custom) LOAD="$LOAD_FROM" ;;
            none) LOAD="" ;;
            *) echo "[ERR] unsupported LOAD_INIT=${LOAD_INIT}"; exit 1 ;;
        esac ;;
    L)
        MODEL=JiT_L; CFG=2.4; INTERVAL_MIN=0.1; INTERVAL_MAX=1.0
        LOAD="${CKPT_ROOT}/JiT-L.pth" ;;
    H)
        MODEL=JiT_H; CFG=2.2; INTERVAL_MIN=0.1; INTERVAL_MAX=1.0
        LOAD="${CKPT_ROOT}/JiT-H.pth" ;;
    *) echo "[ERR] unsupported MODEL_SIZE=${MODEL_SIZE}"; exit 1 ;;
esac

# Keep the stable FD-Adv baseline CFG by default. AMFD_CFG remains an explicit
# override for ablations.
CFG="${AMFD_CFG:-$CFG}"
RECIPE_SUFFIX="-cfg${CFG}-mw${MAIN_WARMUP_EPOCHS}-as${FD_ADV_START_STEP}-aw${FD_ADV_WARMUP_STEPS}-aweps${FD_ADV_WHITEN_EPS}-rr${FD_ADV_RESIDUAL_RMS_KAPPA}-fbn${FD_ADV_FREEZE_BATCHNORM}-dlrw${FD_ADV_CRITIC_LR_WARMUP}-fg${FD_ADV_FINITE_GUARD}"

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
        main_amfd.py \
        --project Jit-B-adv-amfd \
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
        --epochs 100 --steps_per_epoch 1250 --warmup_epochs "$MAIN_WARMUP_EPOCHS" \
        --lr 1e-5 --lr_sched cosine --min_lr 0.0 \
        --grad_checkpointing \
        --fd_repr_grad_checkpoint_models siglip \
        --fd_eigvalsh --fd_ema_beta 0.999 \
        "${FD_WHITEN_ARGS[@]}" \
        --fd_whiten_eps "$FD_WHITEN_EPS" \
        --auto_resume --enable_wandb \
        "$@"
}

run_one "${MODEL}-fd-sim-advinc-w${FD_ADV_WEIGHT}-from-${LOAD_INIT}-adv-lr-${FD_ADV_LR}${AMFD_SUFFIX}${RECIPE_SUFFIX}" \
    --fd_repr_models "$SIGLIP" "$MAE" inception \
    --fd_repr_pool_types cls cls cls \
    --fd_target_sizes 224 224 256 \
    --fd_adv_repr_models inception \
    --fd_adv_weight "$FD_ADV_WEIGHT" \
    --fd_adv_backbone repr \
    --fd_adv_lr "$FD_ADV_LR" \
    --fd_adv_steps "$FD_ADV_STEPS" \
    --fd_adv_update_freq "$FD_ADV_UPDATE_FREQ" \
    --fd_adv_grad_clip "$FD_ADV_GRAD_CLIP" \
    "${FD_ADV_STABILITY_ARGS[@]}" \
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
