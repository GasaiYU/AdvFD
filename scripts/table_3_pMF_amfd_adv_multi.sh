#!/usr/bin/env bash
# Table 3: multi-node pMF at 256px/512px with AMFD on the static branch and
# adversarial FD. Set MODEL_SIZE in {B,L,H} and RES in {256,512}.
#
# Two-node copy of table_3_pMF_amfd_adv.sh. The loss, model, and AMFD recipe are
# identical; only the launcher geometry changes:
#
#   NNODES         1    -> 2
#   TOTAL_GPUS     8    -> 16
#   GLOBAL_BSZ     1024 -> 1024   unchanged, so per-GPU batch drops 128 -> 64
#   --lr           1e-6 -> 1e-6   unchanged, because the global batch is
#                                 unchanged; optimizer arithmetic per step
#                                 matches the single-node run.
#
# Holding the global batch fixed is the point of this script: the 100x1250-step
# schedule, the 1e-6 cosine LR, and the FD EMA window all stay comparable with
# the single-node numbers. The cost is scaling efficiency. The FD loss
# all-gathers features every step, so halving the per-GPU batch to 64 raises the
# comms-to-compute ratio, and 2 nodes will land well short of a 2x speedup. If
# you want throughput instead of comparability, set GLOBAL_BSZ=2048 to keep 128
# samples per GPU and accept that the 1e-6 learning rate then needs retuning.
#
# AMFD itself is world-size agnostic. update_amortizers all-reduces amortizer
# gradients before stepping, and the amortizer optimizer is ZeRO-1, so its AdamW
# moments shard over all 16 ranks instead of 8 -- per-rank amortizer optimizer
# memory roughly halves relative to the single-node run.
#
# Launch on every node with the same NNODES, MASTER_ADDR, and MASTER_PORT, and a
# unique NODE_RANK:
#
#   # Node 0 (replace 10.0.0.1 with node 0's reachable IP/hostname):
#   MODEL_SIZE=B RES=256 NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 \
#       bash scripts/table_3_pMF_amfd_adv_multi.sh
#
#   # Node 1:
#   MODEL_SIZE=B RES=256 NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.0.1 \
#       bash scripts/table_3_pMF_amfd_adv_multi.sh
#
# OUTPUT_DIR must be on a shared filesystem, otherwise --auto_resume and
# checkpointing break across nodes. CKPT_ROOT and DATA_ROOT must be readable
# from both nodes.
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
: "${NNODES:=2}"
# Keep these :? messages free of apostrophes: inside "${VAR:?...}" a single quote
# is still active syntax, so a stray one swallows the following lines and turns
# the next guard into dead code.
: "${NODE_RANK:?Set NODE_RANK to the zero-based rank of this node (0..NNODES-1)}"
: "${MASTER_ADDR:?Set MASTER_ADDR to the reachable IP address or hostname of node 0}"
: "${MASTER_PORT:=29500}"
: "${GPUS_PER_NODE:=8}"
# Unchanged from the single-node script on purpose. See the header.
: "${GLOBAL_BSZ:=1024}"
: "${ENABLE_WANDB:=1}"
: "${MODEL_SIZE:=B}"
: "${RES:=256}"
: "${MAE:=vit_large_patch16_224.mae}"
: "${SIGLIP:=vit_so400m_patch16_siglip_256.v2_webli}"
: "${FD_MAIN_REPRS:=sim}"       # inception | sim
# Space-separated selectors for --fd_repr_grad_checkpoint_models, matched
# against each repr's full name and short name. Empty string omits the flag
# entirely, which turns repr grad checkpointing off for every encoder.
#
#   FD_CKPT_MODELS="siglip"      default; SigLIP only
#   FD_CKPT_MODELS="siglip mae"  both ViTs
#   FD_CKPT_MODELS=""            off for all reprs
#   FD_CKPT_MODELS="all"         every repr (see the inception caveat below)
#
# Inception cannot checkpoint regardless: load_repr_model's inception branch
# never forwards grad_checkpointing, so "all" is in effect "siglip mae".
# This setting is memory-vs-compute only and does not change results, so it is
# deliberately absent from exp_name -- runs that differ only here can resume
# from each other.
# "=" not ":=" on purpose: ":=" substitutes the default for an empty value too,
# which would make FD_CKPT_MODELS="" silently mean "siglip". With "=" the
# default applies only when the variable is unset, so an explicit empty string
# survives and switches repr grad checkpointing off. The assignment form still
# defines the variable, which set -u requires below.
: "${FD_CKPT_MODELS=siglip}"
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

# AMFD on the static FD branch. AMFD_STATIC=0 leaves this script equivalent to
# the plain-FD baseline. When 1, the static FD loss is replaced by AMFD on every
# --fd_repr_models entry; the Inception FD-Adv branch is untouched.
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

if ! [[ "$NNODES" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERR] NNODES must be a positive integer, got: $NNODES" >&2
    exit 1
fi
if ! [[ "$NODE_RANK" =~ ^[0-9]+$ ]] || (( NODE_RANK >= NNODES )); then
    echo "[ERR] NODE_RANK must be in [0, $((NNODES - 1))], got: $NODE_RANK" >&2
    exit 1
fi
if ! [[ "$GPUS_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERR] GPUS_PER_NODE must be a positive integer, got: $GPUS_PER_NODE" >&2
    exit 1
fi
if ! [[ "$GLOBAL_BSZ" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERR] GLOBAL_BSZ must be a positive integer, got: $GLOBAL_BSZ" >&2
    exit 1
fi

TOTAL_GPUS=$(( NNODES * GPUS_PER_NODE ))
if (( GLOBAL_BSZ % TOTAL_GPUS != 0 )); then
    echo "[ERR] GLOBAL_BSZ=$GLOBAL_BSZ must be divisible by total GPUs=$TOTAL_GPUS" >&2
    exit 1
fi
BATCH_SIZE=$(( GLOBAL_BSZ / TOTAL_GPUS ))

if (( NNODES > 1 )) && [ "$MASTER_ADDR" = "127.0.0.1" ]; then
    echo "[ERR] NNODES=$NNODES but MASTER_ADDR=127.0.0.1; use node 0's routable address" >&2
    exit 1
fi

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

# Unquoted on purpose: FD_CKPT_MODELS is a space-separated selector list that
# has to word-split into separate argv entries. --fd_repr_grad_checkpoint_models
# is nargs="+", so passing the flag with no values is an argparse error; an
# empty FD_CKPT_MODELS therefore has to drop the flag altogether.
FD_CKPT_ARGS=()
if [ -n "$FD_CKPT_MODELS" ]; then
    FD_CKPT_ARGS=(--fd_repr_grad_checkpoint_models $FD_CKPT_MODELS)
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

echo "[INFO] node_rank=${NODE_RANK}/${NNODES}, gpus_per_node=${GPUS_PER_NODE}, total_gpus=${TOTAL_GPUS}"
echo "[INFO] global_batch_size=${GLOBAL_BSZ}, batch_size_per_gpu=${BATCH_SIZE}"
echo "[INFO] rendezvous=${MASTER_ADDR}:${MASTER_PORT}"
echo "[INFO] fd_repr_grad_checkpoint_models=${FD_CKPT_MODELS:-<none>}"

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
        "${FD_CKPT_ARGS[@]}" \
        --fd_eigvalsh --fd_ema_beta 0.999 \
        "${FD_WHITEN_ARGS[@]}" \
        --fd_whiten_eps "$FD_WHITEN_EPS" \
        --auto_resume "$WANDB_FLAG" \
        "${EXTRA[@]}" \
        "$@"
}

# -gbs/-n suffix keeps this run's output_dir distinct from the single-node
# script's, so --auto_resume cannot pick up a checkpoint from a different
# launcher geometry.
run_one "${MODEL}_${RES}-${FD_MAIN_TAG}${FD_ADV_TAG}-w${FD_ADV_WEIGHT}${FD_ADV_UPDATE_SUFFIX}${FD_ADV_DETACH_REAL_SUFFIX}${FD_WHITEN_SUFFIX}${FD_ADV_WHITEN_SUFFIX}-${FD_ADV_LR}${AMFD_SUFFIX}-gbs${GLOBAL_BSZ}-n${NNODES}" \
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
