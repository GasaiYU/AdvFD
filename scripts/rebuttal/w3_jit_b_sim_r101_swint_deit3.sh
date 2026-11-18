#!/usr/bin/env bash
# W3 control: JiT-B + SIM-FD + frozen ResNet-101 + Swin-T + DeiT-III-B.
# Extra encoders (all ImageNet-1K supervised, 224 input, none is an evaluation
# encoder of FD-r6/FD-r3):
#   resnet101.a1_in1k                     2048-d GAP features
#   swin_tiny_patch4_window7_224.ms_in1k  768-d spatial-mean features
#   deit3_base_patch16_224.fb_in1k        768-d CLS token (after the final norm)
# All encoders stay frozen/eval; G gradients flow THROUGH them. Each extra FD is
# self-normalized and weighted EXTRA_FD_WEIGHT (default 1.0), like the SIM ones.
# There is no AdvFD branch in the frozen-encoder runs.
#
# MODE=overhead (default): one node, 8 GPUs, global batch 1024, JiT-B checkpoint.
#   Measures SIM, +R101, +R101+SwinT, +R101+SwinT+DeiT3 and SIM+AdvFD with the
#   actual training loop: 50 warmup + 200 measured iterations starting at virtual
#   step 5000, so AdvFD's 1k/4k ramp is complete (1 D-step every 2 G-steps,
#   global weight 0.1). Timing covers the full iteration (G forward/backward
#   through every encoder, FD eigensolves, optimizer, EMA, AdvFD D-step), max
#   across ranks; it excludes initialization, queue fill, reference statistics,
#   checkpoint I/O and evaluation. Reports incremental latency and peak ALLOCATED
#   memory relative to SIM. Cost measurements, not trained checkpoints.
#   CASES="sim sim_resnet101_swin_t_deit3 sim_advfd" measures a subset; the first
#   case must be sim (the baseline).
# MODE=stats: prepare the three reference statistics on one node.
# MODE=train: 125k G steps. NNODES=2 for two machines (batch 64/GPU).
# Missing statistics are prepared automatically when NNODES=1.
#
# GRAD_CKPT_MODELS picks repr models with gradient checkpointing, default
# "siglip deit3". Overhead and training use the same setting, so the measured
# cost includes DeiT recomputation. GRAD_CKPT_MODELS=siglip disables it for DeiT
# (faster, more memory). Results go to a directory named after this setting.
#
# MODE=overhead bash scripts/rebuttal/w3_jit_b_sim_r101_swint_deit3.sh
# MODE=stats    bash scripts/rebuttal/w3_jit_b_sim_r101_swint_deit3.sh
# MODE=train NNODES=2 NODE_RANK=0 bash scripts/rebuttal/w3_jit_b_sim_r101_swint_deit3.sh  # on 10.82.122.141
# MODE=train NNODES=2 NODE_RANK=1 bash scripts/rebuttal/w3_jit_b_sim_r101_swint_deit3.sh
# Without InfiniBand (memlock is 64 KB in this container), prepend on both nodes:
#   NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth01 GLOO_SOCKET_IFNAME=eth01
# DRY_RUN=1 prints the commands only.

set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export HF_HOME="${HF_HOME:-/mmu-vcg/gaomingju/data/models/}"
export TORCH_HOME="${TORCH_HOME:-/mmu-vcg/gaomingju/data/models/}"
# Prefer downloaded weights in $TORCH_HOME/hub/checkpoints before Hugging Face.
export TIMM_USE_OLD_CACHE="${TIMM_USE_OLD_CACHE:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
: "${MODE:=overhead}"
: "${DATA_ROOT:=/mmu-vcg/zhangxu34/datasets/ImageNet-1K/}"
: "${CKPT_ROOT:=./checkpoints/base}"
: "${OUTPUT_DIR:=./work_dirs}"
: "${NNODES:=1}"
: "${NODE_RANK:=0}"
if [[ "$NNODES" == "1" ]]; then : "${MASTER_ADDR:=127.0.0.1}"; else : "${MASTER_ADDR:=10.82.122.141}"; fi
: "${MASTER_PORT:=29500}"
: "${GPUS_PER_NODE:=8}"
: "${GLOBAL_BSZ:=1024}"
: "${SEED:=1}"
: "${EXTRA_FD_WEIGHT:=1.0}"
: "${GRAD_CKPT_MODELS:=siglip deit3}"
: "${CASES:=sim sim_resnet101 sim_resnet101_swin_t sim_resnet101_swin_t_deit3 sim_advfd}"
: "${PROFILE_WARMUP_STEPS:=50}"
: "${PROFILE_STEPS:=200}"
: "${ENABLE_WANDB:=1}"
# Reference-stat pass: STATS_DECODE_THREADS threads per GPU need no /dev/shm;
# STATS_DECODE_THREADS=0 falls back to STATS_NUM_WORKERS DataLoader workers.
: "${STATS_DECODE_THREADS:=8}"
: "${STATS_NUM_WORKERS:=10}"
: "${STATS_BATCH_SIZE:=128}"
: "${DRY_RUN:=0}"

case "$MODE" in overhead|train|stats) ;; *) echo "[ERR] MODE must be overhead, train, or stats" >&2; exit 1 ;; esac
if [[ "$MODE" == "overhead" && "$NNODES" != "1" ]]; then
    echo "[ERR] MODE=overhead runs on one node so every case uses the same GPUs and interconnect" >&2; exit 1
fi
if [[ "$NNODES" != "1" ]]; then
    case "$MASTER_ADDR" in
        localhost|127.*|::1|0.0.0.0)
            echo "[ERR] MASTER_ADDR must be reachable from the other machine, not a loopback address" >&2; exit 1 ;;
    esac
fi
TOTAL_GPUS=$(( NNODES * GPUS_PER_NODE ))
if (( TOTAL_GPUS < 1 || GLOBAL_BSZ < 1 || GLOBAL_BSZ % TOTAL_GPUS != 0 )); then
    echo "[ERR] GLOBAL_BSZ must be positive and divisible by NNODES * GPUS_PER_NODE" >&2
    exit 1
fi
BATCH_SIZE=$(( GLOBAL_BSZ / TOTAL_GPUS ))
read -r -a GRAD_CKPT <<< "$GRAD_CKPT_MODELS"
read -r -a CASE_LIST <<< "$CASES"
if (( ${#GRAD_CKPT[@]} == 0 )); then echo "[ERR] GRAD_CKPT_MODELS is empty" >&2; exit 1; fi
if [[ "${CASE_LIST[0]:-}" != "sim" ]]; then echo "[ERR] CASES must start with sim (the baseline)" >&2; exit 1; fi
for CASE in "${CASE_LIST[@]}"; do
    case "$CASE" in
        sim|sim_resnet101|sim_resnet101_swin_t|sim_resnet101_swin_t_deit3|sim_advfd) ;;
        *) echo "[ERR] Unknown case: $CASE" >&2; exit 1 ;;
    esac
done

SIGLIP=vit_so400m_patch16_siglip_256.v2_webli
MAE=vit_large_patch16_224.mae
RESNET101=resnet101.a1_in1k
RESNET101_STATS=data/fid_stats/resnet101_a1_in1k_in256_t224_stats.npz
SWIN=swin_tiny_patch4_window7_224.ms_in1k
SWIN_STATS=data/fid_stats/swin_tiny_patch4_window7_224_ms_in1k_in256_t224_stats.npz
DEIT3=deit3_base_patch16_224.fb_in1k
DEIT3_STATS=data/fid_stats/deit3_base_patch16_224_fb_in1k_in256_t224_stats.npz
PROJECT=rebuttal_w3
CKPT_TAG="${GRAD_CKPT_MODELS// /-}"
PROFILE_DIR="${OUTPUT_DIR}/${PROJECT}/overhead-r101-swint-deit3-seed${SEED}-gbs${GLOBAL_BSZ}-gpus${TOTAL_GPUS}-extra${EXTRA_FD_WEIGHT}-ckpt-${CKPT_TAG}"
LAUNCH=(torchrun --nnodes="$NNODES" --node_rank="$NODE_RANK"
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" --nproc_per_node="$GPUS_PER_NODE")

run_command() {
    printf '[W3-R101-SwinT-DeiT3] Command: '
    printf '%q ' "$@"
    printf '\n'
    if [[ "$DRY_RUN" != "1" ]]; then "$@"; fi
}

if [[ "$DRY_RUN" != "1" ]]; then
    if [[ ! -d "${DATA_ROOT}/train" ]]; then
        echo "[ERR] DATA_ROOT must contain train/: $DATA_ROOT" >&2; exit 1
    fi
    if [[ "$MODE" != "stats" && ! -f "${CKPT_ROOT}/JiT-B.pth" ]]; then
        echo "[ERR] Missing ${CKPT_ROOT}/JiT-B.pth" >&2; exit 1
    fi
fi

ensure_stats() {
    local model="$1" stats="$2"
    if [[ -f "$stats" ]]; then return 0; fi
    if [[ "$NNODES" != "1" ]]; then
        echo "[ERR] Missing on node ${NODE_RANK}: $stats. Run MODE=stats on one node first." >&2
        if [[ "$DRY_RUN" == "1" ]]; then return 0; fi
        exit 1
    fi
    run_command torchrun --standalone --nproc_per_node="$GPUS_PER_NODE" compute_repr_stats.py \
        --model "$model" --data_path "$DATA_ROOT" --img_size 256 --target_size 224 \
        --batch_size "$STATS_BATCH_SIZE" --num_workers "$STATS_NUM_WORKERS" \
        --decode_threads "$STATS_DECODE_THREADS"
}
NEED_R101=0; NEED_SWIN=0; NEED_DEIT3=0
if [[ "$MODE" == "overhead" ]]; then
    for CASE in "${CASE_LIST[@]}"; do
        case "$CASE" in *resnet101*) NEED_R101=1 ;; esac
        case "$CASE" in *swin_t*) NEED_SWIN=1 ;; esac
        case "$CASE" in *deit3*) NEED_DEIT3=1 ;; esac
    done
else
    NEED_R101=1; NEED_SWIN=1; NEED_DEIT3=1
fi
if (( NEED_R101 )); then ensure_stats "$RESNET101" "$RESNET101_STATS"; fi
if (( NEED_SWIN )); then ensure_stats "$SWIN" "$SWIN_STATS"; fi
if (( NEED_DEIT3 )); then ensure_stats "$DEIT3" "$DEIT3_STATS"; fi
if [[ "$MODE" == "stats" ]]; then exit 0; fi

COMMON=(--project "$PROJECT" --output_dir "$OUTPUT_DIR"
    --data_path "$DATA_ROOT" --load_from "${CKPT_ROOT}/JiT-B.pth"
    --model JiT_B --rope_2d --learned_pe --legacy_time_convention
    --cfg 3.0 --interval_min 0.1 --interval_max 1.0 --ema_type edm --num_sampling_steps 1
    --batch_size "$BATCH_SIZE" --seed "$SEED"
    --epochs 100 --steps_per_epoch 1250 --warmup_epochs 5
    --lr 1e-5 --lr_sched cosine --min_lr 0.0
    --grad_checkpointing --fd_repr_grad_checkpoint_models "${GRAD_CKPT[@]}"
    --fd_eigvalsh --fd_ema_beta 0.999 --fd_whiten_eps 1e-3
    --eval_bsz 256 --num_images_for_eval_and_search 50000
    --vis_freq 100 --online_eval --eval_freq 10000 --print_freq 20
    --milestone_interval 10 --save_freq 5)
SIM=(--fd_repr_models "$SIGLIP" "$MAE" inception
    --fd_repr_pool_types cls cls cls --fd_target_sizes 224 224 256)
SIM_R101=(--fd_repr_models "$SIGLIP" "$MAE" inception "$RESNET101"
    --fd_repr_pool_types cls cls cls cls --fd_target_sizes 224 224 256 224
    --fd_repr_weights 1 1 1 "$EXTRA_FD_WEIGHT")
SIM_R101_SWIN=(--fd_repr_models "$SIGLIP" "$MAE" inception "$RESNET101" "$SWIN"
    --fd_repr_pool_types cls cls cls cls cls --fd_target_sizes 224 224 256 224 224
    --fd_repr_weights 1 1 1 "$EXTRA_FD_WEIGHT" "$EXTRA_FD_WEIGHT")
SIM_R101_SWIN_DEIT3=(--fd_repr_models "$SIGLIP" "$MAE" inception "$RESNET101" "$SWIN" "$DEIT3"
    --fd_repr_pool_types cls cls cls cls cls cls --fd_target_sizes 224 224 256 224 224 224
    --fd_repr_weights 1 1 1 "$EXTRA_FD_WEIGHT" "$EXTRA_FD_WEIGHT" "$EXTRA_FD_WEIGHT")
ADV=(--fd_adv_repr_models inception --fd_adv_weight 0.1 --fd_adv_backbone repr
    --fd_adv_lr 1e-6 --fd_adv_steps 1 --fd_adv_update_freq 2 --fd_adv_grad_clip 1.0
    --fd_adv_detach_real --fd_adv_start_step 1000 --fd_adv_warmup_steps 4000
    --fd_adv_whiten_eps 1e-3 --fd_adv_neg_real_degrade_ratio 0
    --fd_adv_log_raw --fd_adv_log_raw_freq 1000 --fd_adv_ema_beta 0.99)

if [[ "$MODE" == "overhead" ]]; then
    echo "[W3-R101-SwinT-DeiT3] overhead: GPUs=${TOTAL_GPUS}, batch/GPU=${BATCH_SIZE}, global batch=${GLOBAL_BSZ}, cases: ${CASE_LIST[*]}"
    echo "[W3-R101-SwinT-DeiT3] results: ${PROFILE_DIR}"
    for CASE in "${CASE_LIST[@]}"; do
        PROFILE=(--current_step 5000 --profile_warmup_steps "$PROFILE_WARMUP_STEPS"
            --profile_steps "$PROFILE_STEPS" --profile_output "${PROFILE_DIR}/${CASE}.json")
        case "$CASE" in
            sim) REPR_ARGS=("${SIM[@]}") ;;
            sim_resnet101) REPR_ARGS=("${SIM_R101[@]}") ;;
            sim_resnet101_swin_t) REPR_ARGS=("${SIM_R101_SWIN[@]}") ;;
            sim_resnet101_swin_t_deit3) REPR_ARGS=("${SIM_R101_SWIN_DEIT3[@]}") ;;
            sim_advfd) REPR_ARGS=("${SIM[@]}" "${ADV[@]}") ;;
        esac
        run_command "${LAUNCH[@]}" scripts/rebuttal/profile_fd_overhead.py \
            "${COMMON[@]}" "${REPR_ARGS[@]}" "${PROFILE[@]}" --exp_name "overhead-${CASE//_/-}"
    done
    run_command python scripts/rebuttal/profile_fd_overhead.py --summarize "$PROFILE_DIR" "${CASE_LIST[@]}"
else
    EXP_NAME="JiT_B-sim-frozen-resnet101-swin-t-deit3-b-w${EXTRA_FD_WEIGHT}-125k-gbs${GLOBAL_BSZ}-${TOTAL_GPUS}gpu-seed${SEED}"
    WANDB_ARGS=(--disable_wandb)
    if [[ "$ENABLE_WANDB" == "1" ]]; then WANDB_ARGS=(--enable_wandb); fi
    echo "[W3-R101-SwinT-DeiT3] train: node=${NODE_RANK}/${NNODES}, GPUs=${TOTAL_GPUS}, batch/GPU=${BATCH_SIZE}, global batch=${GLOBAL_BSZ}, run=${EXP_NAME}"
    run_command "${LAUNCH[@]}" main_fd.py "${COMMON[@]}" "${SIM_R101_SWIN_DEIT3[@]}" \
        --exp_name "$EXP_NAME" --auto_resume "${WANDB_ARGS[@]}"
fi
