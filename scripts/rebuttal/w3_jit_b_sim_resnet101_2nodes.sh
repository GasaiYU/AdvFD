#!/usr/bin/env bash
# W3 quality control: JiT-B + SIM-FD + frozen ResNet-101.
# Two machines, eight GPUs each. Global batch=1024 => batch=64/GPU.
# All four pretrained representation models stay frozen/eval;
# gradients through their generated-image inputs train JiT-B only.
# The four normalized FD weights are all 1.0 by default. No AdvFD branch.
# Same base checkpoint, LR, 125k steps and warmup as the W3 8-GPU setup.
# ResNet-101 reference statistics must exist on BOTH nodes before launch:
#   GPUS_PER_NODE=8 MODE=stats bash scripts/rebuttal/w3_jit_b_sim_resnet50.sh
# W&B is online, with one run created by global rank 0 only.
#
# Default master (node 0): 10.82.122.84; override MASTER_ADDR if needed.
# Node 0: NODE_RANK=0 bash scripts/rebuttal/w3_jit_b_sim_resnet101_2nodes.sh
# Node 1: NODE_RANK=1 bash scripts/rebuttal/w3_jit_b_sim_resnet101_2nodes.sh
# Preview: DRY_RUN=1 NODE_RANK=0 bash scripts/rebuttal/w3_jit_b_sim_resnet101_2nodes.sh
# Run `wandb login` on node 0 first if that account is not already logged in.
# Both nodes need the same code/config, base checkpoint, pretrained weights,
# reference statistics, and access to the same shared OUTPUT_DIR for resume.

set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

case "${NODE_RANK:-}" in
    0|1) ;;
    *) echo "[ERR] Set NODE_RANK=0 on the first machine, NODE_RANK=1 on the second" >&2; exit 1 ;;
esac
: "${MASTER_ADDR:=10.82.122.84}"
case "$MASTER_ADDR" in
    localhost|127.*|::1|0.0.0.0)
        echo "[ERR] MASTER_ADDR must be reachable from the second machine, not a loopback/bind address" >&2; exit 1 ;;
esac
: "${MASTER_PORT:=29500}"
: "${DATA_ROOT:=/mmu-vcg/zhangxu34/datasets/ImageNet-1K/}"
: "${CKPT_ROOT:=./checkpoints/base}"
: "${OUTPUT_DIR:=./work_dirs}"
: "${GLOBAL_BSZ:=1024}"
: "${SEED:=1}"
: "${EXTRA_FD_WEIGHT:=1.0}"
: "${PROJECT:=rebuttal_w3}"
: "${EXP_NAME:=JiT_B-sim-frozen-resnet101-w${EXTRA_FD_WEIGHT}-125k-gbs${GLOBAL_BSZ}-16gpu-seed${SEED}}"
: "${DRY_RUN:=0}"

case "$GLOBAL_BSZ" in
    ''|*[!0-9]*) echo "[ERR] GLOBAL_BSZ must be a positive integer divisible by 16" >&2; exit 1 ;;
esac
if (( GLOBAL_BSZ < 1 || GLOBAL_BSZ % 16 != 0 )); then
    echo "[ERR] GLOBAL_BSZ must be positive and divisible by 16" >&2; exit 1
fi
BATCH_SIZE=$(( GLOBAL_BSZ / 16 ))

export HF_HOME="${HF_HOME:-/mmu-vcg/gaomingju/data/models/}"
export TORCH_HOME="${TORCH_HOME:-/mmu-vcg/gaomingju/data/models/}"
export TIMM_USE_OLD_CACHE="${TIMM_USE_OLD_CACHE:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export WANDB_MODE=online

RESNET101=resnet101.a1_in1k
RESNET101_STATS=data/fid_stats/resnet101_a1_in1k_in256_t224_stats.npz

if [[ "$DRY_RUN" != "1" ]]; then
    for REQUIRED in "${CKPT_ROOT}/JiT-B.pth" "$RESNET101_STATS"; do
        if [[ ! -f "$REQUIRED" ]]; then
            echo "[ERR] Missing on node ${NODE_RANK}: $REQUIRED" >&2
            echo "[ERR] Prepare stats first: GPUS_PER_NODE=8 MODE=stats bash scripts/rebuttal/w3_jit_b_sim_resnet50.sh" >&2
            exit 1
        fi
    done
    if [[ ! -d "${DATA_ROOT}/train" ]]; then
        echo "[ERR] DATA_ROOT must contain train/: $DATA_ROOT" >&2; exit 1
    fi
fi

COMMAND=(torchrun --nnodes=2 --nproc_per_node=8 --node_rank="$NODE_RANK"
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT"
    main_fd.py --project "$PROJECT" --exp_name "$EXP_NAME" --output_dir "$OUTPUT_DIR"
    --data_path "$DATA_ROOT" --load_from "${CKPT_ROOT}/JiT-B.pth"
    --model JiT_B --rope_2d --learned_pe --legacy_time_convention
    --cfg 3.0 --interval_min 0.1 --interval_max 1.0 --ema_type edm --num_sampling_steps 1
    --batch_size "$BATCH_SIZE" --seed "$SEED"
    --epochs 100 --steps_per_epoch 1250 --warmup_epochs 5
    --lr 1e-5 --lr_sched cosine --min_lr 0.0
    --grad_checkpointing --fd_repr_grad_checkpoint_models siglip
    --fd_eigvalsh --fd_ema_beta 0.999 --fd_whiten_eps 1e-3
    --fd_repr_models vit_so400m_patch16_siglip_256.v2_webli vit_large_patch16_224.mae inception
        "$RESNET101"
    --fd_repr_pool_types cls cls cls cls --fd_target_sizes 224 224 256 224
    --fd_repr_weights 1 1 1 "$EXTRA_FD_WEIGHT" --fd_adv_weight 0.0
    --eval_bsz 256 --num_images_for_eval_and_search 50000
    --vis_freq 100 --online_eval --eval_freq 10000 --print_freq 20
    --milestone_interval 10 --save_freq 5 --auto_resume --enable_wandb)

printf '[W3-R101-16GPU] node=%s/2, GPUs=16, batch/GPU=%s, global batch=%s, steps=125000\n' "$NODE_RANK" "$BATCH_SIZE" "$GLOBAL_BSZ"
printf '[W3-R101-16GPU] W&B project=%s, run=%s (global rank 0 only)\n' "$PROJECT" "$EXP_NAME"
printf '[W3-R101-16GPU] Command: '
printf '%q ' "${COMMAND[@]}"
printf '\n'
if [[ "$DRY_RUN" != "1" ]]; then exec "${COMMAND[@]}"; fi
