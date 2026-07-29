#!/usr/bin/env bash
# Sweep all step checkpoints and compute the adversarial/original FD repr norm ratio.
#
# Outputs:
#   $RESULT_DIR/summary.jsonl
#   $RESULT_DIR/<checkpoint_stem>.json
#
# Required only when defaults do not fit your environment:
#   CKPT_DIR   directory containing step_*.pth checkpoints
#   DATA_ROOT  ImageNet root with train/ or val/
#
# Optional overrides:
#   GPUS_PER_NODE=8
#   SPLIT=train|val
#   NUM_IMAGES=10000
#   BATCH_SIZE=256
#   NUM_WORKERS=8
#   REPR_MODEL=inception
#   POOL_TYPE=cls|avg
#   RESULT_DIR=path/to/output_dir

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

export HF_HOME="${HF_HOME:-/mmu-vcg/gaomingju/data/models/}"
export TORCH_HOME="${TORCH_HOME:-/mmu-vcg/gaomingju/data/models/}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

: "${CKPT_DIR:=/mmu-vcg/gaomingju/workspace/foundation/FD-Loss-Ours/work_dirs/Jit-B-adv/JiT_B-fd-inception-advfd-repr-w0.1-pre1000x2000-adv1000-from-base/checkpoints}"
: "${DATA_ROOT:=/mmu-vcg/zhangxu34/datasets/ImageNet-1K}"
: "${GPUS_PER_NODE:=8}"
: "${SPLIT:=train}"
: "${NUM_IMAGES:=10000}"
: "${BATCH_SIZE:=256}"
: "${NUM_WORKERS:=8}"
: "${REPR_MODEL:=inception}"
: "${POOL_TYPE:=cls}"
: "${RESULT_DIR:=$REPO_ROOT/my_tools/fd_adv_norm_ratio_all_no_whiten}"

shopt -s nullglob
CKPTS=("$CKPT_DIR"/step_*.pth)
shopt -u nullglob

if [ "${#CKPTS[@]}" -eq 0 ]; then
    echo "[ERR] no step_*.pth checkpoints found in: $CKPT_DIR" >&2
    exit 1
fi

mkdir -p "$RESULT_DIR"
SUMMARY_JSONL="$RESULT_DIR/summary.jsonl"
: > "$SUMMARY_JSONL"

FAILED=0
for ckpt in "${CKPTS[@]}"; do
    stem="$(basename "$ckpt" .pth)"
    out_json="$RESULT_DIR/${stem}.json"
    echo "[batch] testing $stem"

    if result_json="$( \
        CKPT_DIR="$ckpt" \
        DATA_ROOT="$DATA_ROOT" \
        SPLIT="$SPLIT" \
        NUM_IMAGES="$NUM_IMAGES" \
        BATCH_SIZE="$BATCH_SIZE" \
        NUM_WORKERS="$NUM_WORKERS" \
        REPR_MODEL="$REPR_MODEL" \
        POOL_TYPE="$POOL_TYPE" \
        OUTPUT_JSON="$out_json" \
        GPUS_PER_NODE="$GPUS_PER_NODE" \
        bash "$SCRIPT_DIR/compute_fd_adv_norm_ratio.sh" \
    )"; then
        printf '%s\n' "$result_json" >> "$SUMMARY_JSONL"
        short_line="$(python -c 'import json,sys; d=json.loads(sys.argv[1]); print("{}\t{:.6f}\t{:.6f}".format(d["checkpoint_path"], d["rms_ratio"], d["second_moment_ratio"]))' "$result_json")"
        echo "[ok] $short_line"
    else
        echo "[ERR] failed on $ckpt" >&2
        FAILED=1
    fi
done

echo "[done] wrote $SUMMARY_JSONL"
if [ "$FAILED" -ne 0 ]; then
    exit 1
fi
