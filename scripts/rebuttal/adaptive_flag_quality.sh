#!/usr/bin/env bash
# Do humans agree with what AdvFD's adaptive Inception flags?
#
# Rank the 50k JiT-L FD-loss samples (middle panel of jit_l_threeway) by the
# per-sample real-whitened distance d(x) = ||Lambda^-1/2 V^T (f(x) - mu_r)||^2
# in the adaptive Inception space of the JiT-L AdvFD checkpoint (real stats
# from 50 ImageNet train images per class, eps = --fd_adv_whiten_eps = 1e-3).
# Compare top / random / bottom 1% (500 images each) with MUSIQ, and 100
# images per group with a blind 1-5 human rating study (MOS).
# The pretrained Inception is ranked the same way as a MUSIQ-only control.
#
# STAGE=rank  bash scripts/rebuttal/adaptive_flag_quality.sh   # 1 GPU: scores.csv, groups.json
# STAGE=musiq bash scripts/rebuttal/adaptive_flag_quality.sh   # 1 GPU, needs `pip install pyiqa`
# STAGE=human bash scripts/rebuttal/adaptive_flag_quality.sh   # CPU: human_study/{images,rate.html}
# STAGE=mos RATINGS_DIR=path/to/ratings bash scripts/rebuttal/adaptive_flag_quality.sh
# STAGE=all   bash scripts/rebuttal/adaptive_flag_quality.sh   # rank + musiq + human
#
# Human study: send human_study/images/ + rate.html (not answer_key.csv) to
# each rater. Every rater rates all 300 images in their own random order and
# returns ratings_<id>.csv; put those files in RATINGS_DIR and run STAGE=mos.

set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export HF_HOME="${HF_HOME:-/mmu-vcg/gaomingju/data/models/}"
export TORCH_HOME="${TORCH_HOME:-/mmu-vcg/gaomingju/data/models/}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
: "${STAGE:=all}"
: "${SAMPLES_DIR:=/mmu-vcg/gaomingju/data/FD-Loss/quali/jit_l_threeway}"
: "${PANEL:=1}"
: "${ADV_CKPT:=./work_dirs/Jit-B-adv/JiT_L-fd-sim-advinc-w0.1-from-base/checkpoints/step_0124999.pth}"
: "${DATA_ROOT:=/mmu-vcg/zhangxu34/datasets/ImageNet-1K/}"
: "${OUTPUT_DIR:=./work_dirs/rebuttal_flag_quality/jit_l}"
: "${REAL_PER_CLASS:=50}"
: "${WHITEN_EPS:=1e-3}"
: "${FRACTION:=0.01}"
: "${HUMAN_PER_GROUP:=100}"
: "${MIN_RATERS:=5}"
: "${SEED:=0}"
: "${BATCH_SIZE:=100}"
: "${NUM_WORKERS:=8}"
: "${OVERWRITE:=0}"
: "${PYTHON:=python}"

run() {
    local stage="$1"; shift
    local cmd=("$PYTHON" scripts/rebuttal/adaptive_flag_quality.py "$stage" --out_dir "$OUTPUT_DIR"
        --seed "$SEED" --batch_size "$BATCH_SIZE" --num_workers "$NUM_WORKERS" "$@")
    printf '[flag-quality] '; printf '%q ' "${cmd[@]}"; printf '\n'
    "${cmd[@]}"
}

stage_rank() {
    run rank --samples_dir "$SAMPLES_DIR" --panel "$PANEL" --checkpoint "$ADV_CKPT" \
        --data_root "$DATA_ROOT" --real_per_class "$REAL_PER_CLASS" \
        --whiten_eps "$WHITEN_EPS" --fraction "$FRACTION"
}
stage_musiq() { run musiq; }
stage_human() {
    local extra=()
    if [[ "$OVERWRITE" == "1" ]]; then extra=(--overwrite); fi
    run human --human_per_group "$HUMAN_PER_GROUP" ${extra[@]+"${extra[@]}"}
}
stage_mos() {
    if [[ -z "${RATINGS_DIR:-}" ]]; then
        echo "[ERR] STAGE=mos needs RATINGS_DIR=<folder of ratings_*.csv>" >&2; exit 1
    fi
    run mos --ratings_dir "$RATINGS_DIR" --min_raters "$MIN_RATERS"
}

case "$STAGE" in
    rank) stage_rank ;;
    musiq) stage_musiq ;;
    human) stage_human ;;
    mos) stage_mos ;;
    all) stage_rank; stage_musiq; stage_human ;;
    *) echo "[ERR] STAGE must be rank | musiq | human | mos | all" >&2; exit 1 ;;
esac
