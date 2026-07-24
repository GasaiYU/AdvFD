#!/usr/bin/env bash
# Closed-form JiT-spectrum -> pMF transfer experiment.
#
# 1. Generate 50k images from the JiT-B Inception-FD checkpoint.
# 2. Extract one fixed 256x256 natural-spectrum RGB pattern at its original
#    pixel-space amplitude (no training and no RMS normalization).
# 3. Generate an independent 50k pMF-B image set.
# 4. Add the fixed pattern to every pMF image in pixel space.
# 5. Compute original and patched Inception FID from the two folders.
# 6. Export a deterministic 50-image patched subset under hacking/images.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

export HF_HOME="${HF_HOME:-/mmu-vcg/gaomingju/data/models/}"
export TORCH_HOME="${TORCH_HOME:-/mmu-vcg/gaomingju/data/models/}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# Model and generation configuration.
: "${JIT_CKPT:=/mmu-vcg/gaomingju/workspace/foundation/FD-Loss-Ours/work_dirs/table_3_JiT-B/JiT_B-fd-inception/checkpoints/step_0125000.pth}"
: "${PMF_CKPT:=./checkpoints/base/pMF-B_256.pth}"
: "${NUM_IMAGES:=50000}"
: "${GPUS_PER_NODE:=8}"
: "${GEN_BSZ_PER_GPU:=256}"
: "${FID_BSZ_PER_GPU:=128}"
: "${FID_NUM_WORKERS:=8}"
: "${APPLY_NUM_WORKERS:=16}"
: "${APPLY_LOG_EVERY:=1000}"
: "${MASTER_PORT:=29640}"

# Fixed, non-optimized spectrum extraction and application parameters.
: "${SPECTRAL_BATCH_SIZE:=64}"
: "${BLUR_SIGMA:=2.0}"
: "${SPECTRAL_SEED:=2026}"
: "${ALPHA:=1.0}"
: "${VIS_IMAGES:=50}"
: "${VIS_SEED:=2026}"

# Output layout.
: "${WORK_ROOT:=./work_dirs/jit_spectrum_to_pmf}"
: "${GEN_ROOT:=${WORK_ROOT}/generated}"
: "${VIS_ROOT:=./hacking/images}"
: "${OVERWRITE:=0}"

JIT_PROJECT="jit_spectrum_source"
JIT_CKPT_TAG="$(basename "$JIT_CKPT" .pth)"
JIT_EXP="JiT_B-fd-inception-${JIT_CKPT_TAG}"
PMF_PROJECT="pmf_spectrum_target"
PMF_CKPT_TAG="$(basename "$PMF_CKPT" .pth)"
PMF_EXP="pMF_B-256-${PMF_CKPT_TAG}"
: "${SPECTRUM_DIR:=${WORK_ROOT}/jit_spectral_pattern_preserve_scale/${JIT_CKPT_TAG}}"
: "${PATCHED_ROOT:=${WORK_ROOT}/pmf_patched/${PMF_CKPT_TAG}}"
: "${FID_ROOT:=${WORK_ROOT}/fid/${JIT_CKPT_TAG}_to_${PMF_CKPT_TAG}}"

JIT_IMAGE_DIR="${GEN_ROOT}/${JIT_PROJECT}/${JIT_EXP}/gen_images/ema=online-cfg=3.0-steps=1-interval_min=0.1-interval_max=1.0"
PMF_IMAGE_DIR="${GEN_ROOT}/${PMF_PROJECT}/${PMF_EXP}/gen_images/ema=online-cfg=8.5-steps=1-interval_min=0.1-interval_max=0.7"
PATTERN_PATH="${SPECTRUM_DIR}/spectral_pattern.npy"
ALPHA_LABEL="${ALPHA//./p}"
PATCHED_DIR="${PATCHED_ROOT}/alpha_${ALPHA_LABEL}"
VIS_DIR="${VIS_ROOT}/jit_${JIT_CKPT_TAG}_to_pmf_${PMF_CKPT_TAG}_alpha_${ALPHA_LABEL}"
BASELINE_FID_CSV="${FID_ROOT}/pmf_original.csv"
PATCHED_FID_CSV="${FID_ROOT}/pmf_spectral_alpha_${ALPHA_LABEL}.csv"

count_pngs() {
    local directory="$1"
    if [[ ! -d "$directory" ]]; then
        echo 0
        return
    fi
    find "$directory" -maxdepth 1 -type f -name '*.png' | wc -l | tr -d ' '
}

require_file() {
    local path="$1"
    if [[ ! -f "$path" ]]; then
        echo "[ERR] required file not found: $path" >&2
        exit 1
    fi
}

prepare_generated_folder() {
    local label="$1"
    local image_dir="$2"
    local checkpoint="$3"
    local preset="$4"
    local project="$5"
    local experiment="$6"
    local existing
    existing="$(count_pngs "$image_dir")"
    if [[ "$existing" -eq "$NUM_IMAGES" ]]; then
        echo "[reuse] ${label}: ${image_dir} (${existing} PNGs)"
        return
    fi
    if [[ "$existing" -ne 0 && "$OVERWRITE" != "1" ]]; then
        echo "[ERR] ${label} folder is partial: ${existing}/${NUM_IMAGES}" >&2
        echo "[ERR] set OVERWRITE=1 to regenerate the complete derived folder" >&2
        exit 1
    fi

    echo "[generate] ${label}: checkpoint=${checkpoint}"
    CKPT_PATH="$checkpoint" \
    PRESET="$preset" \
    GPUS_PER_NODE="$GPUS_PER_NODE" \
    NUM_IMAGES="$NUM_IMAGES" \
    EVAL_BSZ="$GEN_BSZ_PER_GPU" \
    MASTER_PORT="$MASTER_PORT" \
    RESULT_ROOT="$GEN_ROOT" \
    PROJECT="$project" \
    EXP_NAME="$experiment" \
    bash scripts/evaluate_released_ckpt.sh --gen_only

    existing="$(count_pngs "$image_dir")"
    if [[ "$existing" -ne "$NUM_IMAGES" ]]; then
        echo "[ERR] ${label} generation produced ${existing}/${NUM_IMAGES} PNGs" >&2
        exit 1
    fi
}

require_file "$JIT_CKPT"
require_file "$PMF_CKPT"
require_file "data/fid_stats/jit_in256_stats.npz"
require_file "data/fid_stats/guided_diffusion_stats.npz"

mkdir -p "$WORK_ROOT" "$FID_ROOT" "$VIS_ROOT"

# Stage 1: generate the exact JiT checkpoint source distribution.
prepare_generated_folder \
    "JiT spectrum source" \
    "$JIT_IMAGE_DIR" \
    "$JIT_CKPT" \
    "JiT_B" \
    "$JIT_PROJECT" \
    "$JIT_EXP"

# Stage 2: one-pass closed-form spectrum extraction.
if [[ -f "$PATTERN_PATH" ]]; then
    echo "[reuse] spectral pattern: $PATTERN_PATH"
else
    extraction_args=()
    if [[ "$OVERWRITE" == "1" ]]; then
        extraction_args+=(--overwrite)
    fi
    python hacking/extract_spectral_pattern.py \
        --image_dir "$JIT_IMAGE_DIR" \
        --num_images "$NUM_IMAGES" \
        --batch_size "$SPECTRAL_BATCH_SIZE" \
        --device cuda \
        --blur_sigma "$BLUR_SIGMA" \
        --seed "$SPECTRAL_SEED" \
        --preserve_spectral_scale \
        --output_dir "$SPECTRUM_DIR" \
        "${extraction_args[@]}"
fi
require_file "$PATTERN_PATH"

# Stage 3: generate the independent pMF target distribution.
prepare_generated_folder \
    "pMF target" \
    "$PMF_IMAGE_DIR" \
    "$PMF_CKPT" \
    "pMF_B_256" \
    "$PMF_PROJECT" \
    "$PMF_EXP"

# Stage 4: add the exact same full-resolution pattern to all 50k pMF images.
patched_count="$(count_pngs "$PATCHED_DIR")"
if [[ "$patched_count" -gt "$NUM_IMAGES" ]]; then
    echo "[ERR] patched folder contains too many PNGs: ${patched_count}/${NUM_IMAGES}" >&2
    exit 1
fi
if [[ "$patched_count" -eq "$NUM_IMAGES" ]]; then
    echo "[verify] complete patched folder before FID: $PATCHED_DIR"
fi
if [[ "$patched_count" -le "$NUM_IMAGES" ]]; then
    apply_args=()
    if [[ "$patched_count" -ne 0 ]]; then
        echo "[resume] patched folder: ${patched_count}/${NUM_IMAGES}"
        apply_args+=(--resume)
    elif [[ "$OVERWRITE" == "1" ]]; then
        apply_args+=(--overwrite)
    fi
    python hacking/apply_fourier_pattern.py \
        --input_dir "$PMF_IMAGE_DIR" \
        --output_dir "$PATCHED_DIR" \
        --pattern "$PATTERN_PATH" \
        --alpha "$ALPHA" \
        --alpha_space pixel \
        --preserve_pattern_scale \
        --num_workers "$APPLY_NUM_WORKERS" \
        --log_every "$APPLY_LOG_EVERY" \
        --output_format png \
        "${apply_args[@]}"
fi
patched_count="$(count_pngs "$PATCHED_DIR")"
if [[ "$patched_count" -ne "$NUM_IMAGES" ]]; then
    echo "[ERR] patched folder contains ${patched_count}/${NUM_IMAGES} PNGs" >&2
    exit 1
fi

# Stage 5: evaluate baseline and patched folders in one Inception load.
baseline_csv_exists=0
patched_csv_exists=0
[[ -f "$BASELINE_FID_CSV" ]] && baseline_csv_exists=1
[[ -f "$PATCHED_FID_CSV" ]] && patched_csv_exists=1
if [[ "$baseline_csv_exists" -ne "$patched_csv_exists" ]]; then
    if [[ "$OVERWRITE" != "1" ]]; then
        echo "[ERR] only one expected FID CSV exists under $FID_ROOT" >&2
        echo "[ERR] set OVERWRITE=1 to archive it and recompute both folders" >&2
        exit 1
    else
        timestamp="$(date +%Y%m%d_%H%M%S)"
        [[ -f "$BASELINE_FID_CSV" ]] && mv "$BASELINE_FID_CSV" "${BASELINE_FID_CSV}.${timestamp}.bak"
        [[ -f "$PATCHED_FID_CSV" ]] && mv "$PATCHED_FID_CSV" "${PATCHED_FID_CSV}.${timestamp}.bak"
    fi
elif [[ "$baseline_csv_exists" -eq 1 ]]; then
    echo "[reuse] FID CSVs already exist under $FID_ROOT"
fi
if [[ ! -f "$BASELINE_FID_CSV" || ! -f "$PATCHED_FID_CSV" ]]; then
    torchrun \
        --nproc_per_node="$GPUS_PER_NODE" \
        --master_port="$MASTER_PORT" \
        eval_all_fds.py \
        --image_folder "$PMF_IMAGE_DIR" "$PATCHED_DIR" \
        --models inception \
        --no_prc \
        --batch_size "$FID_BSZ_PER_GPU" \
        --num_workers "$FID_NUM_WORKERS" \
        --output_csv "$BASELINE_FID_CSV" "$PATCHED_FID_CSV"
fi

# Stage 6: deterministic 50-image qualitative subset.
visualization_args=()
if [[ "$OVERWRITE" == "1" ]]; then
    visualization_args+=(--overwrite)
fi
if [[ ! -f "${VIS_DIR}/apply_pattern_manifest.json" || "$OVERWRITE" == "1" ]]; then
    python hacking/apply_fourier_pattern.py \
        --input_dir "$PMF_IMAGE_DIR" \
        --output_dir "$VIS_DIR" \
        --pattern "$PATTERN_PATH" \
        --alpha "$ALPHA" \
        --alpha_space pixel \
        --preserve_pattern_scale \
        --num_workers "$APPLY_NUM_WORKERS" \
        --log_every "$APPLY_LOG_EVERY" \
        --output_format png \
        --random_sample "$VIS_IMAGES" \
        --sample_seed "$VIS_SEED" \
        "${visualization_args[@]}"
else
    echo "[reuse] 50-image visualization: $VIS_DIR"
fi

python -c '
import csv
import sys

def read_adm(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    matches = [float(row["fd"]) for row in rows if row["model"] == "FID(ADM)"]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one FID(ADM) row in {path}, found {len(matches)}")
    return matches[0]

baseline = read_adm(sys.argv[1])
patched = read_adm(sys.argv[2])
print(f"[result] pMF baseline FID(ADM): {baseline:.6f}")
print(f"[result] patched FID(ADM):      {patched:.6f}")
print(f"[result] delta:                 {patched - baseline:+.6f}")
' "$BASELINE_FID_CSV" "$PATCHED_FID_CSV"

echo
echo "[done] JiT source images:     $JIT_IMAGE_DIR"
echo "[done] extracted pattern:     $PATTERN_PATH"
echo "[done] pMF original images:   $PMF_IMAGE_DIR"
echo "[done] pMF patched images:    $PATCHED_DIR"
echo "[done] baseline FID CSV:      $BASELINE_FID_CSV"
echo "[done] patched FID CSV:       $PATCHED_FID_CSV"
echo "[done] 50 patched samples:    $VIS_DIR"
