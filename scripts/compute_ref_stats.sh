#!/usr/bin/env bash
# Compute the ImageNet reference statistics used by the paper experiments.
# This only regenerates statistics that come directly from ImageNet images.

set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to the ImageNet root with train/ and val/ subdirectories}"
: "${GPUS_PER_NODE:=8}"
: "${MASTER_PORT:=29500}"
: "${IMG_SIZE:=256}"

run_stats() {
    local model="$1"
    local output_name="$2"
    local repr_input_size="${3:-256}"

    torchrun --nproc_per_node="$GPUS_PER_NODE" --master_port="$MASTER_PORT" \
        compute_repr_stats.py \
        --model "$model" \
        --data_path "$DATA_ROOT" \
        --img_size "$IMG_SIZE" \
        --target_size "$repr_input_size" \
        --output_name "$output_name"
}

run_stats convnext convnext_in256_t224_stats.npz 224
run_stats vit_large_patch14_dinov2.lvd142m vit_large_patch14_dinov2_lvd142m_in256_t256_stats.npz
run_stats vit_large_patch14_clip_224.openai vit_large_patch14_clip_224_openai_in256_t256_stats.npz
run_stats vit_large_patch16_224.mae vit_large_patch16_224_mae_in256_t224_stats.npz 224
run_stats vit_so400m_patch16_siglip_256.v2_webli vit_so400m_patch16_siglip_256_v2_webli_in256_t224_stats.npz 224
