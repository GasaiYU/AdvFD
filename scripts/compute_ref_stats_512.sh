#!/usr/bin/env bash
# Compute ImageNet reference statistics with 512x512 center-cropped inputs.
# The output filenames use "in512" to avoid overwriting the paper's 256 stats.

set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to the ImageNet root with train/ and val/ subdirectories}"
: "${GPUS_PER_NODE:=8}"
: "${MASTER_PORT:=29500}"
: "${IMG_SIZE:=512}"

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

run_stats convnextv2_base.fcmae_ft_in22k_in1k convnext_in512_t448_stats.npz 448
run_stats vit_large_patch14_dinov2.lvd142m vit_large_patch14_dinov2_lvd142m_in512_t448_stats.npz 448
run_stats vit_large_patch14_clip_224.openai vit_large_patch14_clip_224_openai_in512_t448_stats.npz 448
run_stats vit_large_patch16_224.mae vit_large_patch16_224_mae_in512_t448_stats.npz 448
run_stats vit_so400m_patch16_siglip_256.v2_webli vit_so400m_patch16_siglip_256_v2_webli_in512_t448_stats.npz 448
