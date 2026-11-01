#!/usr/bin/env bash
# Run the width-72 and dual-publication Loom cases with their required layouts.
# Pin a supported GPU with ROCR_VISIBLE_DEVICES before invoking this script.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source_root="$(cd -- "$script_dir/../../../.." && pwd)"
loom_build="$source_root/build/ggml/src/ggml-hrx/hrx/src/ggml-hrx-deps-build/loom/src/loom/tools"
loom_link="$loom_build/loom-link/loom-link"
loom_benchmark="$loom_build/loom-benchmark/loom-benchmark"
corpus="$source_root/ggml/src/ggml-hrx/kernel-corpus/kernels/loom-libs"
output="${1:-$source_root/build/vision-flash-attention-cases}"
mkdir -p "$output"

"$loom_link" --mode=merge --allow-unresolved --output="$output/linked.loom" \
  "$corpus/ops/flash_attention/prefill_f32_f16_wmma.loom" \
  "$corpus/ops/flash_attention/prefill_f32_f16_wmma_publish_f16.loom" \
  "$corpus/motifs/flash_attention/qk_block_wmma.loom" \
  "$corpus/motifs/flash_attention/pv_block_wmma.loom" \
  "$corpus/motifs/flash_attention/online_softmax.loom" \
  "$corpus/motifs/flash_attention/publish_f32.loom"

common=(
  --device=amdgpu
  --scenario-correctness=true
  --config=ggml.flash_attention.query_head_count=1
  --config=ggml.flash_attention.key_value_head_count=1
  --config=ggml.flash_attention.attention_scale=1.0
  --config=ggml.flash_attention.gate_stride_head=128
  --config=ggml.flash_attention.gate_stride_token=128
)
tiled=(
  --config=ggml.flash_attention.qk_head_size=80
  --config=ggml.flash_attention.value_head_size=128
  --config=ggml.flash_attention.logical_qk_head_size=72
  --config=ggml.flash_attention.logical_value_head_size=72
  --config=ggml.flash_attention.query_token_stride=80
  --config=ggml.flash_attention.query_head_stride=80
  --config=ggml.flash_attention.key_token_stride=80
  --config=ggml.flash_attention.key_head_stride=80
  --config=ggml.flash_attention.value_token_stride=80
  --config=ggml.flash_attention.value_head_stride=80
  --config=ggml.flash_attention.output_token_stride=72
  --config=ggml.flash_attention.output_head_stride=72
  --config=ggml.flash_attention.tiled_loaders=1
  --config=ggml.flash_attention.apply_gate=0
)

"$loom_benchmark" "$output/linked.loom" \
  --benchmark=@ggml_flash_attention_f32_f16_wmma_tiled_head72 \
  "${common[@]}" "${tiled[@]}" \
  --config=ggml.flash_attention.has_mask=1 \
  --config=ggml.flash_attention.mask_token_stride=70 \
  --config=ggml.flash_attention.test.context_capacity=70 \
  --output="$output/masked.json"

"$loom_benchmark" "$output/linked.loom" \
  --benchmark=@ggml_flash_attention_f32_f16_wmma_tiled_unmasked_tail \
  "${common[@]}" "${tiled[@]}" \
  --config=ggml.flash_attention.has_mask=0 \
  --config=ggml.flash_attention.test.context_capacity=65 \
  --output="$output/unmasked.json"

"$loom_benchmark" "$output/linked.loom" \
  --benchmark=@ggml_flash_attention_f32_f16_wmma_gated_dual_publish \
  "${common[@]}" \
  --config=ggml.flash_attention.qk_head_size=128 \
  --config=ggml.flash_attention.value_head_size=128 \
  --config=ggml.flash_attention.apply_gate=1 \
  --config=ggml.flash_attention.test.context_capacity=64 \
  --output="$output/gated.json"
