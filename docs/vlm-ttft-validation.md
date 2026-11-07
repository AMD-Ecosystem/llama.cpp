# VLM time-to-first-token prototypes: validation guide

This branch adds gfx11 (RDNA3/3.5) prototypes that cut llama.cpp VLM time to first token (TTFT).
Every new behavior is behind an environment variable that defaults OFF, so an unmodified run of
this branch behaves like the base `gfx11` branch. This guide lists the exact settings used for
the measured results so they can be reproduced.

## Build

Same flags as a normal gfx1151 HIP build:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGPU_TARGETS=gfx1151 -DGGML_HIP=ON -DGGML_OPENMP=OFF -DGGML_RPC=ON \
  -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_TESTS=ON
cmake --build build -j
```

Measured with a ROCm 10.1 nightly toolchain (clang 23).

## Unit tests

Run with all gates ON and again with all gates OFF; both must pass:

```bash
export GGML_CUDA_MMDQ=1 GGML_CUDA_MMDQ_MOE=1 GGML_CUDA_MMDQ_Q8_0=1 GGML_CUDA_MMDQ_GLU=1 \
       GGML_CUDA_FATTN_VISION=1 GGML_CUDA_FATTN_WMMA_NO_GQA=1 GGML_CUDA_FATTN_WMMA_D72=1 \
       GGML_CUDA_FUSE_NORM_MUL_ADD=1 GGML_CUDA_ROPE_MULTI_BLOCK=64 GGML_CUDA_MOE_REDUCE_X4=1
for op in MUL_MAT MUL_MAT_ID FLASH_ATTN_EXT NORM NORM_MUL_ADD ROPE FFN_SWIGLU MOE_FFN_SWIGLU MMDQ_ADD MOE_REDUCE; do
  ./build/bin/test-backend-ops test -o $op
done
```

## Server settings

All runs use llama-server with:

```bash
llama-server -m <model.gguf> --mmproj <mmproj.gguf> -ngl 999 -fa 1 -c 8192 \
  -b 2048 -ub <ubatch> --no-cache-prompt --ctx-checkpoints 0
```

- `--no-cache-prompt`: every request is a cold prefill (the image is re-encoded each time).
- `--ctx-checkpoints 0`: context checkpoints cannot be reused without prompt caching but cost a
  device-to-host copy per request.
- `ROCBLAS_USE_HIPBLASLT=1` in the environment.

### Gates common to every model

```bash
GGML_CUDA_MMDQ=1
GGML_CUDA_FATTN_VISION=1 GGML_CUDA_FATTN_WMMA_NO_GQA=1
GGML_CUDA_FUSE_CUBLAS_BIAS=1 GGML_CUDA_FUSE_CUBLAS_FFN_ACT=1 GGML_CUDA_FUSE_NORM_MUL_ADD=1
GGML_CUDA_ROPE_VISION_BLOCK=64
LLAMA_SERVER_MTMD_MIXED=1 LLAMA_MIXED_EMBD_BIDIR=1
MTMD_IMAGE_THREADS=8 MTMD_VISION_FUSE_QKV=1 MTMD_VISION_KV_CAST_EARLY=1
```

### Per-model additions

| Model | `-ub` | Additional gates | Request notes |
|---|---:|---|---|
| Gemma-3-4B-IT | 512 | `GGML_CUDA_FATTN_WMMA_D72=1` | `enable_thinking: false` |
| Gemma-3-12B-IT | 512 | `GGML_CUDA_FATTN_WMMA_D72=1` | `enable_thinking: false` |
| Gemma-4-26B-A4B-IT | 1024 | `GGML_CUDA_FATTN_WMMA_D72=1 GGML_CUDA_MMDQ_MOE=1 GGML_CUDA_MMDQ_Q8_0=1 LLAMA_ARG_IMAGE_MAX_TOKENS=280` | `enable_thinking: false` |
| Gemma-4-31B-IT | 512 | `GGML_CUDA_FATTN_WMMA_D72=1 LLAMA_ARG_IMAGE_MAX_TOKENS=280` | `enable_thinking: false` |
| Gemma-4-31B-IT (OpenNav) | 512 | `GGML_CUDA_FATTN_WMMA_D72=1 LLAMA_ARG_IMAGE_MAX_TOKENS=280` | `enable_thinking: false`, `ignore_eos: true` |
| MiniCPM-o-2_6 | 512 | `GGML_CUDA_FATTN_WMMA_D72=1` | `enable_thinking: false` |
| Qwen2.5-VL-3B-Instruct | 2048 | `MTMD_QWEN25VL_WINDOW_BATCH=1 MTMD_QWEN25VL_WINDOW_SPLIT=1 MTMD_QWEN25VL_FUSE_QKV_FFN=1 MTMD_QWEN25VL_KV_CAST_EARLY=1 GGML_CUDA_FUSE_CUBLAS_FFN=1 GGML_CUDA_MMDQ_GLU=1 GGML_CUDA_ROPE_MULTI_BLOCK=64` | `enable_thinking: false` |
| Qwen2.5-VL-7B-Instruct | 1024 | same as Qwen2.5-VL-3B | `enable_thinking: false` |
| Qwen3-VL-4B-Instruct | 512 | none | `enable_thinking: false` |
| Qwen3.5-35B-A3B | 2048 | `GGML_CUDA_FATTN_WMMA_D72=1 GGML_CUDA_MMDQ_MOE=1 GGML_CUDA_MMDQ_Q8_0=1` | `enable_thinking: false` |
| Qwen3-Omni-30B-A3B-Instruct | 2048 | `GGML_CUDA_FATTN_WMMA_D72=1 GGML_CUDA_MMDQ_MOE=1 GGML_CUDA_MMDQ_Q8_0=1 GGML_CUDA_MMDQ_GLU=1 GGML_CUDA_ROPE_MULTI_BLOCK=64 GGML_CUDA_MOE_REDUCE_X4=1` | do NOT send `chat_template_kwargs` (see below), `ignore_eos: true` |

Notes:

- Leave `GGML_CUDA_MMDQ_FUSE_ADD` OFF: it is correct but measured slower.
- `LLAMA_ARG_IMAGE_MAX_TOKENS=280` matches the Hugging Face processor (and vLLM) image budget for
  Gemma-4; llama.cpp's default is 1120, which makes llama.cpp encode about 42% more vision
  patches on a 1024x800 image.
- Qwen3-Omni is not a thinking model. Sending `enable_thinking: false` makes the template append an
  empty think block, and with a long repetitive prompt the model then emits only newlines (vLLM
  does the same). Omit `chat_template_kwargs` for this model on both backends.
- `GGML_CUDA_FATTN_WMMA_NO_GQA=1` is listed in the common set, but it is only kept where its
  accuracy was validated (Qwen2.5-VL, head dim 80; and head dim 64 with f32 accumulation). It
  does not fire for D=72 encoders.

## Workload

- One deterministic 1024x800 RGB PNG: `R = x*255//1023`, `G = y*255//799`,
  `B = ((x//64 + y//64) % 2) * 255` (a gradient with a 64-pixel checkerboard in blue).
- Prompt: the image first, then the text `Describe this image in detail. ` followed by
  `Discuss the colors, shapes, spatial relationships, visible patterns, and image composition. `
  repeated 35 times (trailing space trimmed).
- Streaming `/v1/chat/completions`, temperature 0, 32 output tokens.
- 5 warmup requests, then 30 measured requests, sequential, model resident.
- TTFT = request sent to the first streamed chunk with non-empty `content` (role-only and empty
  chunks are ignored). Report the median of the 30 requests.
- Compare against vLLM on the same board, same image and prompt, with prefix caching and the
  multimodal processor cache disabled. Board-to-board differences of 5-8% were seen, so always
  pair llama.cpp and vLLM runs on one board and alternate their order.

## Expected results (gfx1151, Strix Halo)

| Model | llama.cpp (ms) | vLLM (ms) | Ratio |
|---|---:|---:|---:|
| Gemma-3-4B-IT | 527.8 | 489.8 | 1.08x |
| Gemma-3-12B-IT | 1,051.9 | 1,085.9 | 0.97x |
| Gemma-4-26B-A4B-IT | 595.9 | 669.4 | 0.89x |
| Gemma-4-31B-IT | 2,292.4 | 2,151.1 | 1.07x |
| Gemma-4-31B-IT (OpenNav) | 2,299.6 | 2,151.7 | 1.07x |
| MiniCPM-o-2_6 | 913.2 | 1,111.1 | 0.82x |
| Qwen2.5-VL-3B-Instruct | 675.5 | 711.3 | 0.95x |
| Qwen2.5-VL-7B-Instruct | 1,070.9 | 984.5 | 1.09x |
| Qwen3-Omni-30B-A3B-Instruct | 735.1 | 729.9 | 1.01x |
| Qwen3-VL-4B-Instruct | 715.2 | 702.4 | 1.02x |
| Qwen3.5-35B-A3B | 774.8 | 733.1 | 1.06x |

The Qwen3-Omni number above was measured with `GGML_CUDA_MMDQ_FUSE_ADD=1` also set; the table's
recommended Omni setting (without it) is expected to be equal or faster.

## Accuracy checks

- Vision encoder: compare embeddings (gates ON vs OFF) against an fp32 CPU reference
  (`-fa off`, CPU backend). Gated runs should be at or below the gates-OFF GPU error.
- LLM: KL divergence vs an fp16 dequant + hipBLAS reference on wikitext should not increase.
- Greedy text over 32 tokens should match gates OFF, apart from near-tie tokens that a `-ub`
  change alone also flips.

## Long-prompt text LLMs

The long-prompt text LLM prototypes (causal prefill attention, more GEMM formats, server flags) and
their validation steps are in `docs/llm-long-prompt-ttft-validation.md`.
