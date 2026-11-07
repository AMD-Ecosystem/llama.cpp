# Long-prompt LLM time-to-first-token prototypes: validation guide

This branch adds gfx11 (RDNA3/3.5) prototypes that cut llama.cpp time to first token (TTFT) for text
LLMs with long prompts (2K-8K tokens). They build on the VLM work in `docs/vlm-ttft-validation.md`
(same branch). Every new behavior is behind an environment variable that defaults OFF, so an
unmodified run of this branch behaves like the base `gfx11` branch.

## Build

Same flags as a normal gfx1151 HIP build:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGPU_TARGETS=gfx1151 -DGGML_HIP=ON -DGGML_OPENMP=OFF -DGGML_RPC=ON \
  -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_TESTS=ON
cmake --build build -j
```

Measured with a ROCm 10.1 nightly toolchain (clang 23).

## Gates

Final gate set used for every row (`fin`):

```bash
GGML_CUDA_MMDQ=1 GGML_CUDA_MMDQ_GLU=1 GGML_CUDA_ROPE_MULTI_BLOCK=64 GGML_CUDA_MMDQ_Q8_0=1
GGML_CUDA_FATTN_PREFILL=1 GGML_CUDA_FATTN_PREFILL_D256_SXH=1
GGML_CUDA_MMDQ_SHARE_SRC1=1 GGML_CUDA_MMDQ_Q5_K=1 GGML_CUDA_MMDQ_Q5_0=1 GGML_CUDA_MMDQ_GEGLU=1
GGML_CUDA_MMDQ_SWIZZLE=2 GGML_CUDA_NORM_ROPE_WARP=1 GGML_CUDA_ROPE_BLOCK=64
LLAMA_SPM_FAST_REV_MERGE=1
ROCBLAS_USE_HIPBLASLT=1
```

What the new ones do:

- `GGML_CUDA_FATTN_PREFILL=1`: prefill flash attention for causal masks and GQA, on the RDNA3 WMMA
  kernel of `GGML_CUDA_FATTN_VISION` (`fattn-vision.cu`). Head dim 64/128 (16 queries per wave) and
  256 (two waves per 16 queries, `GGML_CUDA_FATTN_PREFILL_D256_NG` query groups per block, default
  8). Mask tiles are classified once per graph (all 0 / all -inf / mixed); all -inf tiles are
  skipped. Needs n_kv % 16 == 0 and at least 64 queries.
- `GGML_CUDA_FATTN_PREFILL_D256_SXH=1`: the D=256 path exchanges partial scores in f16 (one barrier
  less per tile).
- `GGML_CUDA_MMDQ_Q5_K=1`, `GGML_CUDA_MMDQ_Q5_0=1`: dense Q5_K / Q5_0 weights in the fp16 dequant GEMM
  (Q5_0 also in the gate/up/SwiGLU fusion). Qwen2.5-0.5B "Q4_K_M" is mostly Q5_0.
- `GGML_CUDA_MMDQ_GEGLU=1`: GEGLU (Gemma) in the gate/up fusion.
- `GGML_CUDA_MMDQ_SWIZZLE=2`: process token tiles in groups of 2 when activations exceed 8 MiB, so
  large activations stay in cache (long prompts with large ubatch).
- `GGML_CUDA_MMDQ_SHARE_SRC1=1`: one f16 copy of the attention input shared by the Q/K/V GEMMs.
- `GGML_CUDA_MMDQ_FUSE_BIAS=1`: bias add in the GEMM epilogue (no residual). Used on Qwen2.5 rows.
- `GGML_CUDA_ROPE_BLOCK=32|64`, `GGML_CUDA_NORM_ROPE_WARP=1`: smaller rope launches, and one wave per
  row in the fused q/k norm + rope (Qwen3, Qwen3.6, Gemma-3).
- `LLAMA_SPM_FAST_REV_MERGE=1`: SPM tokenizer merge history in a hash map. Same tokens; an
  8,000-token Gemma prompt tokenizes in about 16 ms instead of 30.
- Leave `GGML_CUDA_MMDQ_FUSE_ADD` OFF (measured slower).

## Server settings

```bash
llama-server -m <model.gguf> -ngl 999 -t 8 -fa on -c <max_model_len+256> \
  -b <max(ub,2048)> -ub <ubatch> -np 1 -ctk f16 -ctv f16 \
  --no-cache-prompt -cram 0 --ctx-checkpoints 0
```

- `--no-cache-prompt`, `-cram 0`, `--ctx-checkpoints 0`: no prefix reuse, matching vLLM run without
  prefix caching. Without `-cram 0` the server copies the whole slot KV cache to host memory on every
  new request (38 ms on Qwen2.5-3B, about 300 ms on Llama-2-7B, 356 ms on Gemma-3-12B); without
  `--ctx-checkpoints 0` it snapshots SWA / recurrent state per request. Neither is ever reused here.

## Per-row settings

| Row | GGUF | Prompt tokens | `-ub` | Extra gates |
|---|---|---:|---:|---|
| Qwen2.5-0.5B-Instruct | bartowski Qwen2.5-0.5B-Instruct-Q4_K_M | 3968 | 2048 | `GGML_CUDA_ROPE_BLOCK=32 GGML_CUDA_MMDQ_FUSE_BIAS=1` |
| Qwen2.5-3B-Instruct | bartowski Qwen2.5-3B-Instruct-Q4_K_M | 3968 | 2048 | `GGML_CUDA_MMDQ_FUSE_BIAS=1` |
| Qwen2.5-7B-Instruct | bartowski Qwen2.5-7B-Instruct-Q4_K_M | 3968 | 1024 | `GGML_CUDA_MMDQ_FUSE_BIAS=1` |
| Qwen3-1.7B | unsloth Qwen3-1.7B-Q4_K_M | 3968 | 2048 | none |
| Qwen3-4B | unsloth Qwen3-4B-Q4_K_M | 3968 | 1024 | none |
| Qwen3-8B | unsloth Qwen3-8B-Q4_K_M | 3968 | 1024 | none |
| Llama-2-7B | TheBloke llama-2-7b.Q4_K_M | 1920 | 1024 | none |
| Gemma-2B | QuantFactory gemma-2b-it.Q4_K_M | 8000 | 1024 | none |
| Qwen3.6-27B | unsloth Qwen3.6-27B-Q4_K_M | 4096 | 1024 | none |
| Gemma-3-12B-IT | unsloth gemma-3-12b-it-Q4_K_M | 4096 | 1024 | none |

## Workload and measurement

- Both backends driven by the same client, `vllm bench serve`:
  `--backend openai --random-input-len <tokens> --num-prompts 30 --num-warmups 5 --max-concurrency 1
  --temperature 0 --disable-shuffle --percentile-metrics ttft,tpot,e2el`, random dataset with seed 0,
  `/v1/completions`, `--tokenizer` = the vLLM model's HF tokenizer. Report median TTFT.
- vLLM side: the nightly vLLM build (0.30.1rc1.dev1736, ROCm 10.3 nightly 20261105) with the nightly
  row arguments (prefix caching off; Gemma-3-12B uses `--attention-backend ROCM_RDNA_ATTN`).
- Confirm the server-reported prompt token count is identical on both sides.
- Pair llama.cpp and vLLM on the same board and alternate their order; board-to-board differences of
  5-10% were seen. An older vLLM build without `ROCM_RDNA_ATTN` measured 1.5-11% slower, so use the
  nightly vLLM as the reference.

## Expected results (gfx1151, Strix Halo, same board)

| Row | vLLM (ms) | llama.cpp default (ms) | llama.cpp final (ms) | Final / vLLM |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B-Instruct 3968 | 159.1 | 297.2 | 165.8 | 1.04x |
| Qwen2.5-3B-Instruct 3968 | 862.9 | 1,324.0 | 911.9 | 1.06x |
| Qwen2.5-7B-Instruct 3968 | 1,817.0 | 2,639.2 | 1,785.3 | 0.98x |
| Qwen3-1.7B 3968 | 652.2 | 884.9 | 561.7 | 0.86x |
| Qwen3-4B 3968 | 1,157.8 | 1,968.9 | 1,255.4 | 1.08x |
| Qwen3-8B 3968 | 1,993.7 | 3,100.1 | 2,177.2 | 1.09x |
| Llama-2-7B 1920 | 865.4 | 1,619.7 | 912.0 | 1.05x |
| Gemma-2B 8000 | 1,368.2 | 2,108.6 | 1,496.5 | 1.09x |
| Qwen3.6-27B 4096 | 7,762.5 | 10,140.6 | 7,878.5 | 1.02x |
| Gemma-3-12B-IT 4096 | 4,371.8 | 5,098.8 | 3,250.5 | 0.74x |

"Default" is this branch with all gates unset and the server's default cache flags. Qwen3-8B and
Gemma-2B have the smallest margin.

## Tests

Run with the gates ON and again OFF; both must pass:

```bash
for op in FLASH_ATTN_EXT MUL_MAT MUL_MAT_ID FFN_SWIGLU ROPE RMS_NORM_MUL_ROPE ROPE_SET_ROWS MMDQ_ADD; do
  ./build/bin/test-backend-ops test -o $op
done
LLAMA_SPM_FAST_REV_MERGE=1 ./build/bin/test-tokenizer-0 models/ggml-vocab-llama-spm.gguf
```

`FLASH_ATTN_EXT` includes new causal and random-mask prefill cases for head dims 64/128/256.

## Accuracy checks

- KL divergence vs an fp16 dequant + hipBLAS reference build (same source with
  `GGML_CUDA_FORCE_CUBLAS`), wikitext, context 4096 x 3: expected 0.00002-0.00036 with the gates vs
  0.0004-0.004 with them off.
- Greedy 32 tokens on 3 real wikitext prompts per row: expected to match the fp16 reference on 26 of 30
  prompts with the gates (18 of 30 with them off); remaining differences are near-tie tokens.
- Note: random-token benchmark prompts give odd text on both backends; judge output quality on real
  text prompts.
