# HRX2 Known Limitations

This file tracks concrete HRX2 implementation limitations that should survive
individual tuning sessions. General Loom author feedback belongs elsewhere;
this file is for llama.cpp backend behavior and follow-up work.

## Removed HIP Bridge Kernels

Temporary HIP C++ bridge kernels for Q4_K, Q5_K, Q6_K, and FA0 attention have
been removed from the production HRX2 build and catalog. Their schedules are
preserved as prior art in `docs/development/hrx2-hip-bridge-prior-art.md`.

Current status:

- Q4_K, Q5_K, and Q6_K route coverage uses Loom bytecode providers and JIT
  output only.
- Focused backend-op gates for the affected quantized matmul routes pass.
- A reduced Llama 3.2 3B Q4_K HRX2/Vulkan integration smoke runs with zero CPU
  compute fallback.
- FA0 direct attention fusion has no accepted Loom replacement yet; production
  uses the unfused Loom/HRX2 attention route chain.

## Remaining Performance Gaps

The no-HIP bridge smoke from 2026-06-16 shows HRX2 is functional but still below
Vulkan on the representative Q4_K slice:

| Case | HRX2/Vulkan |
| --- | ---: |
| `decode-p1n64` | 0.376 |
| `prefill-p64n0` | 0.150 |
| `prefill-p512n0` | 0.170 |

The largest p64/p512 blocker is quantized prompt matmul route quality, especially
`mul_mat_q4_k_q8_1_x4_mmq64x32`. Future work should start from prior schedules,
Loom compile reports, emitted assembly, and focused kernel sweeps before
promoting integration routes.
