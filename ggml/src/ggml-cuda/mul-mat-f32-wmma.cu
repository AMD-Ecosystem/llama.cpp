#include "mul-mat-f32-wmma.cuh"

// F32 x F32 -> F32 product on RDNA3 WMMA for the router of a MoE layer at prompt batch sizes (512 outputs, K = 2560).
//
// The WMMA units take half precision, so both operands are split into a high and a low half, x = x_hi + x_lo with
// x_hi = half(x) and x_lo = half(x - x_hi), which keeps 22 of the 24 bits of a float. A product is the sum of three
// WMMA products, x_hi*w_hi + x_lo*w_hi + x_hi*w_lo (x_lo*w_lo is below the float rounding of the sum). The three
// products of 16 values of K are summed in a fresh WMMA accumulator and then added to the running float sum: the
// accumulation inside a WMMA does not round to nearest, and chaining all of K through it loses about 2^-15 of the
// result, 20 times more than a float GEMM. This way the error against a double precision product is that of a
// float GEMM or a little smaller (max 4e-6 against 1e-5 of hipBLAS sgemm at K = 2560, activations ~N(0,1),
// weights ~N(0,0.03)); only the order of the sum differs, so a router decision can only change for near ties.
//
// Both operands are read as float, split on the way into LDS and shared by the 4 waves of a block. dst[token][output]:
// the tokens are the rows of the A operand and the outputs the columns of the B operand, so that a lane's accumulator
// is one output of 8 tokens and the 16 lanes of a half wave store 16 consecutive outputs of a token. The activations
// and weights must stay inside the half range (|x| < 65504 and not below ~1e-7 to keep the low half), which holds
// for normalized activations and router weights.
#define MUL_MAT_F32_WMMA_KC  32 // K per staged chunk
#define MUL_MAT_F32_WMMA_MAX_ROWS 1024 // most rows of the first operand
#define MUL_MAT_F32_WMMA_ROW 40 // halves per LDS row: 16 byte aligned, and the 8 lanes of a 16 byte read hit different banks

template <int TM, int TN, int WAVES_M, int WAVES_N>
static __global__ void __launch_bounds__(32*WAVES_M*WAVES_N) mul_mat_f32_wmma_kernel(
        const float * __restrict__ W, const float * __restrict__ X, float * __restrict__ dst,
        const int64_t n_out, const int64_t n_tok, const int K, const int64_t sw, const int64_t sx, const int64_t sd) {
#if defined(AMD_WMMA_256B_AVAILABLE)
    constexpr int KC      = MUL_MAT_F32_WMMA_KC;
    constexpr int ROW     = MUL_MAT_F32_WMMA_ROW;
    constexpr int THREADS = 32*WAVES_M*WAVES_N;
    constexpr int FA      = TM/WAVES_M/16; // 16 token tiles per wave
    constexpr int FB      = TN/WAVES_N/16; // 16 output tiles per wave
    constexpr int LX      = TM*(KC/4)/THREADS; // float4 loads of the tokens per thread and chunk
    constexpr int LW      = TN*(KC/4)/THREADS;
    static_assert(LX >= 1 && TM*(KC/4) % THREADS == 0 && LW >= 1 && TN*(KC/4) % THREADS == 0, "tile does not fit the block");

    using halfx16_t = __attribute__((ext_vector_type(16))) _Float16;
    using halfx4_t  = __attribute__((ext_vector_type(4)))  _Float16;
    using floatx8_t = __attribute__((ext_vector_type(8)))  float;

    const int tid    = threadIdx.x;
    const int lane   = tid % 32;
    const int wave   = tid / 32;
    const int g      = lane / 16;
    const int col    = lane % 16;
    const int wave_m = wave / WAVES_N;
    const int wave_n = wave % WAVES_N;

    const int64_t out0 = (int64_t) blockIdx.x*TN;
    const int64_t tok0 = (int64_t) blockIdx.y*TM;

    __shared__ _Float16 s_x[2][2][TM][ROW]; // [buffer][hi, lo][token][k]
    __shared__ _Float16 s_w[2][2][TN][ROW]; // [buffer][hi, lo][output][k]

    float4 x_regs[LX];
    float4 w_regs[LW];
    auto load_chunk = [&](const int k0) {
#pragma unroll
        for (int l = 0; l < LX; ++l) {
            const int i = tid + l*THREADS;
            const int r = i / (KC/4);
            const int c = i % (KC/4);
            x_regs[l] = tok0 + r < n_tok ? *(const float4 *) (X + (tok0 + r)*sx + k0 + c*4) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
#pragma unroll
        for (int l = 0; l < LW; ++l) {
            const int i = tid + l*THREADS;
            const int r = i / (KC/4);
            const int c = i % (KC/4);
            w_regs[l] = out0 + r < n_out ? *(const float4 *) (W + (out0 + r)*sw + k0 + c*4) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    };
    auto split_store = [&](const float4 v, _Float16 * hi, _Float16 * lo) {
        const float xs[4] = {v.x, v.y, v.z, v.w};
        halfx4_t h4, l4;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const _Float16 h = (_Float16) xs[e];
            h4[e] = h;
            l4[e] = (_Float16) (xs[e] - (float) h);
        }
        *(halfx4_t *) hi = h4;
        *(halfx4_t *) lo = l4;
    };
    auto store_chunk = [&](const int buf) {
#pragma unroll
        for (int l = 0; l < LX; ++l) {
            const int i = tid + l*THREADS;
            const int r = i / (KC/4);
            const int c = i % (KC/4);
            split_store(x_regs[l], &s_x[buf][0][r][c*4], &s_x[buf][1][r][c*4]);
        }
#pragma unroll
        for (int l = 0; l < LW; ++l) {
            const int i = tid + l*THREADS;
            const int r = i / (KC/4);
            const int c = i % (KC/4);
            split_store(w_regs[l], &s_w[buf][0][r][c*4], &s_w[buf][1][r][c*4]);
        }
    };
    auto frag = [](const _Float16 * p) {
        const int4 * q = (const int4 *) p;
        union { int4 i[2]; halfx16_t h; } u;
        u.i[0] = q[0];
        u.i[1] = q[1];
        return u.h;
    };

    floatx8_t acc[FA][FB];
#pragma unroll
    for (int ta = 0; ta < FA; ++ta) {
#pragma unroll
        for (int tb = 0; tb < FB; ++tb) {
            acc[ta][tb] = floatx8_t{0, 0, 0, 0, 0, 0, 0, 0};
        }
    }

    const int n_chunks = K / KC;
    load_chunk(0);
    store_chunk(0);
    __syncthreads();

    for (int c = 0; c < n_chunks; ++c) {
        const int buf = c & 1;
        const bool has_next = c + 1 < n_chunks;
        if (has_next) {
            load_chunk((c + 1)*KC);
        }

#pragma unroll
        for (int ks = 0; ks < KC/16; ++ks) {
            halfx16_t a_hi[FA], a_lo[FA], b_hi[FB], b_lo[FB];
#pragma unroll
            for (int ta = 0; ta < FA; ++ta) {
                const int row = wave_m*(FA*16) + ta*16 + col;
                a_hi[ta] = frag(&s_x[buf][0][row][ks*16]);
                a_lo[ta] = frag(&s_x[buf][1][row][ks*16]);
            }
#pragma unroll
            for (int tb = 0; tb < FB; ++tb) {
                const int row = wave_n*(FB*16) + tb*16 + col;
                b_hi[tb] = frag(&s_w[buf][0][row][ks*16]);
                b_lo[tb] = frag(&s_w[buf][1][row][ks*16]);
            }
            // the three products of 16 values of K are summed on their own and then added to the running sum:
            // chaining all of K through the WMMA accumulator loses about 2^-15 of the result (the accumulation
            // inside a WMMA does not round to nearest), the float adds here do
#pragma unroll
            for (int ta = 0; ta < FA; ++ta) {
#pragma unroll
                for (int tb = 0; tb < FB; ++tb) {
                    floatx8_t p = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_lo[ta], b_hi[tb], floatx8_t{0, 0, 0, 0, 0, 0, 0, 0});
                    p = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_hi[ta], b_lo[tb], p);
                    p = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_hi[ta], b_hi[tb], p);
                    acc[ta][tb] += p;
                }
            }
        }
        if (has_next) {
            store_chunk(buf ^ 1);
        }
        __syncthreads();
    }

    // accumulator i of this lane is token 2*i + g of the tile, output col
#pragma unroll
    for (int ta = 0; ta < FA; ++ta) {
#pragma unroll
        for (int tb = 0; tb < FB; ++tb) {
            const int64_t out = out0 + wave_n*(FB*16) + tb*16 + col;
            if (out < n_out) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int64_t tok = tok0 + wave_m*(FA*16) + ta*16 + 2*i + g;
                    if (tok < n_tok) {
                        dst[tok*sd + out] = acc[ta][tb][i];
                    }
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(W, X, dst, n_out, n_tok, K, sw, sx, sd);
    NO_DEVICE_CODE;
#endif // defined(AMD_WMMA_256B_AVAILABLE)
}

bool ggml_cuda_should_use_mul_mat_f32_wmma(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc) {
#if defined(GGML_USE_HIP)
    // tuned on RDNA3.5 (Strix Halo); GGML_HIP_F32_WMMA=0 / 1 turns it off / on
    static const int env_wmma = ggml_cuda_env_switch("GGML_HIP_F32_WMMA");
    static const int min_tok  = getenv("GGML_HIP_F32_WMMA_MIN_TOKENS") ? atoi(getenv("GGML_HIP_F32_WMMA_MIN_TOKENS")) : 16;
    if (!GGML_CUDA_CC_IS_RDNA3(cc) || !ggml_cuda_switch_or_rdna3_5(env_wmma, cc)) {
        return false;
    }
    return src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
        src0->nb[0] == sizeof(float) && src1->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float) &&
        src0->ne[0] % MUL_MAT_F32_WMMA_KC == 0 && src0->nb[1] % 16 == 0 && src1->nb[1] % 16 == 0 && dst->nb[1] % sizeof(float) == 0 &&
        (uintptr_t) src0->data % 16 == 0 && (uintptr_t) src1->data % 16 == 0 &&
        // router-like shapes only: a few hundred outputs (the half range of the split also rules out unnormalized activations)
        src0->ne[1] >= 16 && src0->ne[1] <= MUL_MAT_F32_WMMA_MAX_ROWS && src1->ne[1] >= min_tok;
#else
    GGML_UNUSED_VARS(src0, src1, dst, cc);
    return false;
#endif // defined(GGML_USE_HIP)
}

template <int TM, int TN, int WAVES_M, int WAVES_N>
static void launch_mul_mat_f32_wmma(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const dim3 block(32*WAVES_M*WAVES_N, 1, 1);
    const dim3 grid((src0->ne[1] + TN - 1)/TN, (src1->ne[1] + TM - 1)/TM, 1);
    mul_mat_f32_wmma_kernel<TM, TN, WAVES_M, WAVES_N><<<grid, block, 0, ctx.stream()>>>(
        (const float *) src0->data, (const float *) src1->data, (float *) dst->data,
        src0->ne[1], src1->ne[1], (int) src0->ne[0],
        src0->nb[1]/sizeof(float), src1->nb[1]/sizeof(float), dst->nb[1]/sizeof(float));
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_mul_mat_f32_wmma(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    launch_mul_mat_f32_wmma<64, 64, 2, 2>(ctx, src0, src1, dst);
}
