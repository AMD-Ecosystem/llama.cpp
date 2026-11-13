#include "mmdq.cuh"
#include "convert.cuh"
#include "mmid.cuh"
#include "unary.cuh"
#include "vecdotq.cuh"

#include <cstdlib>

// The structure follows the gfx11 W4A16 Triton GEMM in vLLM (Apache-2.0, rdna_hybrid_w4a16.py):
// fp16 activations, 4-bit -> fp16 via the 0x6400 exponent trick, fp16 WMMA with fp32 accumulators.

#define MMDQ_BK  64 // K per main loop step; one Q4_K/Q6_K block holds 4 steps
#define MMDQ_PAD  8 // halfs of LDS row padding, keeps the 16-row fragment reads free of bank conflicts

#if (defined(GGML_USE_HIP) && defined(RDNA3))
typedef _Float16 mmdq_half16_t __attribute__((ext_vector_type(16)));
typedef float    mmdq_float8_t __attribute__((ext_vector_type(8)));
#endif // (defined(GGML_USE_HIP) && defined(RDNA3))

// bytes 0,1 (lo) or 2,3 (hi) of v, each < 64, as fp16 pair (1024 + b)
static __device__ __forceinline__ half2 mmdq_bytes_lo(const uint32_t v) {
    const uint32_t u = (v & 0xFFu) | ((v << 8) & 0xFF0000u) | 0x64006400u;
    return *(const half2 *) &u;
}

static __device__ __forceinline__ half2 mmdq_bytes_hi(const uint32_t v) {
    const uint32_t u = ((v >> 16) & 0xFFu) | ((v >> 8) & 0xFF0000u) | 0x64006400u;
    return *(const half2 *) &u;
}

static __device__ __forceinline__ uint32_t mmdq_h2_as_u32(const half2 h) {
    return *(const uint32_t *) &h;
}

// raw weight data for one MMDQ_BK step of one row, as loaded from global memory
template <ggml_type type> struct mmdq_raw;

template <> struct mmdq_raw<GGML_TYPE_Q4_K> {
    static constexpr int threads_per_row = 2;
    uint4 hdr; // d, dmin, scales[12]
    uint4 qs;  // 16 bytes of qs: lo nibbles of sub-block 2j, hi nibbles of sub-block 2j+1
};

template <> struct mmdq_raw<GGML_TYPE_Q5_K> {
    static constexpr int threads_per_row = 2;
    uint4 hdr; // d, dmin, scales[12]
    uint4 qs;  // as for Q4_K
    uint4 qh;  // qh bytes of the same 16 positions, bit 2j/2j+1 is bit 4 of the low/high nibble
};

template <> struct mmdq_raw<GGML_TYPE_Q6_K> {
    static constexpr int threads_per_row = 2;
    int ql[8];
    int qh[8];
    int sc; // 2 int8 scales
    half d;
};

template <> struct mmdq_raw<GGML_TYPE_Q5_1> {
    static constexpr int threads_per_row = 2;
    half2    dm;
    uint32_t qh;
    uint4    qs; // one block of 32 values per thread
};

template <> struct mmdq_raw<GGML_TYPE_Q5_0> {
    static constexpr int threads_per_row = 2;
    half     d;
    uint32_t qh;
    uint4    qs; // one block of 32 values per thread
};

template <> struct mmdq_raw<GGML_TYPE_Q8_0> {
    static constexpr int threads_per_row = 2;
    int  qs[8]; // one block of 32 values per thread
    half d;
};

// h: which part of the row this thread handles, kq: K index of the step inside the block
static __device__ __forceinline__ void mmdq_load(mmdq_raw<GGML_TYPE_Q4_K> & raw, const char * row, const int ib, const int kq, const int h) {
    const block_q4_K * b = (const block_q4_K *) row + ib;
    raw.hdr = *(const uint4 *) b;
    raw.qs  = *(const uint4 *) (b->qs + 32*(kq/64) + 16*h);
}

static __device__ __forceinline__ void mmdq_load(mmdq_raw<GGML_TYPE_Q5_K> & raw, const char * row, const int ib, const int kq, const int h) {
    const block_q5_K * b = (const block_q5_K *) row + ib;
    raw.hdr = *(const uint4 *) b;
    raw.qs  = *(const uint4 *) (b->qs + 32*(kq/64) + 16*h);
    raw.qh  = *(const uint4 *) (b->qh + 16*h);
}

static __device__ __forceinline__ void mmdq_load(mmdq_raw<GGML_TYPE_Q6_K> & raw, const char * row, const int ib, const int kq, const int h) {
    const block_q6_K * b = (const block_q6_K *) row + ib;
    const int c  = kq/64;
    const int ip = c/2;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        raw.ql[i] = get_int_b2(b->ql + 64*ip + 32*h, i);
        raw.qh[i] = get_int_b2(b->qh + 32*ip, i);
    }
    const uint16_t * sc16 = (const uint16_t *) b->scales;
    raw.sc = sc16[2*c + h];
    raw.d  = b->d;
}

// ib/kq are in units of QK_K like for the K-quants, thread h takes the block kq/32 + h
static __device__ __forceinline__ void mmdq_load(mmdq_raw<GGML_TYPE_Q5_1> & raw, const char * row, const int ib, const int kq, const int h) {
    const block_q5_1 * b = (const block_q5_1 *) row + ib*(QK_K/QK5_1) + kq/QK5_1 + h;
    raw.dm = *(const half2 *) &b->dm;
    raw.qh = *(const uint32_t *) b->qh;
    const uint2 * qs = (const uint2 *) b->qs;
    const uint2 q0 = qs[0];
    const uint2 q1 = qs[1];
    raw.qs = make_uint4(q0.x, q0.y, q1.x, q1.y);
}

// Q5_0 blocks are 22 bytes, read with 2 byte loads
static __device__ __forceinline__ void mmdq_load(mmdq_raw<GGML_TYPE_Q5_0> & raw, const char * row, const int ib, const int kq, const int h) {
    const block_q5_0 * b = (const block_q5_0 *) row + ib*(QK_K/QK5_0) + kq/QK5_0 + h;
    raw.d  = b->d;
    raw.qh = get_int_b2(b->qh, 0);
    raw.qs = make_uint4(get_int_b2(b->qs, 0), get_int_b2(b->qs, 1), get_int_b2(b->qs, 2), get_int_b2(b->qs, 3));
}

static __device__ __forceinline__ void mmdq_load(mmdq_raw<GGML_TYPE_Q8_0> & raw, const char * row, const int ib, const int kq, const int h) {
    const block_q8_0 * b = (const block_q8_0 *) row + ib*(QK_K/QK8_0) + kq/QK8_0 + h;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        raw.qs[i] = get_int_b2(b->qs, i);
    }
    raw.d = b->d;
}

// get_scale_min_k4 on the scales held in hdr.y/z/w
static __device__ __forceinline__ void mmdq_scale_min_q4_K(const uint4 & hdr, const int j, int & sc, int & m) {
    if (j < 4) {
        sc = (hdr.y >> (8*j)) & 63;
        m  = (hdr.z >> (8*j)) & 63;
    } else {
        const int s = 8*(j - 4);
        sc = ((hdr.w >> s) & 0xF) | (((hdr.y >> (s + 6)) & 3) << 4);
        m  = ((hdr.w >> (s + 4)) & 0xF) | (((hdr.z >> (s + 6)) & 3) << 4);
    }
}

// Q4_K/Q5_K: dequantize 32 fp16 values, qh5 holds the Q5_K high bits (0 for Q4_K)
// v[0..7]: sub-block 2j values 16h..16h+15, v[8..15]: sub-block 2j+1 values 16h..16h+15
static __device__ __forceinline__ void mmdq_dequant_k(const uint4 & hdr, const uint4 & qs, const uint4 & qh5, const int kq, uint32_t * v) {
    const int j = kq/64;
    int s0, m0, s1, m1;
    mmdq_scale_min_q4_K(hdr, 2*j + 0, s0, m0);
    mmdq_scale_min_q4_K(hdr, 2*j + 1, s1, m1);
    const half2 dm   = *(const half2 *) &hdr.x;
    const float d    = __low2float(dm);
    const float dmin = __high2float(dm);
    const half2 d0  = __float2half2_rn(d*s0);
    const half2 b0  = __float2half2_rn(-dmin*m0);
    const half2 d1  = __float2half2_rn(d*s1);
    const half2 b1  = __float2half2_rn(-dmin*m1);
    const half2 off = __float2half2_rn(1024.0f);

    const uint32_t q[4] = {qs.x, qs.y, qs.z, qs.w};
    const uint32_t u[4] = {qh5.x, qh5.y, qh5.z, qh5.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t ql = (q[i] & 0x0F0F0F0Fu)        | (((u[i] >> (2*j + 0)) & 0x01010101u) << 4);
        const uint32_t qh = ((q[i] >> 4) & 0x0F0F0F0Fu) | (((u[i] >> (2*j + 1)) & 0x01010101u) << 4);
        v[2*i + 0] = mmdq_h2_as_u32(__hfma2(__hsub2(mmdq_bytes_lo(ql), off), d0, b0));
        v[2*i + 1] = mmdq_h2_as_u32(__hfma2(__hsub2(mmdq_bytes_hi(ql), off), d0, b0));
        v[2*i + 8] = mmdq_h2_as_u32(__hfma2(__hsub2(mmdq_bytes_lo(qh), off), d1, b1));
        v[2*i + 9] = mmdq_h2_as_u32(__hfma2(__hsub2(mmdq_bytes_hi(qh), off), d1, b1));
    }
}

static __device__ __forceinline__ void mmdq_dequant(const mmdq_raw<GGML_TYPE_Q4_K> & raw, const int kq, const int h, uint32_t * v) {
    GGML_UNUSED(h);
    mmdq_dequant_k(raw.hdr, raw.qs, make_uint4(0, 0, 0, 0), kq, v);
}

static __device__ __forceinline__ void mmdq_dequant(const mmdq_raw<GGML_TYPE_Q5_K> & raw, const int kq, const int h, uint32_t * v) {
    GGML_UNUSED(h);
    mmdq_dequant_k(raw.hdr, raw.qs, raw.qh, kq, v);
}

// Q6_K: v[0..15] are the values 32h..32h+31 of the 64-step
static __device__ __forceinline__ void mmdq_dequant(const mmdq_raw<GGML_TYPE_Q6_K> & raw, const int kq, const int h, uint32_t * v) {
    const int odd = (kq/64) % 2;
    const int sl  = 4*odd;
    const int sh  = 4*odd + 2*h;
    const float d = __half2float(raw.d);
    const half2 dA  = __float2half2_rn(d*(int8_t) (raw.sc & 0xFF));
    const half2 dB  = __float2half2_rn(d*(int8_t) ((raw.sc >> 8) & 0xFF));
    const half2 off = __float2half2_rn(1056.0f); // 1024 + 32
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const uint32_t q = (((uint32_t) raw.ql[i] >> sl) & 0x0F0F0F0Fu) | ((((uint32_t) raw.qh[i] >> sh) & 0x03030303u) << 4);
        const half2 ds = i < 4 ? dA : dB;
        v[2*i + 0] = mmdq_h2_as_u32(__hmul2(__hsub2(mmdq_bytes_lo(q), off), ds));
        v[2*i + 1] = mmdq_h2_as_u32(__hmul2(__hsub2(mmdq_bytes_hi(q), off), ds));
    }
}

// Q5_1: v[0..15] are the values 32h..32h+31 of the 64-step
static __device__ __forceinline__ void mmdq_dequant(const mmdq_raw<GGML_TYPE_Q5_1> & raw, const int kq, const int h, uint32_t * v) {
    GGML_UNUSED_VARS(kq, h);
    const half2 d   = __low2half2(raw.dm);
    const half2 m   = __high2half2(raw.dm);
    const half2 off = __float2half2_rn(1024.0f);
    const uint32_t q[4] = {raw.qs.x, raw.qs.y, raw.qs.z, raw.qs.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        // bit j of qh is bit 4 of value j, values 4i..4i+3 (low nibbles) and 16+4i..16+4i+3 (high nibbles)
        const uint32_t hl = (raw.qh >> (4*i))      & 0xFu;
        const uint32_t hh = (raw.qh >> (16 + 4*i)) & 0xFu;
        const uint32_t bl = ((hl & 1u) << 4) | ((hl & 2u) << 11) | ((hl & 4u) << 18) | ((hl & 8u) << 25);
        const uint32_t bh = ((hh & 1u) << 4) | ((hh & 2u) << 11) | ((hh & 4u) << 18) | ((hh & 8u) << 25);
        const uint32_t ql = (q[i] & 0x0F0F0F0Fu) | bl;
        const uint32_t qu = ((q[i] >> 4) & 0x0F0F0F0Fu) | bh;
        v[2*i + 0] = mmdq_h2_as_u32(__hfma2(__hsub2(mmdq_bytes_lo(ql), off), d, m));
        v[2*i + 1] = mmdq_h2_as_u32(__hfma2(__hsub2(mmdq_bytes_hi(ql), off), d, m));
        v[2*i + 8] = mmdq_h2_as_u32(__hfma2(__hsub2(mmdq_bytes_lo(qu), off), d, m));
        v[2*i + 9] = mmdq_h2_as_u32(__hfma2(__hsub2(mmdq_bytes_hi(qu), off), d, m));
    }
}

// Q5_0: as Q5_1 with value = (q - 16)*d
static __device__ __forceinline__ void mmdq_dequant(const mmdq_raw<GGML_TYPE_Q5_0> & raw, const int kq, const int h, uint32_t * v) {
    GGML_UNUSED_VARS(kq, h);
    const half2 d   = __half2half2(raw.d);
    const half2 off = __float2half2_rn(1040.0f); // 1024 + 16
    const uint32_t q[4] = {raw.qs.x, raw.qs.y, raw.qs.z, raw.qs.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t hl = (raw.qh >> (4*i))      & 0xFu;
        const uint32_t hh = (raw.qh >> (16 + 4*i)) & 0xFu;
        const uint32_t bl = ((hl & 1u) << 4) | ((hl & 2u) << 11) | ((hl & 4u) << 18) | ((hl & 8u) << 25);
        const uint32_t bh = ((hh & 1u) << 4) | ((hh & 2u) << 11) | ((hh & 4u) << 18) | ((hh & 8u) << 25);
        const uint32_t ql = (q[i] & 0x0F0F0F0Fu) | bl;
        const uint32_t qu = ((q[i] >> 4) & 0x0F0F0F0Fu) | bh;
        v[2*i + 0] = mmdq_h2_as_u32(__hmul2(__hsub2(mmdq_bytes_lo(ql), off), d));
        v[2*i + 1] = mmdq_h2_as_u32(__hmul2(__hsub2(mmdq_bytes_hi(ql), off), d));
        v[2*i + 8] = mmdq_h2_as_u32(__hmul2(__hsub2(mmdq_bytes_lo(qu), off), d));
        v[2*i + 9] = mmdq_h2_as_u32(__hmul2(__hsub2(mmdq_bytes_hi(qu), off), d));
    }
}

// Q8_0: v[0..15] are the values 32h..32h+31 of the 64-step; the int8 values are biased by 128 to use the 0x6400 trick
static __device__ __forceinline__ void mmdq_dequant(const mmdq_raw<GGML_TYPE_Q8_0> & raw, const int kq, const int h, uint32_t * v) {
    GGML_UNUSED_VARS(kq, h);
    const half2 d   = __half2half2(raw.d);
    const half2 off = __float2half2_rn(1152.0f); // 1024 + 128
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const uint32_t q = (uint32_t) raw.qs[i] ^ 0x80808080u;
        v[2*i + 0] = mmdq_h2_as_u32(__hmul2(__hsub2(mmdq_bytes_lo(q), off), d));
        v[2*i + 1] = mmdq_h2_as_u32(__hmul2(__hsub2(mmdq_bytes_hi(q), off), d));
    }
}

static __device__ __forceinline__ void mmdq_store_v16(const uint32_t * v, half * dst0, half * dst1) {
    uint4 * d0 = (uint4 *) dst0;
    uint4 * d1 = (uint4 *) dst1;
    d0[0] = make_uint4(v[0],  v[1],  v[2],  v[3]);
    d0[1] = make_uint4(v[4],  v[5],  v[6],  v[7]);
    d1[0] = make_uint4(v[8],  v[9],  v[10], v[11]);
    d1[1] = make_uint4(v[12], v[13], v[14], v[15]);
}

// write the dequantized values of one row and one 64-step to the LDS row
static __device__ __forceinline__ void mmdq_store(const mmdq_raw<GGML_TYPE_Q4_K> & raw, half * srow, const int kq, const int h) {
    uint32_t v[16];
    mmdq_dequant(raw, kq, h, v);
    mmdq_store_v16(v, srow + 16*h, srow + 32 + 16*h);
}

static __device__ __forceinline__ void mmdq_store(const mmdq_raw<GGML_TYPE_Q5_K> & raw, half * srow, const int kq, const int h) {
    uint32_t v[16];
    mmdq_dequant(raw, kq, h, v);
    mmdq_store_v16(v, srow + 16*h, srow + 32 + 16*h);
}

static __device__ __forceinline__ void mmdq_store(const mmdq_raw<GGML_TYPE_Q6_K> & raw, half * srow, const int kq, const int h) {
    uint32_t v[16];
    mmdq_dequant(raw, kq, h, v);
    mmdq_store_v16(v, srow + 32*h, srow + 32*h + 16);
}

static __device__ __forceinline__ void mmdq_store(const mmdq_raw<GGML_TYPE_Q5_0> & raw, half * srow, const int kq, const int h) {
    uint32_t v[16];
    mmdq_dequant(raw, kq, h, v);
    mmdq_store_v16(v, srow + 32*h, srow + 32*h + 16);
}

static __device__ __forceinline__ void mmdq_store(const mmdq_raw<GGML_TYPE_Q8_0> & raw, half * srow, const int kq, const int h) {
    uint32_t v[16];
    mmdq_dequant(raw, kq, h, v);
    mmdq_store_v16(v, srow + 32*h, srow + 32*h + 16);
}

// MUL_MAT_ID: the rows of y are sorted by expert (see mm_ids_helper), each block works on BT rows of one expert.
struct mmdq_ids_args {
    const int32_t * ids_src1;      // compact row -> row of y
    const int32_t * ids_dst;       // compact row -> row of dst
    const int32_t * expert_bounds; // compact rows of expert e: [expert_bounds[e], expert_bounds[e+1])
    const int32_t * tile_start;    // first block of expert e, tile_start[n_experts] = number of blocks
    int             n_experts;
    int64_t         stride_expert_x;
};

// glu: 0 = plain GEMM, 1 = SwiGLU, 2 = dense GEGLU. Hidden output uses FP16.
// x is the gate weight and x_up the up weight. GEGLU is separately opt-in.
// LDS rows alternate between 16 gate rows and the same 16 up rows, so each wave holds both (needs an even TO).
// Without glu (dense only): dst[t][o] += bias[o] + res[t][o] for the non-null ones.
struct mmdq_epi_args {
    const char  * x_up;
    half        * dst_h;
    const float * bias;
    const float * res;
    int64_t       stride_col_res;
};

// dst[t][o] = sum_k y[t][k] * x[o][k]; t: token (src1 column), o: output row (src0 row)
// BT x BO output tile per block, NWT x NWO waves, each wave computes (BT/NWT) x (BO/NWO)
// add: bit 0 = add epi.bias, bit 1 = add epi.res (compile time, a runtime check in the store loop slows Q4_K down by up to 16%)
template <ggml_type type, bool has_ids, int glu, int add, int BT, int BO, int NWT, int NWO>
__launch_bounds__(32*NWT*NWO, 1)
static __global__ void mul_mat_dq(
        const char * __restrict__ x, const half * __restrict__ y, float * __restrict__ dst,
        const int ncols_x, const int nrows_x, const int ncols_y, const int64_t stride_row_x, const int64_t stride_col_dst,
        const mmdq_ids_args ids, const mmdq_epi_args epi) {
#if (defined(GGML_USE_HIP) && defined(RDNA3))
    static_assert(!glu || (!has_ids && (BO/(16*NWO)) % 2 == 0), "bad glu config");
    static_assert(add == 0 || (!has_ids && !glu), "bad add config");
    constexpr int nthreads = 32*NWT*NWO;
    constexpr int TT  = BT/(16*NWT);
    constexpr int TO  = BO/(16*NWO);
    constexpr int LDK = MMDQ_BK + MMDQ_PAD;
    constexpr int tpr = mmdq_raw<type>::threads_per_row;
    constexpr int nload_y = BT*(MMDQ_BK/8)/nthreads;
    constexpr int nload_x = BO*tpr/nthreads;
    static_assert(nload_y*nthreads == BT*(MMDQ_BK/8), "bad y tile");
    static_assert(nload_x*nthreads == BO*tpr, "bad x tile");

    __shared__ half sy[BT*LDK];
    __shared__ half sx[BO*LDK];

    const int tid  = threadIdx.x;
    const int lane = tid % 32;
    const int wave = tid / 32;
    const int wt   = wave / NWO;
    const int wo   = wave % NWO;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int o0   = by*(glu ? BO/2 : BO);

    int t0    = bx*BT;
    int t_end = ncols_y;
    if constexpr (has_ids) {
        const int b = blockIdx.x;
        if (b >= ids.tile_start[ids.n_experts]) {
            return;
        }
        int lo = 0;
        int hi = ids.n_experts;
        while (hi - lo > 1) {
            const int mid = (lo + hi)/2;
            if (ids.tile_start[mid] <= b) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        t0    = ids.expert_bounds[lo] + (b - ids.tile_start[lo])*BT;
        t_end = ids.expert_bounds[lo + 1];
        x    += lo*ids.stride_expert_x;
    }

    uint4         ry[nload_y];
    mmdq_raw<type> rx[nload_x];

    auto load_global = [&](const int k0) {
#pragma unroll
        for (int i = 0; i < nload_y; ++i) {
            const int idx = tid + i*nthreads;
            const int t   = t0 + idx/(MMDQ_BK/8);
            const int c   = idx % (MMDQ_BK/8);
            if (t < t_end) {
                const int row = has_ids ? ids.ids_src1[t] : t;
                ry[i] = *(const uint4 *) (y + (int64_t) row*ncols_x + k0 + 8*c);
            } else {
                ry[i] = make_uint4(0, 0, 0, 0);
            }
        }
#pragma unroll
        for (int i = 0; i < nload_x; ++i) {
            const int idx = tid + i*nthreads;
            if constexpr (glu) {
                const int r = idx/tpr;
                const int o = min(o0 + (r/32)*16 + r % 16, nrows_x - 1);
                mmdq_load(rx[i], ((r/16) % 2 ? epi.x_up : x) + o*stride_row_x, k0/QK_K, k0 % QK_K, idx % tpr);
            } else {
                const int o = min(o0 + idx/tpr, nrows_x - 1);
                mmdq_load(rx[i], x + o*stride_row_x, k0/QK_K, k0 % QK_K, idx % tpr);
            }
        }
    };

    auto store_lds = [&](const int k0) {
#pragma unroll
        for (int i = 0; i < nload_y; ++i) {
            const int idx = tid + i*nthreads;
            *(uint4 *) (sy + (idx/(MMDQ_BK/8))*LDK + 8*(idx % (MMDQ_BK/8))) = ry[i];
        }
#pragma unroll
        for (int i = 0; i < nload_x; ++i) {
            const int idx = tid + i*nthreads;
            mmdq_store(rx[i], sx + (idx/tpr)*LDK, k0 % QK_K, idx % tpr);
        }
    };

    mmdq_float8_t acc[TT][TO];
#pragma unroll
    for (int i = 0; i < TT; ++i) {
#pragma unroll
        for (int j = 0; j < TO; ++j) {
            acc[i][j] = mmdq_float8_t{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        }
    }

    load_global(0);
    store_lds(0);
    __syncthreads();

    for (int k0 = 0; k0 < ncols_x; k0 += MMDQ_BK) {
        const bool has_next = k0 + MMDQ_BK < ncols_x;
        if (has_next) {
            load_global(k0 + MMDQ_BK);
        }

#pragma unroll
        for (int kk = 0; kk < MMDQ_BK; kk += 16) {
            mmdq_half16_t A[TT];
            mmdq_half16_t B[TO];
#pragma unroll
            for (int i = 0; i < TT; ++i) {
                const uint4 * src = (const uint4 *) (sy + (wt*TT*16 + i*16 + lane % 16)*LDK + kk);
                uint4 tmp[2] = {src[0], src[1]};
                memcpy(&A[i], tmp, sizeof(A[i]));
            }
#pragma unroll
            for (int j = 0; j < TO; ++j) {
                const uint4 * src = (const uint4 *) (sx + (wo*TO*16 + j*16 + lane % 16)*LDK + kk);
                uint4 tmp[2] = {src[0], src[1]};
                memcpy(&B[j], tmp, sizeof(B[j]));
            }
#pragma unroll
            for (int i = 0; i < TT; ++i) {
#pragma unroll
                for (int j = 0; j < TO; ++j) {
                    acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(A[i], B[j], acc[i][j]);
                }
            }
        }

        if (has_next) {
            __syncthreads();
            store_lds(k0 + MMDQ_BK);
            __syncthreads();
        }
    }

    // acc element l of a lane: token 2*l + lane/16, output row lane % 16
    if constexpr (glu) {
#pragma unroll
        for (int i = 0; i < TT; ++i) {
#pragma unroll
            for (int j = 0; j < TO; j += 2) {
                const int o = o0 + (wo*TO + j)/2*16 + lane % 16;
                if (o >= nrows_x) {
                    continue;
                }
#pragma unroll
                for (int l = 0; l < 8; ++l) {
                    const int t = t0 + wt*TT*16 + i*16 + 2*l + lane/16;
                    if (t < t_end) {
                        epi.dst_h[(int64_t) t*stride_col_dst + o] = __float2half((glu == 2 ? ggml_cuda_op_gelu_single(acc[i][j][l]) : ggml_cuda_op_silu_single(acc[i][j][l])) * acc[i][j + 1][l]);
                    }
                }
            }
        }
        return;
    }
#pragma unroll
    for (int i = 0; i < TT; ++i) {
#pragma unroll
        for (int j = 0; j < TO; ++j) {
            const int o = o0 + wo*TO*16 + j*16 + lane % 16;
            if (o >= nrows_x) {
                continue;
            }
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                const int t = t0 + wt*TT*16 + i*16 + 2*l + lane/16;
                if (t < t_end) {
                    const int row = has_ids ? ids.ids_dst[t] : t;
                    float v = acc[i][j][l];
                    if constexpr (add & 1) {
                        v += epi.bias[o];
                    }
                    if constexpr (add & 2) {
                        v += epi.res[(int64_t) row*epi.stride_col_res + o];
                    }
                    dst[row*stride_col_dst + o] = v;
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(x, y, dst, ncols_x, nrows_x, ncols_y, stride_row_x, stride_col_dst, ids, epi);
    NO_DEVICE_CODE;
#endif // (defined(GGML_USE_HIP) && defined(RDNA3))
}

// tile_start[e] = number of BT-row blocks of all experts before e
static __global__ void mmdq_tile_start(const int32_t * __restrict__ expert_bounds, int32_t * __restrict__ tile_start, const int n_experts, const int BT) {
    __shared__ int buf[1024];
    const int e = threadIdx.x;
    const int n = e < n_experts ? (expert_bounds[e + 1] - expert_bounds[e] + BT - 1)/BT : 0;
    buf[e] = n;
    __syncthreads();
    for (int off = 1; off < 1024; off *= 2) {
        const int v = e >= off ? buf[e - off] : 0;
        __syncthreads();
        buf[e] += v;
        __syncthreads();
    }
    if (e < n_experts) {
        tile_start[e] = buf[e] - n;
    }
    if (e == n_experts - 1) {
        tile_start[n_experts] = buf[e];
    }
}

// MoE experts with few tokens per expert: each lane reads its weight row from global memory into registers and dequantizes it there,
// so every weight is read once per expert and token group; only the activations go through LDS.
// Block: NW waves x 16 output rows, up to 16*NT tokens of one expert.
#define MMDQ_MOE_KC 128 // K per LDS activation chunk

// The dequantized 64-step of a lane is 2 groups of 8 dwords. Group g holds 16-K step s_own[g] in lanes 0-15 and step s_other[g] in
// lanes 16-31, so for s_own[g] lanes 16-31 take the group from lanes 0-15 and for s_other[g] the other way around.
template <ggml_type type> struct mmdq_moe_map {
    static constexpr int s_own[2]   = {0, 2};
    static constexpr int s_other[2] = {1, 3};
};

template <> struct mmdq_moe_map<GGML_TYPE_Q6_K> {
    static constexpr int s_own[2]   = {0, 1};
    static constexpr int s_other[2] = {2, 3};
};

template <> struct mmdq_moe_map<GGML_TYPE_Q5_1> : mmdq_moe_map<GGML_TYPE_Q6_K> {};

#define MMDQ_MOE_MAX_EXPERTS 512

// glu: waves [0, NW/2) compute gate rows, waves [NW/2, NW) the same up rows, dst_h = half(silu(gate) * up)
template <ggml_type type, int NT, int NW, bool glu>
__launch_bounds__(32*NW, 1)
static __global__ void mul_mat_dq_moe(
        const char * __restrict__ x, const half * __restrict__ y, float * __restrict__ dst,
        const int ncols_x, const int nrows_x, const int64_t stride_row_x, const int64_t stride_col_dst, const mmdq_ids_args ids,
        const mmdq_epi_args epi) {
#if (defined(GGML_USE_HIP) && defined(RDNA3))
    constexpr int NWR = glu ? NW/2 : NW; // waves per 16-row group
    static_assert(!glu || (NW % 2 == 0 && 16*NT*16*NWR*sizeof(float) <= 16*NT*(MMDQ_MOE_KC + MMDQ_PAD)*sizeof(half)), "bad glu config");
    constexpr int TMAX     = 16*NT;
    constexpr int LDK      = MMDQ_MOE_KC + MMDQ_PAD;
    constexpr int nthreads = 32*NW;
    constexpr int nload    = TMAX*(MMDQ_MOE_KC/8)/nthreads; // uint4 of activations per thread and chunk
    static_assert(nload*nthreads == TMAX*(MMDQ_MOE_KC/8), "bad activation tile");

    __shared__ half sy[TMAX*LDK];
    __shared__ int  s_tile_start[MMDQ_MOE_MAX_EXPERTS + 1];
    __shared__ int  s_rows[TMAX];

    const int tid  = threadIdx.x;
    const int lane = tid % 32;
    const int wave = tid / 32;
    const int h    = lane / 16;

    // find the expert of this block with the tile table in LDS, a search in global memory is a chain of dependent loads
    for (int e = tid; e <= ids.n_experts; e += nthreads) {
        s_tile_start[e] = ids.tile_start[e];
    }
    __syncthreads();
    // blockIdx.x: row block, blockIdx.y: token tile, so that the blocks reading the same activations run close together in time
    const int b = blockIdx.y;
    if (b >= s_tile_start[ids.n_experts]) {
        return;
    }
    int lo = 0;
    int hi = ids.n_experts;
    while (hi - lo > 1) {
        const int mid = (lo + hi)/2;
        if (s_tile_start[mid] <= b) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    const int t0     = ids.expert_bounds[lo] + (b - s_tile_start[lo])*TMAX;
    const int ntok   = min(TMAX, ids.expert_bounds[lo + 1] - t0);
    const int ntiles = (ntok + 15)/16;
    for (int t = tid; t < TMAX; t += nthreads) {
        s_rows[t] = t < ntok ? ids.ids_src1[t0 + t] : -1;
    }

    const bool is_up = glu && wave >= NWR;
    const int o = blockIdx.x*16*NWR + (wave % NWR)*16 + lane % 16;
    const char * xrow = (is_up ? epi.x_up : x) + lo*ids.stride_expert_x + min(o, nrows_x - 1)*stride_row_x;

    mmdq_float8_t acc[NT];
#pragma unroll
    for (int i = 0; i < NT; ++i) {
        acc[i] = mmdq_float8_t{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    }

    // the weights and activations of the next chunk are loaded to registers while the current chunk is used
    uint4 ry[nload];
    mmdq_raw<type> raw[2];
    mmdq_raw<type> raw_next[2];

    auto load_y = [&](const int kc) {
#pragma unroll
        for (int i = 0; i < nload; ++i) {
            const int idx = tid + i*nthreads;
            const int t   = idx / (MMDQ_MOE_KC/8);
            const int c   = idx % (MMDQ_MOE_KC/8);
            const int row = s_rows[t];
            ry[i] = row >= 0 && kc + 8*c < ncols_x ? *(const uint4 *) (y + (int64_t) row*ncols_x + kc + 8*c) : make_uint4(0, 0, 0, 0);
        }
    };
    auto store_y = [&]() {
#pragma unroll
        for (int i = 0; i < nload; ++i) {
            const int idx = tid + i*nthreads;
            *(uint4 *) (sy + (idx / (MMDQ_MOE_KC/8))*LDK + 8*(idx % (MMDQ_MOE_KC/8))) = ry[i];
        }
    };
    auto load_x = [&](mmdq_raw<type> * r, const int kc) {
#pragma unroll
        for (int j = 0; j < MMDQ_MOE_KC/64; ++j) {
            const int k = kc + 64*j;
            if (k < ncols_x) {
                mmdq_load(r[j], xrow, k/QK_K, k % QK_K, h);
            }
        }
    };

    __syncthreads(); // s_rows
    load_x(raw, 0);
    load_y(0);
    store_y();
    __syncthreads();

    for (int kc = 0; kc < ncols_x; kc += MMDQ_MOE_KC) {
        const bool has_next = kc + MMDQ_MOE_KC < ncols_x;
        if (has_next) {
            load_x(raw_next, kc + MMDQ_MOE_KC);
            load_y(kc + MMDQ_MOE_KC);
        }

#pragma unroll
        for (int j = 0; j < MMDQ_MOE_KC/64; ++j) {
            const int k0 = kc + 64*j;
            if (k0 >= ncols_x) {
                break; // K is a multiple of 64, only the last chunk can be partial
            }

            uint32_t w[16];
            mmdq_dequant(raw[j], k0 % QK_K, h, w);

            // B fragments of the 4 16-K steps
            mmdq_half16_t B[4];
#pragma unroll
            for (int g = 0; g < 2; ++g) {
                uint32_t b_own[8];
                uint32_t b_other[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const uint32_t wo = w[8*g + i];
                    const uint32_t wx = __builtin_amdgcn_permlanex16(wo, wo, 0x76543210u, 0xfedcba98u, false, false);
                    b_own[i]   = h == 0 ? wo : wx;
                    b_other[i] = h == 0 ? wx : wo;
                }
                memcpy(&B[mmdq_moe_map<type>::s_own[g]],   b_own,   sizeof(B[0]));
                memcpy(&B[mmdq_moe_map<type>::s_other[g]], b_other, sizeof(B[0]));
            }

#pragma unroll
            for (int i = 0; i < NT; ++i) {
                if (i < ntiles) {
#pragma unroll
                    for (int st = 0; st < 4; ++st) {
                        const uint4 * src = (const uint4 *) (sy + (16*i + lane % 16)*LDK + 64*j + 16*st);
                        uint4 tmp[2] = {src[0], src[1]};
                        mmdq_half16_t A;
                        memcpy(&A, tmp, sizeof(A));
                        acc[i] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(A, B[st], acc[i]);
                    }
                }
                // keep the A loads of a tile next to its WMMAs, hoisting all of them costs too many VGPRs
                __builtin_amdgcn_sched_barrier(0);
            }
        }

        if (has_next) {
#pragma unroll
            for (int j = 0; j < MMDQ_MOE_KC/64; ++j) {
                raw[j] = raw_next[j];
            }
            __syncthreads();
            store_y();
            __syncthreads();
        }
    }

    // acc element l of a lane: token 2*l + lane/16, output row lane % 16
    if constexpr (glu) {
        // the up waves pass their results to the gate waves through LDS
        float * s_up = (float *) sy;
        __syncthreads();
        if (is_up) {
#pragma unroll
            for (int i = 0; i < NT; ++i) {
#pragma unroll
                for (int l = 0; l < 8; ++l) {
                    s_up[(16*i + 2*l + h)*16*NWR + (wave - NWR)*16 + lane % 16] = acc[i][l];
                }
            }
        }
        __syncthreads();
        if (is_up || o >= nrows_x) {
            return;
        }
#pragma unroll
        for (int i = 0; i < NT; ++i) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                const int t = 16*i + 2*l + h;
                if (t < ntok) {
                    const float up = s_up[t*16*NWR + wave*16 + lane % 16];
                    epi.dst_h[ids.ids_dst[t0 + t]*stride_col_dst + o] = __float2half(ggml_cuda_op_silu_single(acc[i][l]) * up);
                }
            }
        }
        return;
    }
    if (o >= nrows_x) {
        return;
    }
#pragma unroll
    for (int i = 0; i < NT; ++i) {
#pragma unroll
        for (int l = 0; l < 8; ++l) {
            const int t = 16*i + 2*l + h;
            if (t < ntok) {
                dst[ids.ids_dst[t0 + t]*stride_col_dst + o] = acc[i][l];
            }
        }
    }
#else
    GGML_UNUSED_VARS(x, y, dst, ncols_x, nrows_x, stride_row_x, stride_col_dst, ids, epi);
    NO_DEVICE_CODE;
#endif // (defined(GGML_USE_HIP) && defined(RDNA3))
}

template <ggml_type type, int NT, int NW, bool glu>
static void mul_mat_dq_moe_launch(
        const char * x, const half * y, float * dst, const int ncols_x, const int nrows_x, const int n_rows,
        const int64_t stride_row_x, const int64_t stride_col_dst, const mmdq_ids_args & ids, const mmdq_epi_args & epi, cudaStream_t stream) {
    mmdq_tile_start<<<1, 1024, 0, stream>>>(ids.expert_bounds, (int32_t *) ids.tile_start, ids.n_experts, 16*NT);
    constexpr int rows_per_block = 16*(glu ? NW/2 : NW);
    const dim3 grid((nrows_x + rows_per_block - 1)/rows_per_block, (n_rows + 16*NT - 1)/(16*NT) + ids.n_experts, 1);
    mul_mat_dq_moe<type, NT, NW, glu><<<grid, 32*NW, 0, stream>>>(x, y, dst, ncols_x, nrows_x, stride_row_x, stride_col_dst, ids, epi);
}

template <ggml_type type, bool has_ids, int glu, int BT, int BO, int NWT, int NWO>
static void mul_mat_dq_launch(
        const char * x, const half * y, float * dst, const int ncols_x, const int nrows_x, const int ncols_y,
        const int64_t stride_row_x, const int64_t stride_col_dst, const mmdq_ids_args & ids, const mmdq_epi_args & epi_in, cudaStream_t stream) {
    if constexpr (glu && (has_ids || (BO/(16*NWO)) % 2 != 0)) {
        GGML_ABORT("mmdq config not supported with glu");
    } else {
        const mmdq_epi_args & epi = epi_in;
        int nblocks_t = (ncols_y + BT - 1)/BT;
        if constexpr (has_ids) {
            // ncols_y is the number of compact rows, each expert adds at most one partial block
            nblocks_t += ids.n_experts;
            mmdq_tile_start<<<1, 1024, 0, stream>>>(ids.expert_bounds, (int32_t *) ids.tile_start, ids.n_experts, BT);
        }
        constexpr int rows_per_block = glu ? BO/2 : BO;
        const dim3 grid(nblocks_t, (nrows_x + rows_per_block - 1)/rows_per_block, 1);
        const dim3 block(32*NWT*NWO, 1, 1);
        if constexpr (!has_ids && !glu) {
            const int add = (epi.bias ? 1 : 0) | (epi.res ? 2 : 0);
            switch (add) {
                case 1: mul_mat_dq<type, false, false, 1, BT, BO, NWT, NWO><<<grid, block, 0, stream>>>(x, y, dst, ncols_x, nrows_x, ncols_y, stride_row_x, stride_col_dst, ids, epi); return;
                case 2: mul_mat_dq<type, false, false, 2, BT, BO, NWT, NWO><<<grid, block, 0, stream>>>(x, y, dst, ncols_x, nrows_x, ncols_y, stride_row_x, stride_col_dst, ids, epi); return;
                case 3: mul_mat_dq<type, false, false, 3, BT, BO, NWT, NWO><<<grid, block, 0, stream>>>(x, y, dst, ncols_x, nrows_x, ncols_y, stride_row_x, stride_col_dst, ids, epi); return;
                default: break;
            }
        }
        mul_mat_dq<type, has_ids, glu, 0, BT, BO, NWT, NWO><<<grid, block, 0, stream>>>(x, y, dst, ncols_x, nrows_x, ncols_y, stride_row_x, stride_col_dst, ids, epi);
    }
}

static int mmdq_env_int(const char * name, const int def) {
    const char * env = getenv(name);
    return env != nullptr ? atoi(env) : def;
}

static bool mmdq_env_flag(const char * name) {
    return mmdq_env_int(name, 0) != 0;
}

// ncols_y_cfg: columns per expert on average for MUL_MAT_ID
template <ggml_type type, bool has_ids, int glu>
static void mul_mat_dq_switch(
        const char * x, const half * y, float * dst, const int ncols_x, const int nrows_x, const int ncols_y, const int ncols_y_cfg,
        const int64_t stride_row_x, const int64_t stride_col_dst, const mmdq_ids_args & ids, const mmdq_epi_args & epi, cudaStream_t stream) {
    static const int cfg_env = mmdq_env_int(has_ids ? "GGML_CUDA_MMDQ_ID_CFG" : "GGML_CUDA_MMDQ_CFG", -1);
    int cfg = cfg_env;
    if (cfg < 0) {
        // tuned on gfx1151 with Qwen2.5-VL-7B, Gemma-4-31B and MoE expert shapes
        if (has_ids) {
            cfg = ncols_y_cfg <= 24 ? 8 : 5;
        } else if (ncols_y <= 64) {
            cfg = 3;
        } else if (ncols_y <= 128) {
            cfg = 4;
        } else {
            // glu: nrows_x is the number of gate rows, the block works on as many up rows
            cfg = type == GGML_TYPE_Q4_K && (glu ? 2 : 1)*nrows_x >= 16384 ? 6 : 0;
        }
    }
#define MMDQ_CASE(id, BT, BO, NWT, NWO) case id: mul_mat_dq_launch<type, has_ids, glu, BT, BO, NWT, NWO>(x, y, dst, ncols_x, nrows_x, ncols_y, stride_row_x, stride_col_dst, ids, epi, stream); break
    switch (cfg) {
        MMDQ_CASE(0, 128, 128, 2, 4);
        MMDQ_CASE(1,  64, 128, 1, 4);
        MMDQ_CASE(2,  32, 128, 1, 4);
        MMDQ_CASE(3,  64,  64, 2, 2);
        MMDQ_CASE(4, 128,  64, 2, 2);
        MMDQ_CASE(5,  32,  64, 1, 2);
        MMDQ_CASE(6, 128, 256, 2, 4);
        MMDQ_CASE(7,  64,  32, 2, 1);
        MMDQ_CASE(8,  16,  64, 1, 4);
        MMDQ_CASE(9,  16, 128, 1, 4);
        default: GGML_ABORT("bad mmdq config %d", cfg);
    }
#undef MMDQ_CASE
}

template <bool has_ids, int glu = 0>
static void mul_mat_dq_switch_type(
        const ggml_type type, const char * x, const half * y, float * dst, const int ncols_x, const int nrows_x, const int ncols_y,
        const int ncols_y_cfg, const int64_t stride_row_x, const int64_t stride_col_dst, const mmdq_ids_args & ids, cudaStream_t stream,
        const mmdq_epi_args & epi = {}) {
    switch (type) {
        case GGML_TYPE_Q4_K:
            mul_mat_dq_switch<GGML_TYPE_Q4_K, has_ids, glu>(x, y, dst, ncols_x, nrows_x, ncols_y, ncols_y_cfg, stride_row_x, stride_col_dst, ids, epi, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_dq_switch<GGML_TYPE_Q6_K, has_ids, glu>(x, y, dst, ncols_x, nrows_x, ncols_y, ncols_y_cfg, stride_row_x, stride_col_dst, ids, epi, stream);
            break;
        case GGML_TYPE_Q5_0:
            if constexpr (!has_ids) {
                mul_mat_dq_switch<GGML_TYPE_Q5_0, false, glu>(x, y, dst, ncols_x, nrows_x, ncols_y, ncols_y_cfg, stride_row_x, stride_col_dst, ids, epi, stream);
                break;
            }
            GGML_ABORT("unsupported type for mmdq: %s", ggml_type_name(type));
        case GGML_TYPE_Q5_K:
            if constexpr (!glu) {
                mul_mat_dq_switch<GGML_TYPE_Q5_K, has_ids, false>(x, y, dst, ncols_x, nrows_x, ncols_y, ncols_y_cfg, stride_row_x, stride_col_dst, ids, epi, stream);
                break;
            }
            [[fallthrough]];
        case GGML_TYPE_Q8_0:
            if constexpr (!has_ids && !glu) {
                mul_mat_dq_switch<GGML_TYPE_Q8_0, false, false>(x, y, dst, ncols_x, nrows_x, ncols_y, ncols_y_cfg, stride_row_x, stride_col_dst, ids, epi, stream);
                break;
            }
            [[fallthrough]];
        default:
            GGML_ABORT("unsupported type for mmdq: %s", ggml_type_name(type));
    }
}

// checks shared by MUL_MAT and MUL_MAT_ID
static bool mmdq_supported_shape(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, const int cc) {
    if (!GGML_CUDA_CC_IS_RDNA3_5(cc)) {
        return false;
    }
    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            (ggml_prec_get_acc(dst) != GGML_PREC_DEFAULT && ggml_prec_get_acc(dst) != GGML_PREC_F32) ||
            ggml_prec_get_src(dst, 1) != GGML_PREC_DEFAULT ||
            (dst->op == GGML_OP_MUL_MAT && ggml_get_op_params_i32(dst, 1) != 0) ||
            getenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE") != nullptr) {
        return false;
    }
    if (src0->nb[0] != ggml_type_size(src0->type) || !ggml_is_contiguous(src1) || dst->nb[0] != sizeof(float)) {
        return false;
    }
    // Q4_K/Q5_K rows are read with 16 byte loads, Q6_K/Q8_0 rows with 2 byte loads (Q5_1 is checked by the caller)
    if (src0->type != GGML_TYPE_Q6_K && src0->type != GGML_TYPE_Q8_0 && src0->type != GGML_TYPE_Q5_1 && src0->type != GGML_TYPE_Q5_0 && ((uintptr_t) src0->data % 16 != 0 || src0->nb[1] % 16 != 0 || src0->nb[2] % 16 != 0)) {
        return false;
    }
    return src0->ne[1] <= INT_MAX && src1->ne[1]*src1->ne[2] <= INT_MAX;
}

static bool mmdq_supported(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, const int cc) {
    static const bool enabled = mmdq_env_flag("GGML_CUDA_MMDQ");
    return enabled && mmdq_supported_shape(src0, src1, dst, cc);
}

bool ggml_cuda_should_use_mmdq(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc) {
    static const bool no_q6k    = mmdq_env_flag("GGML_CUDA_MMDQ_NO_Q6K");
    // below these batch sizes mmq is as fast for Q4_K; for Q6_K mmq is slower at any batch size
    static const int  min_batch_q4_K = mmdq_env_int("GGML_CUDA_MMDQ_MIN_BATCH", 64);
    static const int  min_batch_q6_K = mmdq_env_int("GGML_CUDA_MMDQ_MIN_BATCH", 32);
    static const bool q8_0           = mmdq_env_flag("GGML_CUDA_MMDQ_Q8_0");
    static const int  min_batch_q8_0 = mmdq_env_int("GGML_CUDA_MMDQ_Q8_0_MIN_BATCH", 576);

    if (!mmdq_supported(src0, src1, dst, cc)) {
        return false;
    }
    static const bool q5_0           = mmdq_env_flag("GGML_CUDA_MMDQ_Q5_0");
    const bool is_q8_0 = src0->type == GGML_TYPE_Q8_0;
    const bool is_q5_0 = src0->type == GGML_TYPE_Q5_0;
    if (src0->type != GGML_TYPE_Q4_K && (src0->type != GGML_TYPE_Q6_K || no_q6k) && (!is_q8_0 || !q8_0) && (!is_q5_0 || !q5_0)) {
        return false;
    }
    // Q8_0/Q5_0 are read in steps of 64 values
    if ((is_q8_0 || is_q5_0) && src0->ne[0] % 64 != 0) {
        return false;
    }
    const int min_batch = is_q8_0 ? min_batch_q8_0 : src0->type == GGML_TYPE_Q4_K ? min_batch_q4_K : min_batch_q6_K;
    if (src1->ne[1] < min_batch) {
        return false;
    }
    // Q8_0: mmq is faster for few output rows
    if (is_q8_0 && (src0->ne[1] < 512 || (src0->ne[1] < 1024 && src1->ne[1] < 1024))) {
        return false;
    }
    // too few output tiles to fill the GPU
    if (src0->ne[1] < 1024 && src1->ne[1] < 512) {
        return false;
    }
    return src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1;
}

static bool mmdq_moe_enabled() {
    static const bool enabled = mmdq_env_flag("GGML_CUDA_MMDQ_MOE");
    return enabled;
}

bool ggml_cuda_should_use_mmdq_id(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst, int cc) {
    static const int  min_batch_moe = mmdq_env_int("GGML_CUDA_MMDQ_MOE_MIN_BATCH", 16);
    static const bool mmdq_moe_all  = mmdq_env_flag("GGML_CUDA_MMDQ_MOE_ALL"); // skip the shape rules, for tests

    const bool moe = mmdq_moe_enabled();
    if (!moe || !mmdq_supported_shape(src0, src1, dst, cc)) {
        return false;
    }
    if (src0->type != GGML_TYPE_Q4_K && src0->type != GGML_TYPE_Q5_K && src0->type != GGML_TYPE_Q6_K && (!moe || src0->type != GGML_TYPE_Q5_1)) {
        return false;
    }
    // Q5_1 is read in steps of 64 values with 8 byte loads
    if (src0->type == GGML_TYPE_Q5_1 && (src0->ne[0] % 64 != 0 || (uintptr_t) src0->data % 8 != 0 || src0->nb[1] % 8 != 0 || src0->nb[2] % 8 != 0)) {
        return false;
    }
    // src1: [K, n_expert_used or 1, n_tokens], dst: [N, n_expert_used, n_tokens], one weight matrix per expert
    if (src1->ne[2] < min_batch_moe || src1->ne[3] != 1 || src0->ne[3] != 1 || src0->ne[2] > MMDQ_MOE_MAX_EXPERTS) {
        return false;
    }
    if (!ggml_is_contiguous(dst) || ids->type != GGML_TYPE_I32 || ids->nb[0] != sizeof(int32_t)) {
        return false;
    }
    if (moe && !mmdq_moe_all) {
        // measured on gfx1151: mmq is as fast or faster with < 16 tokens per expert on average, or < 32 for short K-quant rows
        const int64_t min_tokens_per_expert = src0->type != GGML_TYPE_Q5_1 && src0->ne[0] < 2048 ? 32 : 16;
        if (src1->ne[2]*ids->ne[0] < min_tokens_per_expert*src0->ne[2]) {
            return false;
        }
    }
    return ids->ne[0] == dst->ne[1] && src1->ne[2] == dst->ne[2];
}

// MUL_MAT_ID with the weight streaming kernel
template <bool glu = false>
static void mul_mat_dq_moe_switch(
        const ggml_type type, const char * x, const half * y, float * dst, const int ncols_x, const int nrows_x, const int n_rows,
        const int64_t stride_row_x, const int64_t stride_col_dst, const mmdq_ids_args & ids, cudaStream_t stream,
        const mmdq_epi_args & epi = {}) {
    // up to 96 tokens of one expert per block was the best or close to the best on every measured shape (gfx1151)
    static const int cfg = mmdq_env_int("GGML_CUDA_MMDQ_MOE_CFG", 2);
#define MMDQ_MOE_CASE(NT, NW) \
    switch (type) { \
        case GGML_TYPE_Q4_K: mul_mat_dq_moe_launch<GGML_TYPE_Q4_K, NT, NW, glu>(x, y, dst, ncols_x, nrows_x, n_rows, stride_row_x, stride_col_dst, ids, epi, stream); break; \
        case GGML_TYPE_Q5_K: mul_mat_dq_moe_launch<GGML_TYPE_Q5_K, NT, NW, glu>(x, y, dst, ncols_x, nrows_x, n_rows, stride_row_x, stride_col_dst, ids, epi, stream); break; \
        case GGML_TYPE_Q6_K: mul_mat_dq_moe_launch<GGML_TYPE_Q6_K, NT, NW, glu>(x, y, dst, ncols_x, nrows_x, n_rows, stride_row_x, stride_col_dst, ids, epi, stream); break; \
        case GGML_TYPE_Q5_1: mul_mat_dq_moe_launch<GGML_TYPE_Q5_1, NT, NW, glu>(x, y, dst, ncols_x, nrows_x, n_rows, stride_row_x, stride_col_dst, ids, epi, stream); break; \
        default: GGML_ABORT("unsupported type for mmdq moe: %s", ggml_type_name(type)); \
    }
    switch (cfg) {
        case 1: MMDQ_MOE_CASE(2, 8); break;
        case 2: MMDQ_MOE_CASE(6, 8); break;
        case 3: MMDQ_MOE_CASE(8, 8); break;
        default: GGML_ABORT("bad mmdq moe config %d", cfg);
    }
#undef MMDQ_MOE_CASE
}

void ggml_cuda_mul_mat_dq(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const float * bias, const ggml_tensor * res) {
    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();

    auto & shared = ctx.mmdq_src1;
    const bool use_shared = shared.src1 == src1 && ctx.curr_stream_no == 0;
    ggml_cuda_pool_alloc<half> src1_f16(ctx.pool());
    const half * y;
    if (use_shared) {
        if (!shared.buf) {
            shared.buf = std::make_unique<ggml_cuda_pool_alloc<half>>(ctx.pool(), ne10*ne11);
            ggml_get_to_fp16_cuda(GGML_TYPE_F32)(src1->data, shared.buf->get(), ne10*ne11, stream);
        }
        y = shared.buf->get();
    } else {
        src1_f16.alloc(ne10*ne11);
        ggml_get_to_fp16_cuda(GGML_TYPE_F32)(src1->data, src1_f16.get(), ne10*ne11, stream);
        y = src1_f16.get();
    }

    const mmdq_epi_args epi = {nullptr, nullptr, bias, res ? (const float *) res->data : nullptr, res ? (int64_t) (res->nb[1]/sizeof(float)) : 0};
    mul_mat_dq_switch_type<false>(src0->type, (const char *) src0->data, y, (float *) dst->data, ne00, ne01, ne11, ne11,
        nb01, nb1/sizeof(float), mmdq_ids_args{}, stream, epi);

    if (use_shared && --shared.remaining <= 0) {
        ggml_cuda_mmdq_src1_release(ctx);
    }
}

void ggml_cuda_mmdq_src1_release(ggml_backend_cuda_context & ctx) {
    ctx.mmdq_src1.buf.reset();
    ctx.mmdq_src1.src1      = nullptr;
    ctx.mmdq_src1.remaining = 0;
}

void ggml_cuda_mmdq_src1_share(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const int i) {
    static const bool enabled = mmdq_env_flag("GGML_CUDA_MMDQ_SHARE_SRC1");
    const ggml_tensor * node = cgraph->nodes[i];
    if (!enabled || ctx.mmdq_src1.src1 != nullptr || ctx.curr_stream_no != 0 || node->op != GGML_OP_MUL_MAT) {
        return;
    }
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!ggml_cuda_should_use_mmdq(node->src[0], node->src[1], node, cc)) {
        return;
    }
    // users close by in the graph (attention Q/K/V with their norm/rope nodes in between)
    int n = 0;
    for (int j = i + 1; j < cgraph->n_nodes && j <= i + 32; ++j) {
        const ggml_tensor * t = cgraph->nodes[j];
        if (t->op == GGML_OP_MUL_MAT && t->src[1] == node->src[1] && ggml_cuda_should_use_mmdq(t->src[0], t->src[1], t, cc)) {
            n++;
        }
    }
    if (n > 0) {
        ctx.mmdq_src1.src1      = node->src[1];
        ctx.mmdq_src1.remaining = n + 1;
    }
}

bool ggml_cuda_should_use_mmdq_glu(const ggml_tensor * glu, const ggml_tensor * down, int cc) {
    static const bool enabled = mmdq_env_flag("GGML_CUDA_MMDQ_GLU");
    static const bool geglu_enabled = mmdq_env_flag("GGML_CUDA_MMDQ_GEGLU");
    if (!enabled || glu->op != GGML_OP_GLU || ggml_get_op_params_i32(glu, 1) != 0) {
        return false;
    }
    const ggml_glu_op activation = ggml_get_glu_op(glu);
    if (activation != GGML_GLU_OP_SWIGLU &&
            !(geglu_enabled && activation == GGML_GLU_OP_GEGLU && glu->src[0]->op == GGML_OP_MUL_MAT)) {
        return false;
    }
    const ggml_tensor * gate = glu->src[0];
    const ggml_tensor * up   = glu->src[1];
    if (up == nullptr || gate->op != up->op || down->op != gate->op || down->src[1] != glu ||
            (gate->op != GGML_OP_MUL_MAT && (gate->op != GGML_OP_MUL_MAT_ID || !mmdq_moe_enabled()))) {
        return false;
    }
    const ggml_tensor * w_gate = gate->src[0];
    const ggml_tensor * w_up   = up->src[0];
    if (gate->src[1] != up->src[1] || w_gate->type != w_up->type || w_gate->ne[0] != w_up->ne[0] || w_gate->ne[1] != w_up->ne[1] ||
            w_gate->ne[2] != w_up->ne[2] || w_gate->nb[1] != w_up->nb[1] || w_gate->nb[2] != w_up->nb[2]) {
        return false;
    }
    if (!ggml_are_same_shape(glu, gate) || !ggml_are_same_shape(gate, up) || !ggml_is_contiguous(glu) || !ggml_is_contiguous(gate) || !ggml_is_contiguous(up)) {
        return false;
    }
    if (gate->op == GGML_OP_MUL_MAT) {
        return (w_gate->type == GGML_TYPE_Q4_K || w_gate->type == GGML_TYPE_Q6_K || w_gate->type == GGML_TYPE_Q5_0) &&
            ggml_cuda_should_use_mmdq(w_gate, gate->src[1], gate, cc) && ggml_cuda_should_use_mmdq(w_up, up->src[1], up, cc) &&
            ggml_cuda_should_use_mmdq(down->src[0], glu, down, cc);
    }
    // down reads glu row (token, slot) = token*n_expert_used + slot, which is the ids_dst row of gate/up
    const ggml_tensor * ids = gate->src[2];
    return up->src[2] == ids && down->src[2] == ids && glu->ne[1] == ids->ne[0] && down->src[0]->ne[2] == w_gate->ne[2] &&
        ggml_cuda_should_use_mmdq_id(w_gate, gate->src[1], ids, gate, cc) && ggml_cuda_should_use_mmdq_id(w_up, up->src[1], ids, up, cc) &&
        ggml_cuda_should_use_mmdq_id(down->src[0], glu, ids, down, cc);
}

static void ggml_cuda_mul_mat_id_dq_glu(ggml_backend_cuda_context & ctx, const ggml_tensor * glu, ggml_tensor * down) {
    const ggml_tensor * w_gate = glu->src[0]->src[0];
    const ggml_tensor * w_up   = glu->src[1]->src[0];
    const ggml_tensor * w_down = down->src[0];
    const ggml_tensor * x      = glu->src[0]->src[1];
    const ggml_tensor * ids    = glu->src[0]->src[2];

    const int64_t n_experts     = w_gate->ne[2];
    const int64_t n_expert_used = ids->ne[0];
    const int64_t n_tokens      = x->ne[2];
    const int64_t n_rows        = n_tokens*n_expert_used;
    const int64_t n_ff          = w_gate->ne[1];

    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), n_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), n_rows);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), n_experts + 1);
    ggml_cuda_pool_alloc<int32_t> tile_start(ctx.pool(), n_experts + 1);

    const int si1  = ids->nb[1] / ggml_element_size(ids);
    const int sis1 = x->nb[2] / x->nb[1];
    ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
        n_experts, n_tokens, n_expert_used, x->ne[1], si1, sis1, /*write_inverse =*/ false, stream);
    CUDA_CHECK(cudaGetLastError());

    ggml_cuda_pool_alloc<half> x_f16(ctx.pool(), ggml_nelements(x));
    ggml_get_to_fp16_cuda(GGML_TYPE_F32)(x->data, x_f16.get(), ggml_nelements(x), stream);

    ggml_cuda_pool_alloc<half> h_f16(ctx.pool(), ggml_nelements(glu));
    const mmdq_ids_args args = {ids_src1.get(), ids_dst.get(), expert_bounds.get(), tile_start.get(), (int) n_experts, (int64_t) w_gate->nb[2]};
    mul_mat_dq_moe_switch<true>(w_gate->type, (const char *) w_gate->data, x_f16.get(), nullptr, x->ne[0], n_ff, n_rows,
        w_gate->nb[1], n_ff, args, stream, mmdq_epi_args{(const char *) w_up->data, h_f16.get()});

    const mmdq_ids_args args_down = {ids_dst.get(), ids_dst.get(), expert_bounds.get(), tile_start.get(), (int) n_experts, (int64_t) w_down->nb[2]};
    mul_mat_dq_moe_switch(w_down->type, (const char *) w_down->data, h_f16.get(), (float *) down->data, n_ff, w_down->ne[1], n_rows,
        w_down->nb[1], down->nb[1]/sizeof(float), args_down, stream);
}

void ggml_cuda_mul_mat_dq_glu(ggml_backend_cuda_context & ctx, const ggml_tensor * glu, const ggml_tensor * down, ggml_tensor * dst, const ggml_tensor * res, const half * input_f16) {
    if (down->op == GGML_OP_MUL_MAT_ID) {
        GGML_ASSERT(res == nullptr && dst == down && input_f16 == nullptr);
        ggml_cuda_mul_mat_id_dq_glu(ctx, glu, dst);
        return;
    }
    const ggml_tensor * w_gate = glu->src[0]->src[0];
    const ggml_tensor * w_up   = glu->src[1]->src[0];
    const ggml_tensor * w_down = down->src[0];
    const ggml_tensor * x      = glu->src[0]->src[1];

    const int64_t ne10 = x->ne[0];
    const int64_t ne11 = x->ne[1];
    const int64_t n_ff = w_gate->ne[1];

    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<half> x_f16(ctx.pool());
    if (input_f16 == nullptr) {
        x_f16.alloc(ne10*ne11);
        ggml_get_to_fp16_cuda(GGML_TYPE_F32)(x->data, x_f16.get(), ne10*ne11, stream);
        input_f16 = x_f16.get();
    }

    ggml_cuda_pool_alloc<half> h_f16(ctx.pool(), n_ff*ne11);
    // Compile-time activation keeps the existing SwiGLU store loop unchanged.
    if (ggml_get_glu_op(glu) == GGML_GLU_OP_GEGLU) {
        mul_mat_dq_switch_type<false, 2>(w_gate->type, (const char *) w_gate->data, input_f16, nullptr, ne10, n_ff, ne11, ne11,
            w_gate->nb[1], n_ff, mmdq_ids_args{}, stream, mmdq_epi_args{(const char *) w_up->data, h_f16.get()});
    } else {
        mul_mat_dq_switch_type<false, 1>(w_gate->type, (const char *) w_gate->data, input_f16, nullptr, ne10, n_ff, ne11, ne11,
            w_gate->nb[1], n_ff, mmdq_ids_args{}, stream, mmdq_epi_args{(const char *) w_up->data, h_f16.get()});
    }

    const mmdq_epi_args epi = {nullptr, nullptr, nullptr, res ? (const float *) res->data : nullptr, res ? (int64_t) (res->nb[1]/sizeof(float)) : 0};
    mul_mat_dq_switch_type<false>(w_down->type, (const char *) w_down->data, h_f16.get(), (float *) dst->data, n_ff, w_down->ne[1], ne11, ne11,
        w_down->nb[1], dst->nb[1]/sizeof(float), mmdq_ids_args{}, stream, epi);
}

void ggml_cuda_mul_mat_id_dq(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();

    const int64_t n_expert_used = ids->ne[0];
    const int64_t n_rows        = ne12*n_expert_used; // compact rows, one per (token, used expert)

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), n_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), n_rows);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), ne02 + 1);
    ggml_cuda_pool_alloc<int32_t> tile_start(ctx.pool(), ne02 + 1);

    const int si1  = ids->nb[1] / ggml_element_size(ids);
    const int sis1 = nb12 / nb11;
    ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
        ne02, ne12, n_expert_used, ne11, si1, sis1, /*write_inverse =*/ false, stream);
    CUDA_CHECK(cudaGetLastError());

    // src1 is converted once, the kernel reads its rows through ids_src1
    ggml_cuda_pool_alloc<half> src1_f16(ctx.pool(), ne10*ne11*ne12);
    ggml_get_to_fp16_cuda(GGML_TYPE_F32)(src1->data, src1_f16.get(), ne10*ne11*ne12, stream);

    const mmdq_ids_args args = {ids_src1.get(), ids_dst.get(), expert_bounds.get(), tile_start.get(), (int) ne02, (int64_t) nb02};
    mul_mat_dq_moe_switch(src0->type, (const char *) src0->data, src1_f16.get(), (float *) dst->data, ne00, ne01, n_rows,
        nb01, nb1/sizeof(float), args, stream);
}
