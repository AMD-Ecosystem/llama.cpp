
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout (constant_id =  0) const uint32_t WorkGroupSize = 128;
layout (constant_id =  1) const uint32_t Br = 1;
layout (constant_id =  2) const uint32_t Bc = 32;
layout (constant_id =  3) const uint32_t HSK = 32;
layout (constant_id =  4) const uint32_t HSV = 32;
layout (constant_id =  5) const uint32_t Clamp = 0;
layout (constant_id =  6) const uint32_t D_split = 16;
layout (constant_id =  7) const uint32_t row_split = 1;
layout (constant_id =  8) const uint32_t SubGroupSize = 32;
layout (constant_id =  9) const uint32_t SHMEM_STAGING = 0;
layout (constant_id = 10) const uint32_t Flags = 0;
layout (constant_id = 11) const uint32_t LIMIT_OCCUPANCY_SHMEM = 0;
// ggml_type enumerant for K/V
layout (constant_id = 12) const uint32_t FaTypeK = 0;
layout (constant_id = 13) const uint32_t FaTypeV = 0;
// sizeof(decode buffer): quants -> ggml block size; F32 -> 16 (decodeBufF32 vec4).
layout (constant_id = 14) const uint32_t FaBlockBytesK = 2;
layout (constant_id = 15) const uint32_t FaBlockBytesV = 2;

const bool USE_MASK_OPT    = (Flags & 1) != 0;
const bool MASK_ENABLE     = (Flags & 2) != 0;
const bool LOGIT_SOFTCAP   = (Flags & 4) != 0;
const bool OLD_AMD_WINDOWS = (Flags & 8) != 0;
// Sparse: gather binding-7 indices instead of scanning [0,KV); p.split_kv = n_kv_max.
const bool USE_SPARSE      = (Flags & 16) != 0;
// Rows are token * gqa_ratio + head; N / gqa_ratio query tokens share one K/V head per workgroup.
const bool GQA_MULTI_TOKEN = (Flags & 32) != 0;

// Round up head sizes to a multiple of 16, for coopmat1/coopmat2 paths
const uint32_t HSK_pad = (HSK + 15) & ~15;
const uint32_t HSV_pad = (HSV + 15) & ~15;

const bool KV_bounds_check = Clamp != 0;

layout (push_constant) uniform parameter {
    uint32_t N;
    uint32_t KV;

    uint32_t ne1;
    uint32_t ne2;
    uint32_t ne3;

    uint32_t neq2;
    uint32_t neq3;
    uint32_t nek2;
    uint32_t nek3;
    uint32_t nev2;
    uint32_t nev3;
    uint32_t nem1;
    uint32_t nem2;
    uint32_t nem3;

    uint32_t nb01;
    uint32_t nb02;
    uint32_t nb03;
    uint32_t nb11;
    uint32_t nb12;
    uint32_t nb13;
    uint32_t nb21;
    uint32_t nb22;
    uint32_t nb23;

    float scale;
    float max_bias;
    float logit_softcap;

    uint32_t mask_n_head_log2;
    float m0;
    float m1;

    uint32_t gqa_ratio;
    uint32_t split_kv;
    uint32_t k_num;
} p;

#define SINK_ENABLE_BIT (1<<24)
#define N_LOG2_MASK 0xFFFF

layout (binding = 4) readonly buffer S {float data_s[];};

layout (binding = 5) writeonly buffer O {D_TYPE data_o[];};
layout (binding = 5) writeonly buffer OV4 {D_TYPEV4 data_ov4[];};

layout (binding = 6) readonly buffer MO {uint32_t data_mask_opt[];};

layout (binding = 7) readonly buffer SP {int32_t data_sparse[];};

#define MASK_OPT_ALL_NEG_INF 1
#define MASK_OPT_ALL_ZERO 2

#define BINDING_IDX_K 0
#define BINDING_IDX_V 1

#include "fa_types.glsl"

#if defined(BFLOAT16)
#define O_TYPE float
#define O_TYPEV4 vec4
#else
#define O_TYPE FLOAT_TYPE
#define O_TYPEV4 FLOAT_TYPEV4
#endif

// These can't be `const` globals because GLSL forbids function calls in global
// const initializers, even when the spec constants would let the driver fold
// them. Macros expand at the use site and fold after specialization.
#define BLOCK_SIZE_K fa_block_elems(FaTypeK)
#define BLOCK_SIZE_V fa_block_elems(FaTypeV)
// F16 reads f16 elements directly from the binding; everything else routes
// through dequantize4 / the MMQ helpers to unpack from the packed block layout.
#define USE_DECODE_K (FaTypeK != GGML_TYPE_F16)
#define USE_DECODE_V (FaTypeV != GGML_TYPE_F16)

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))


// Store column zero. This is used to save per-row m and L values for split_k.
ACC_TYPE perElemOpStoreCol0(const in uint32_t r, const in uint32_t c, const in ACC_TYPE elem, const in uint32_t o_offset, const in uint32_t iq2, const in uint32_t N)
{
    if (r < N && c == 0) {
        uint32_t offset = iq2 + r;
        data_o[o_offset + offset] = D_TYPE(elem);
    }
    return elem;
}

// Load the slope matrix, indexed by Q's dimension 2.
ACC_TYPE perElemOpComputeSlope(const in uint32_t r, const in uint32_t c, const in ACC_TYPE elem, const in uint32_t iq2)
{
    const uint32_t h = iq2 + (r % p.gqa_ratio);

    uint32_t n_head_log2 = p.mask_n_head_log2 & N_LOG2_MASK;

    const ACC_TYPE base = ACC_TYPE(h < n_head_log2 ? p.m0 : p.m1);
    const int      exph = int(h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1);

    return ACC_TYPE(pow(base, ACC_TYPE(exph)));
}

// Load the sink value, indexed by Q's dimension 2.
ACC_TYPE perElemOpGetSink(const in uint32_t r, const in uint32_t c, const in ACC_TYPE elem, const in uint32_t iq2)
{
    const uint32_t h = iq2 + (r % p.gqa_ratio);

    return ACC_TYPE(data_s[h]);
}

uint32_t i, N, KV, split_k_index, Tr, start_j, end_j,
         gqa_iq1, iq2, iq3, rk2, rk3, rv2, rv3, ik2, ik3, iv2, iv3,
         q_stride, k_stride, v_stride, m_stride, sparse_base, gqa_tokens, gqa_valid_tokens;

void init_indices()
{
    N = p.N;
    KV = p.KV;

    if (p.k_num > 1) {
        if (p.gqa_ratio > 1) {
            i = 0;
            // batch and split_k share gl_WorkGroupID.x
            gqa_iq1 = gl_WorkGroupID.x / p.k_num;
            split_k_index = gl_WorkGroupID.x % p.k_num;
        } else {
            gqa_iq1 = 0;
            split_k_index = gl_WorkGroupID.x % p.k_num;
            i = gl_WorkGroupID.x / p.k_num;
        }
    } else if (p.gqa_ratio > 1) {
        i = 0;
        gqa_iq1 = gl_WorkGroupID.x;
        split_k_index = 0;
    } else {
        i = gl_WorkGroupID.x;
        gqa_iq1 = 0;
        split_k_index = 0;
    }

    // gqa_iq1 is the first query token of the workgroup.
    gqa_tokens = 1;
    gqa_valid_tokens = 1;
    if (GQA_MULTI_TOKEN) {
        gqa_tokens = p.N / p.gqa_ratio;
        gqa_iq1 *= gqa_tokens;
        // the last token group may be partial (p.ne2 is the number of query tokens)
        gqa_valid_tokens = min(gqa_tokens, p.ne2 - gqa_iq1);
    }

    Tr = CEIL_DIV(N, Br);

    start_j = split_k_index * p.split_kv / Bc;
    end_j = CEIL_DIV(min(KV, (split_k_index + 1) * p.split_kv), Bc);

    // When not using grouped query attention, all rows share the same iq2, equal to gl_WorkGroupID.y.
    // When using grouped query attention, each workgroup does gqa_ratio consecutive values of iq2.
    iq2 = gl_WorkGroupID.y * p.gqa_ratio;
    iq3 = gl_WorkGroupID.z;

    // broadcast factors
    rk2 = p.neq2/p.nek2;
    rk3 = p.neq3/p.nek3;

    rv2 = p.neq2/p.nev2;
    rv3 = p.neq3/p.nev3;

    // k indices
    ik3 = iq3 / rk3;
    ik2 = iq2 / rk2;

    // v indices
    iv3 = iq3 / rv3;
    iv2 = iq2 / rv2;

    // nb?1 are already divided by the type size and are in units of elements.
    // When using grouped query attention, Q is indexed by iq2, so the stride
    // should be nb02 (which is in bytes).
    q_stride = p.gqa_ratio > 1 ? (p.nb02 / 4) : p.nb01;
    k_stride = p.nb11;
    v_stride = p.nb21;
    // When using grouped query attention, all rows use the same mask (stride 0).
    // "p.gqa_ratio >> 16" is just a roundabout way of writing zero
    // that prevents the compiler from folding the "&" through the select
    // and breaking the alignment detection.
    m_stride = (p.gqa_ratio > 1) ? (p.gqa_ratio >> 16) : KV;

    // Sparse: the tile shares one mask row (gqa heads, or Br==1). split_k
    // partitions the n_kv_max blocks.
    if (USE_SPARSE) {
        uint32_t qrow = (p.gqa_ratio > 1) ? gqa_iq1 : (i * Br);
        sparse_base = (((iq3 % p.nem3) * p.nem2 + (iq2 % p.nem2)) * p.nem1 + qrow) * p.split_kv;

        uint32_t total_blocks = CEIL_DIV(p.split_kv, Bc);
        uint32_t per_blocks   = CEIL_DIV(total_blocks, p.k_num);
        start_j = min(split_k_index * per_blocks, total_blocks);
        end_j   = min((split_k_index + 1) * per_blocks, total_blocks);
    }
}

// Select KV blocks that are not fully masked. Only code 1 (ALL_NEG_INF) has its low bit set.
uint32_t mask_opt_active_bits(const in uint32_t word, const in uint32_t w, const in uint32_t n_blocks)
{
    const uint32_t n_valid = min(16u, n_blocks - w * 16);
    const uint32_t valid = n_valid == 16 ? 0x55555555u : (((1u << (2 * n_valid)) - 1u) & 0x55555555u);
    return valid & ~(word & 0x55555555u);
}

// Balance splits over visible KV blocks; splitting the full cache leaves most sliding-window splits idle.
void mask_opt_balance_split_k(const in uint32_t mo_offset, const in uint32_t mo_stride)
{
    const uint32_t n_blocks = CEIL_DIV(KV, Bc);
    uint32_t n_active = 0;
    for (uint32_t w = 0; w < mo_stride; ++w) {
        n_active += bitCount(mask_opt_active_bits(data_mask_opt[mo_offset + w], w, n_blocks));
    }
    const uint32_t per_split = CEIL_DIV(n_active, p.k_num);
    const uint32_t first = split_k_index * per_split;
    const uint32_t last = min(first + per_split, n_active);

    start_j = 0;
    end_j = 0;
    if (first >= last) {
        return;
    }
    uint32_t seen = 0;
    for (uint32_t w = 0; w < mo_stride && seen < last; ++w) {
        const uint32_t bits = mask_opt_active_bits(data_mask_opt[mo_offset + w], w, n_blocks);
        const uint32_t count = bitCount(bits);
        const bool has_first = seen <= first && first < seen + count;
        const bool has_last = seen < last && last <= seen + count;
        if (!has_first && !has_last) {
            seen += count;
            continue;
        }
        for (uint32_t b = 0; b < 16; ++b) {
            if ((bits & (1u << (2 * b))) == 0) {
                continue;
            }
            if (seen == first) {
                start_j = w * 16 + b;
            }
            seen++;
            if (seen == last) {
                end_j = w * 16 + b + 1;
                break;
            }
        }
    }
}

// Resolve a linear KV slot to a real column; false for inactive (sparse padding/-1, or dense OOB).
bool fa_kv_index(uint lin, out uint kv_col) {
    if (USE_SPARSE) {
        if (lin >= p.split_kv) {
            kv_col = 0;
            return false;
        }
        int idx = data_sparse[sparse_base + lin];
        kv_col = idx >= 0 ? uint(idx) : 0;
        return idx >= 0;
    }
    kv_col = lin;
    return !KV_bounds_check || lin < KV;
}

// Bias applied to softmax to stay in fp16 range.
// Based on ggml-cuda issue https://github.com/ggml-org/llama.cpp/issues/18606
const float FATTN_KQ_MAX_OFFSET = 3.0f*0.6931f;

// Query token (relative to gqa_iq1) and head (relative to iq2) of a GQA row.
uint32_t gqa_row_token(const in uint32_t r)
{
    return GQA_MULTI_TOKEN ? r / p.gqa_ratio : 0;
}

uint32_t gqa_row_head(const in uint32_t r)
{
    return GQA_MULTI_TOKEN ? r % p.gqa_ratio : r;
}

// False for rows of a partial last token group that have no query token.
bool gqa_row_valid(const in uint32_t r)
{
    return !GQA_MULTI_TOKEN || gqa_row_token(r) < gqa_valid_tokens;
}

// Offset of a Q row from q_offset, in floats.
uint32_t q_row_offset(const in uint32_t r)
{
    return gqa_row_token(r) * p.nb01 + gqa_row_head(r) * q_stride;
}

// Offset of the mask row used by Q row r, from m_offset. Padding rows reuse the last token.
uint32_t mask_row_offset(const in uint32_t r)
{
    if (GQA_MULTI_TOKEN) {
        return min(gqa_row_token(r), gqa_valid_tokens - 1) * KV;
    }
    return r * m_stride;
}

// token_stride is the query-token stride in vec4 units for O, or elements for split_k L/M.
void gqa_store_row(const in uint32_t r, const in uint32_t c, const in O_TYPEV4 elems, const in uint32_t o_offset, const in uint32_t token_stride)
{
    if (!gqa_row_valid(r)) {
        return;
    }
    const uint32_t offset = (iq2 + gqa_row_head(r)) * HSV / 4 + c;
    data_ov4[o_offset + gqa_row_token(r) * token_stride + offset] = D_TYPEV4(elems);
}

void gqa_store_col0_row(const in uint32_t r, const in ACC_TYPE elem, const in uint32_t o_offset, const in uint32_t token_stride)
{
    if (!gqa_row_valid(r)) {
        return;
    }
    data_o[o_offset + gqa_row_token(r) * token_stride + iq2 + gqa_row_head(r)] = D_TYPE(elem);
}
