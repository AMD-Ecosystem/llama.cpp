// MMQ autotuning core - compiled only with -DGGML_MMQ_TUNE=ON.
//
// Mirrors both branches of ggml_cuda_mul_mat_q in mmq.cu (dense and MoE) but dispatches to
// launch_mul_mat_q<type, mmq_x> with an explicit mmq_x, so every tile width can be measured for
// a given shape instead of only the one mul_mat_q_case would pick.

#include "ggml-mmq-tune.h"

#include "common.cuh"
#include "mmid.cuh"
#include "mmq.cuh"
#include "quantize.cuh"

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

#if defined(GGML_USE_HIP)
// vendors/hip.h aliases most of the event API but not these two.
#define cudaEventCreate      hipEventCreate
#define cudaEventElapsedTime hipEventElapsedTime
#endif

int ggml_mmq_tune_mmq_y(void) {
    return get_mmq_y_host(ggml_cuda_info().devices[ggml_cuda_get_device()].cc);
}

int ggml_mmq_tune_nwarps(void) {
    const int id = ggml_cuda_get_device();
    return mmq_get_nwarps_host(ggml_cuda_info().devices[id].cc, ggml_cuda_info().devices[id].warp_size);
}

#define MMQ_TUNE_RED_BLOCKS 64

// Position-weighted checksum of dst, reduced per block and summed on the host. Configs that
// disagree on it computed something different; a majority vote over the widths then says which
// ones are wrong. The index weight is what makes a misplaced row visible - a plain sum of |dst|
// is permutation invariant.
static __global__ void mmq_tune_checksum(const float * dst, const int64_t n, double * out) {
    __shared__ double s[256];
    double local = 0.0;
    for (int64_t i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += (int64_t) gridDim.x*blockDim.x) {
        local += fabs((double) dst[i]) * (double) (i % 251 + 1);
    }
    s[threadIdx.x] = local;
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            s[threadIdx.x] += s[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[blockIdx.x] = s[0];
    }
}

// Draws top_k distinct experts per token from a Zipf distribution over a shuffled expert order,
// so the hot experts are not always the low indices. Measured routing in MoE models follows
// Zipf with s around 2.
static std::vector<int32_t> mmq_tune_build_ids(int n_tokens, int top_k, int n_experts, float zipf_s) {
    std::mt19937 rng(5678);

    std::vector<int> order(n_experts);
    for (int e = 0; e < n_experts; ++e) {
        order[e] = e;
    }
    std::shuffle(order.begin(), order.end(), rng);

    std::vector<double> cdf(n_experts);
    double acc = 0.0;
    for (int r = 0; r < n_experts; ++r) {
        acc += zipf_s > 0.0f ? 1.0/pow(r + 1.0, (double) zipf_s) : 1.0;
        cdf[r] = acc;
    }
    for (double & c : cdf) {
        c /= acc;
    }

    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::vector<int32_t> ids((size_t) top_k*n_tokens);

    for (int t = 0; t < n_tokens; ++t) {
        for (int j = 0; j < top_k; ++j) {
            int e;
            bool dup;
            int guard = 0;
            do {
                const int rank = std::lower_bound(cdf.begin(), cdf.end(), uni(rng)) - cdf.begin();
                e = order[std::min(rank, n_experts - 1)];
                dup = false;
                for (int p = 0; p < j; ++p) {
                    dup |= ids[(size_t) t*top_k + p] == e;
                }
                // A heavily skewed draw collides often; fall back to a linear probe rather than
                // spinning, which would flatten the distribution we are trying to reproduce.
                if (dup && ++guard > 64) {
                    for (int cand = 0; cand < n_experts; ++cand) {
                        bool used = false;
                        for (int p = 0; p < j; ++p) {
                            used |= ids[(size_t) t*top_k + p] == cand;
                        }
                        if (!used) {
                            e = cand;
                            dup = false;
                            break;
                        }
                    }
                }
            } while (dup);
            ids[(size_t) t*top_k + j] = e;
        }
    }
    return ids;
}

template <ggml_type type>
static bool mmq_tune_launch(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream, const int mmq_x) {
    switch (mmq_x) {
        case   8: launch_mul_mat_q<type,   8>(ctx, args, stream); return true;
        case  16: launch_mul_mat_q<type,  16>(ctx, args, stream); return true;
        case  24: launch_mul_mat_q<type,  24>(ctx, args, stream); return true;
        case  32: launch_mul_mat_q<type,  32>(ctx, args, stream); return true;
        case  40: launch_mul_mat_q<type,  40>(ctx, args, stream); return true;
        case  48: launch_mul_mat_q<type,  48>(ctx, args, stream); return true;
        case  56: launch_mul_mat_q<type,  56>(ctx, args, stream); return true;
        case  64: launch_mul_mat_q<type,  64>(ctx, args, stream); return true;
        case  72: launch_mul_mat_q<type,  72>(ctx, args, stream); return true;
        case  80: launch_mul_mat_q<type,  80>(ctx, args, stream); return true;
        case  88: launch_mul_mat_q<type,  88>(ctx, args, stream); return true;
        case  96: launch_mul_mat_q<type,  96>(ctx, args, stream); return true;
        case 104: launch_mul_mat_q<type, 104>(ctx, args, stream); return true;
        case 112: launch_mul_mat_q<type, 112>(ctx, args, stream); return true;
        case 120: launch_mul_mat_q<type, 120>(ctx, args, stream); return true;
        case 128: launch_mul_mat_q<type, 128>(ctx, args, stream); return true;
        case GGML_MMQ_TUNE_MMQ_X_AUTO: mul_mat_q_case<type>(ctx, args, stream); return true;
        default: return false;
    }
}

// Same three rejection criteria the heuristic loop in mul_mat_q_case applies.
template <ggml_type type>
static bool mmq_tune_valid(const int mmq_x, const int cc, const int warp_size, const int nwarps,
                           const int mmq_y, const size_t smpbo) {
    if (mmq_x == GGML_MMQ_TUNE_MMQ_X_AUTO) {
        return true;
    }
    if (mmq_x < 8 || mmq_x % 8 != 0 || mmq_x > get_mmq_x_max_host(cc)) {
        return false;
    }
    if (mmq_x % mmq_get_granularity_host(mmq_x, cc) != 0) {
        return false;
    }
    return mmq_get_nbytes_shared<type>(mmq_x, mmq_y, cc, warp_size, nwarps) <= smpbo;
}

// Distinct weights for every expert, generated and uploaded one [K,N] slice at a time: building
// the whole f32 tensor at once would need gigabytes of host memory for the larger MoE shapes,
// and reusing a few slices would shrink the weight working set enough to make the measurement
// unrealistically cache-friendly.
template <ggml_type type>
static void mmq_tune_fill_weights(char * dst_d, int64_t k, int64_t n, int n_experts, cudaStream_t stream) {
    const size_t slice_bytes = ggml_row_size(type, k)*n;

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> f32(k*n);
    std::vector<char>  q(slice_bytes);

    for (int e = 0; e < n_experts; ++e) {
        for (float & v : f32) {
            v = dist(rng);
        }
        ggml_quantize_chunk(type, f32.data(), q.data(), 0, n, k, nullptr);
        CUDA_CHECK(cudaMemcpyAsync(dst_d + slice_bytes*e, q.data(), slice_bytes,
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream)); // q is reused for the next expert
    }
}

template <ggml_type type>
static bool mmq_tune_sweep_impl(const ggml_mmq_tune_case & tc, const int * mmq_x_list, const int n_mmq_x,
                                ggml_mmq_tune_point * out) {
    const bool    moe   = tc.n_experts > 0;
    const int64_t ne00  = tc.k;
    const int64_t ne01  = tc.n;
    const int64_t ne02  = moe ? tc.n_experts : 1;
    const int64_t top_k = moe ? tc.top_k : 1;
    // Dense: src1 is [K, m]. MoE: src1 is [K, top_k, m] and dst is [N, top_k, m].
    const int64_t ne11  = moe ? top_k : tc.m;
    const int64_t ne12  = moe ? tc.m : 1;

    if (ne00 % ggml_blck_size(type) != 0 || (moe && (tc.top_k < 1 || tc.top_k > tc.n_experts))) {
        return false;
    }

    const int id        = ggml_cuda_get_device();
    const int cc        = ggml_cuda_info().devices[id].cc;
    const int warp_size = ggml_cuda_info().devices[id].warp_size;
    const size_t smpbo  = ggml_cuda_info().devices[id].smpbo;
    const int nwarps    = mmq_get_nwarps_host(cc, warp_size);
    const int mmq_y     = get_mmq_y_host(cc);

    ggml_backend_cuda_context ctx(id);
    cudaStream_t stream = ctx.stream();

    const size_t x_row_size = ggml_row_size(type, ne00);
    const int64_t n_dst     = ne01*ne11*ne12;

    ggml_cuda_pool_alloc<char>  src0_d(ctx.pool(), x_row_size*ne01*ne02);
    ggml_cuda_pool_alloc<float> src1_d(ctx.pool(), ne00*ne11*ne12);
    ggml_cuda_pool_alloc<float>  dst_d(ctx.pool(), n_dst);
    ggml_cuda_pool_alloc<double> checksum_d(ctx.pool(), MMQ_TUNE_RED_BLOCKS);

    mmq_tune_fill_weights<type>(src0_d.get(), ne00, ne01, ne02, stream);

    {
        std::mt19937 rng(4321);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> y(ne00*ne11*ne12);
        for (float & v : y) {
            v = dist(rng);
        }
        CUDA_CHECK(cudaMemcpyAsync(src1_d.get(), y.data(), y.size()*sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    const int64_t ne10_padded = GGML_PAD(ne00, MATRIX_ROW_PADDING);
    const int64_t s01 = x_row_size / ggml_type_size(type);
    const int64_t s02 = ne01*s01;
    const int64_t s1  = ne01;

    const bool use_stream_k = (GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA)
                            || GGML_CUDA_CC_IS_CDNA(cc);

    // Kept alive for the whole sweep; mmq_args holds raw pointers into them.
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool());
    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool());
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool());
    ggml_cuda_pool_alloc<int32_t> ids_d(ctx.pool());
    ggml_cuda_pool_alloc<char>    src1_q8_1(ctx.pool());

    mmq_args args = {};

    if (!moe) {
        const size_t nbytes = ne11*ne10_padded*sizeof(block_q8_1)/QK8_1 +
            get_mmq_x_max_host(cc)*sizeof(block_q8_1_mmq);
        src1_q8_1.alloc(nbytes);

        quantize_mmq_q8_1_cuda(src1_d.get(), nullptr, src1_q8_1.get(), type,
                               ne00, ne00, ne00*ne11, ne00*ne11, ne10_padded, ne11, 1, 1, stream);
        CUDA_CHECK(cudaGetLastError());

        const int64_t s12 = ne11*ne10_padded*sizeof(block_q8_1) / (QK8_1*sizeof(int));

        args = {
            src0_d.get(), type, (const int *) src1_q8_1.get(), nullptr, nullptr, dst_d.get(),
            ne00, ne01, ne11, s01, ne11, s1,
            1, 1, s02, s12, ne01*ne11,
            1, 1, s02, s12, ne01*ne11,
            use_stream_k, ne11};
    } else {
        const int64_t n_expert_used = top_k;
        const int64_t ne_get_rows   = ne12*n_expert_used;

        const std::vector<int32_t> ids_h = mmq_tune_build_ids(ne12, n_expert_used, ne02, tc.zipf_s);
        ids_d.alloc(ids_h.size());
        CUDA_CHECK(cudaMemcpyAsync(ids_d.get(), ids_h.data(), ids_h.size()*sizeof(int32_t),
                                   cudaMemcpyHostToDevice, stream));

        ids_src1.alloc(ne_get_rows);
        ids_dst.alloc(ne_get_rows);
        expert_bounds.alloc(ne02 + 1);

        // ids is [top_k, m] contiguous, src1 is [K, top_k, m]: si1 = sis1 = top_k.
        ggml_cuda_launch_mm_ids_helper(ids_d.get(), ids_src1.get(), ids_dst.get(), expert_bounds.get(),
                                       ne02, ne12, n_expert_used, ne11, n_expert_used, n_expert_used, stream);
        CUDA_CHECK(cudaGetLastError());

        const size_t nbytes = ne12*n_expert_used*ne10_padded*sizeof(block_q8_1)/QK8_1 +
            get_mmq_x_max_host(cc)*sizeof(block_q8_1_mmq);
        src1_q8_1.alloc(nbytes);

        quantize_mmq_q8_1_cuda(src1_d.get(), ids_src1.get(), src1_q8_1.get(), type,
                               ne00, ne00, ne00*ne11, ne00*ne11*ne12, ne10_padded,
                               ne12*n_expert_used, 1, 1, stream);
        CUDA_CHECK(cudaGetLastError());

        const int64_t s12 = ne11*ne10_padded*sizeof(block_q8_1) / (QK8_1*sizeof(int));

        args = {
            src0_d.get(), type, (const int *) src1_q8_1.get(), ids_dst.get(), expert_bounds.get(), dst_d.get(),
            ne00, ne01, ne_get_rows, s01, ne_get_rows, s1,
            ne02, ne02, s02, s12, ne01*ne11,
            1, 1, ne02*s02, ne12*s12, ne01*ne11*ne12,
            use_stream_k, ne12};
    }

    // ---- sweep ------------------------------------------------------------
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    std::vector<std::vector<double>> samples(n_mmq_x);

    for (int i = 0; i < n_mmq_x; ++i) {
        const int mmq_x = mmq_x_list[i];
        out[i] = { mmq_x, mmq_tune_valid<type>(mmq_x, cc, warp_size, nwarps, mmq_y, smpbo), 0.0, 0.0, 0.0 };

        if (!out[i].valid) {
            continue;
        }
        samples[i].reserve(tc.niter);
        for (int w = 0; w < tc.nwarmup; ++w) {
            mmq_tune_launch<type>(ctx, args, stream, mmq_x);
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    // One timed iteration per config per pass, so clock drift and background load spread across
    // all configs instead of penalising whichever ones happen to run late.
    for (int pass = 0; pass < tc.niter; ++pass) {
        for (int i = 0; i < n_mmq_x; ++i) {
            if (!out[i].valid) {
                continue;
            }
            CUDA_CHECK(cudaEventRecord(ev_start, stream));
            mmq_tune_launch<type>(ctx, args, stream, mmq_x_list[i]);
            CUDA_CHECK(cudaEventRecord(ev_stop, stream));
            CUDA_CHECK(cudaEventSynchronize(ev_stop));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
            samples[i].push_back(ms*1000.0);
        }
    }
    CUDA_CHECK(cudaGetLastError());

    for (int i = 0; i < n_mmq_x; ++i) {
        if (!out[i].valid) {
            continue;
        }

        // Re-run alone so dst holds only this config's output. MoE writes just the routed rows,
        // so clear it first or rows left by the previous config would hide a mismatch.
        CUDA_CHECK(cudaMemsetAsync(dst_d.get(), 0, n_dst*sizeof(float), stream));
        mmq_tune_launch<type>(ctx, args, stream, mmq_x_list[i]);
        mmq_tune_checksum<<<MMQ_TUNE_RED_BLOCKS, 256, 0, stream>>>(dst_d.get(), n_dst, checksum_d.get());

        double partial[MMQ_TUNE_RED_BLOCKS];
        CUDA_CHECK(cudaMemcpyAsync(partial, checksum_d.get(), sizeof(partial), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::sort(samples[i].begin(), samples[i].end());
        out[i].us_median = samples[i][tc.niter/2];
        out[i].us_min    = samples[i][0];
        out[i].checksum  = std::accumulate(partial, partial + MMQ_TUNE_RED_BLOCKS, 0.0);
    }

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    return true;
}

bool ggml_mmq_tune_sweep(const ggml_mmq_tune_case * tcase, const int * mmq_x_list, const int n_mmq_x,
                         ggml_mmq_tune_point * out) {
    if (tcase->niter < 1 || n_mmq_x < 1) {
        return false;
    }

    switch (tcase->type) {
        case GGML_TYPE_Q8_0: return mmq_tune_sweep_impl<GGML_TYPE_Q8_0>(*tcase, mmq_x_list, n_mmq_x, out);
        case GGML_TYPE_Q5_0: return mmq_tune_sweep_impl<GGML_TYPE_Q5_0>(*tcase, mmq_x_list, n_mmq_x, out);
        case GGML_TYPE_Q5_1: return mmq_tune_sweep_impl<GGML_TYPE_Q5_1>(*tcase, mmq_x_list, n_mmq_x, out);
        case GGML_TYPE_Q4_K: return mmq_tune_sweep_impl<GGML_TYPE_Q4_K>(*tcase, mmq_x_list, n_mmq_x, out);
        case GGML_TYPE_Q5_K: return mmq_tune_sweep_impl<GGML_TYPE_Q5_K>(*tcase, mmq_x_list, n_mmq_x, out);
        case GGML_TYPE_Q6_K: return mmq_tune_sweep_impl<GGML_TYPE_Q6_K>(*tcase, mmq_x_list, n_mmq_x, out);
        default:
            fprintf(stderr, "%s: unsupported type %s\n", __func__, ggml_type_name(tcase->type));
            return false;
    }
}
