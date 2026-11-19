#include "argsort.cuh"
#include "top-k.cuh"

// Adjusted implementation thresholds from #28547, can be overridden at build time
#ifndef GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC
#    if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
// not measured on HIP/MUSA, keep the old split
#        define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC 1024
#    else
#        define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC 512
#    endif
#endif // GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC

#ifndef GGML_CUDA_TOP_K_NCOLS_THRESHOLD_ARGSORT
#    define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_ARGSORT 4096
#endif // GGML_CUDA_TOP_K_NCOLS_THRESHOLD_ARGSORT

// bitonic up to this width while nrows fits in one wave of SMs, 0 disables
#ifndef GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS
#    if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
#        define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS 0
#    else
#        define GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS 1024
#    endif
#endif // GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
// DeviceTopK has a race condition before CCCL 3.4.3.
// https://github.com/NVIDIA/cccl/pull/10627
#    if (CCCL_MAJOR_VERSION > 3 || \
         (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION > 4) || \
         (CCCL_MAJOR_VERSION == 3 && CCCL_MINOR_VERSION == 4 && CCCL_PATCH_VERSION >= 3))
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL >= 3.4.3
#endif      // GGML_CUDA_USE_CUB

// max rows for the per-row DeviceTopK / CUB argsort path before switching to radix / bitonic
#ifndef GGML_CUDA_TOP_K_NROWS_THRESHOLD
#    ifdef CUB_TOP_K_AVAILABLE
#        define GGML_CUDA_TOP_K_NROWS_THRESHOLD 2
#    else
#        define GGML_CUDA_TOP_K_NROWS_THRESHOLD 1
#    endif
#endif // GGML_CUDA_TOP_K_NROWS_THRESHOLD

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#endif                            // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

static __device__ __forceinline__ uint32_t top_k_float_to_ordered(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t mask = (uint32_t) (-(int32_t) (bits >> 31)) | 0x80000000U;
    return bits ^ mask;
}

// one state per cache line, so the gather atomics of different rows do not compete
struct alignas(128) top_k_radix_state {
    uint32_t prefix;
    uint32_t prefix_mask;
    int rank;
    int greater_count;
};

static __global__ void top_k_radix_init(top_k_radix_state * states, int nrows, int k) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row] = {0, 0, k, 0};
    }
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_histogram(
        const float * __restrict__ src,
        const top_k_radix_state * __restrict__ states,
        int * __restrict__ block_histograms,
        int ncols,
        int block_cols,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    const int64_t col_begin = (int64_t) row_block * block_cols;
    const int64_t col_end   = min(col_begin + block_cols, (int64_t) ncols);
    __shared__ int histogram[NBINS];

    histogram[tid] = 0;
    __syncthreads();

    const top_k_radix_state state = states[row];
    for (int64_t col = col_begin + tid; col < col_end; col += BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if ((key & state.prefix_mask) == state.prefix) {
            atomicAdd(&histogram[(key >> shift) & (NBINS - 1)], 1);
        }
    }
    __syncthreads();

    const size_t histogram_offset =
        ((size_t) row * blocks_per_row + row_block) * NBINS;
    block_histograms[histogram_offset + tid] = histogram[tid];
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_select(
        const int * __restrict__ block_histograms,
        top_k_radix_state * __restrict__ states,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int histogram[NBINS];

    int count = 0;
    for (int row_block = 0; row_block < blocks_per_row; ++row_block) {
        const size_t offset = ((size_t) row * blocks_per_row + row_block) * NBINS;
        count += block_histograms[offset + tid];
    }
    histogram[tid] = count;
    __syncthreads();

    if (tid == 0) {
        top_k_radix_state state = states[row];
        int bin = NBINS - 1;
        while (bin > 0 && histogram[bin] < state.rank) {
            state.rank -= histogram[bin--];
        }
        state.prefix |= (uint32_t) bin << shift;
        state.prefix_mask |= (uint32_t) (NBINS - 1) << shift;
        states[row] = state;
    }
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_gather(
        const float * __restrict__ src,
        int * __restrict__ dst,
        const int * __restrict__ block_histograms,
        top_k_radix_state * __restrict__ states,
        int ncols,
        int block_cols,
        int k,
        int blocks_per_row) {
    constexpr int NBINS     = 1 << RADIX_BITS;
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = BLOCK_SIZE / warp_size;
    using lane_mask_t = std::conditional_t<warp_size == 64, uint64_t, uint32_t>;
    // HIP requires a 64-bit mask also for a 32-wide warp
    constexpr uint64_t ballot_mask = ~(lane_mask_t) 0;
    const auto popc = [](const lane_mask_t mask) {
        if constexpr (warp_size == 64) {
            return __popcll(mask);
        } else {
            return __popc(mask);
        }
    };
    __shared__ int warp_ties[nwarps];

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int lane = threadIdx.x % warp_size;
    const int warp = threadIdx.x / warp_size;
    const float * row_src = src + (size_t) row * ncols;
    int * row_dst = dst + (size_t) row * k;
    top_k_radix_state * state = &states[row];
    const uint32_t threshold = state->prefix;
    const int rank = state->rank;

    // each warp reads a contiguous part of the contiguous range of the block
    const int     warp_cols  = block_cols / nwarps;
    const int64_t warp_begin = (int64_t) row_block * block_cols + warp * warp_cols;
    const int64_t warp_end   = min(warp_begin + warp_cols, (int64_t) ncols);

    int nties = 0;
    for (int64_t col = warp_begin + lane; col < warp_end; col += warp_size) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if (key > threshold) {
            const int pos = atomicAdd(&state->greater_count, 1);
            row_dst[pos] = col;
        } else if (key == threshold) {
            nties++;
        }
    }

    // the ties fill the last rank slots in column order, the lowest columns win.
    // the histograms of the last pass count the ties of each block in the bin of the threshold.
    const int * block_ties = block_histograms + (size_t) row * blocks_per_row * NBINS + (threshold & (NBINS - 1));
    int pos = 0;
    for (int b = lane; b < row_block; b += warp_size) {
        pos += block_ties[b * NBINS];
    }
    pos = warp_reduce_sum<warp_size>(pos);
    if (pos >= rank || block_ties[row_block * NBINS] == 0) {
        return; // the same result in all threads of the block
    }

    nties = warp_reduce_sum<warp_size>(nties);
    if (lane == 0) {
        warp_ties[warp] = nties;
    }
    __syncthreads();
    for (int w = 0; w < warp; ++w) {
        pos += warp_ties[w];
    }

    // pos is the same in all lanes of the warp, so the lanes leave the loop together
    const int pos_end = min(pos + nties, rank);
    for (int64_t tile = warp_begin; tile < warp_end && pos < pos_end; tile += warp_size) {
        const int64_t col    = tile + lane;
        const bool    is_tie = col < warp_end && top_k_float_to_ordered(row_src[col]) == threshold;

        const lane_mask_t ties    = (lane_mask_t) __ballot_sync(ballot_mask, is_tie);
        const int         tie_pos = pos + popc(ties & (((lane_mask_t) 1 << lane) - 1));
        if (is_tie && tie_pos < pos_end) {
            row_dst[k - rank + tie_pos] = col;
        }
        pos += popc(ties);
    }
}

#if defined(GGML_USE_HIP)

// one block per row, the row stays in registers: a radix histogram of 4 passes of 8 bits over the keys that match the prefix
// found so far gives the k-th largest key T, then the keys above T and the lowest columns of the keys equal to T are written.
// Same selected set as the radix path below (the lowest columns win ties), but one launch instead of 10.
// The histogram has one copy per 4 lanes to keep the shared atomics of the dense first pass apart.
template<int BLOCK_SIZE, int MPT>
static __global__ void __launch_bounds__(BLOCK_SIZE) top_k_block_kernel(
        const float * __restrict__ src, int * __restrict__ dst, const int ncols, const int k) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = BLOCK_SIZE / warp_size;
    constexpr int NCOPY     = 8;
    constexpr int BPL       = 256 / warp_size; // histogram bins per lane
    using lane_mask_t = std::conditional_t<warp_size == 64, uint64_t, uint32_t>;
    constexpr uint64_t ballot_mask = ~(lane_mask_t) 0;
    const auto popc = [](const lane_mask_t mask) {
        if constexpr (warp_size == 64) {
            return __popcll(mask);
        } else {
            return __popc(mask);
        }
    };

    const int tid  = threadIdx.x;
    const int lane = tid % warp_size;
    const int warp = tid / warp_size;

    const float * row_src = src + (size_t) blockIdx.x * ncols;
    int * row_dst = dst + (size_t) blockIdx.x * k;

    uint32_t key[MPT];
#pragma unroll
    for (int j = 0; j < MPT; ++j) {
        const int i = tid + j*BLOCK_SIZE;
        key[j] = i < ncols ? top_k_float_to_ordered(row_src[i]) : 0u;
    }

    __shared__ int hist[NCOPY][256 + 1]; // the odd row length puts the copies in different banks
    __shared__ int sel[3];
    __shared__ int counters[2];
    __shared__ int scan[2][nwarps];

    if (tid < 2) {
        counters[tid] = 0;
    }

    uint32_t prefix = 0;
    uint32_t pmask  = 0;
    int rank = k; // rank of the wanted key among the keys that match the prefix
#pragma unroll 1
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = tid; i < NCOPY*(256 + 1); i += BLOCK_SIZE) {
            (&hist[0][0])[i] = 0;
        }
        __syncthreads();

#pragma unroll
        for (int j = 0; j < MPT; ++j) {
            if (tid + j*BLOCK_SIZE < ncols && (key[j] & pmask) == prefix) {
                atomicAdd(&hist[lane % NCOPY][(key[j] >> shift) & 255], 1);
            }
        }
        __syncthreads();

        if (warp == 0) {
            // lane l owns the bins BPL*l .. BPL*l + BPL - 1
            int c[BPL];
            int lane_sum = 0;
#pragma unroll
            for (int b = 0; b < BPL; ++b) {
                int s = 0;
#pragma unroll
                for (int h = 0; h < NCOPY; ++h) {
                    s += hist[h][lane*BPL + b];
                }
                c[b] = s;
                lane_sum += s;
            }
            // keys in the bins of the lanes above
            int above = lane_sum;
#pragma unroll
            for (int offset = 1; offset < warp_size; offset *= 2) {
                const int v = __shfl_down_sync(ballot_mask, above, offset, warp_size);
                if (lane + offset < warp_size) {
                    above += v;
                }
            }
            above -= lane_sum;
            if (above < rank && above + lane_sum >= rank) {
                int cum = above;
#pragma unroll
                for (int b = BPL - 1; b >= 0; --b) {
                    if (cum + c[b] >= rank) {
                        sel[0] = lane*BPL + b;
                        sel[1] = rank - cum;
                        sel[2] = c[b];
                        break;
                    }
                    cum += c[b];
                }
            }
        }
        __syncthreads();
        prefix |= (uint32_t) sel[0] << shift;
        pmask  |= 255u << shift;
        rank    = sel[1];
    }

    // the keys above T all belong to the result, rank of the keys equal to T do
    const uint32_t T = prefix;
    const int n_gt   = k - rank;
    const int n_ties = sel[2];

    if (rank == n_ties) {
        // every key equal to T is taken, the order of the result is not specified
#pragma unroll
        for (int j = 0; j < MPT; ++j) {
            const int i = tid + j*BLOCK_SIZE;
            const bool gt  = i < ncols && key[j] >  T;
            const bool tie = i < ncols && key[j] == T;
            const lane_mask_t below = ((lane_mask_t) 1 << lane) - 1;
            const lane_mask_t m_gt  = (lane_mask_t) __ballot_sync(ballot_mask, gt);
            const lane_mask_t m_tie = (lane_mask_t) __ballot_sync(ballot_mask, tie);
            int base_gt  = 0;
            int base_tie = 0;
            if (lane == 0) {
                if (m_gt)  { base_gt  = atomicAdd(&counters[0], popc(m_gt)); }
                if (m_tie) { base_tie = atomicAdd(&counters[1], popc(m_tie)); }
            }
            base_gt  = __shfl_sync(ballot_mask, base_gt,  0, warp_size);
            base_tie = __shfl_sync(ballot_mask, base_tie, 0, warp_size);
            if (gt) {
                row_dst[base_gt + popc(m_gt & below)] = i;
            } else if (tie) {
                row_dst[n_gt + base_tie + popc(m_tie & below)] = i;
            }
        }
        return;
    }

    // more keys equal to T than slots left, the lowest columns win: count the ties in column order
    int base = 0;
#pragma unroll
    for (int j = 0; j < MPT; ++j) {
        const int i = tid + j*BLOCK_SIZE;
        const bool gt  = i < ncols && key[j] >  T;
        const bool tie = i < ncols && key[j] == T;
        const lane_mask_t below = ((lane_mask_t) 1 << lane) - 1;
        const lane_mask_t m_gt  = (lane_mask_t) __ballot_sync(ballot_mask, gt);
        const lane_mask_t m_tie = (lane_mask_t) __ballot_sync(ballot_mask, tie);

        if (lane == 0) {
            scan[j & 1][warp] = popc(m_tie);
        }
        int base_gt = 0;
        if (lane == 0 && m_gt) {
            base_gt = atomicAdd(&counters[0], popc(m_gt));
        }
        base_gt = __shfl_sync(ballot_mask, base_gt, 0, warp_size);
        __syncthreads();
        int before = 0;
        int chunk  = 0;
#pragma unroll
        for (int w = 0; w < nwarps; ++w) {
            const int v = scan[j & 1][w];
            before += w < warp ? v : 0;
            chunk  += v;
        }

        const int pos_tie = base + before + popc(m_tie & below);
        if (gt) {
            row_dst[base_gt + popc(m_gt & below)] = i;
        } else if (tie && pos_tie < rank) {
            row_dst[n_gt + pos_tie] = i;
        }
        base += chunk;
    }
}

template<int BLOCK_SIZE, int MPT>
static void top_k_block_launch(const float * src, int * dst, int ncols, int64_t nrows, int k, cudaStream_t stream) {
    top_k_block_kernel<BLOCK_SIZE, MPT><<<(unsigned) nrows, BLOCK_SIZE, 0, stream>>>(src, dst, ncols, k);
    CUDA_CHECK(cudaGetLastError());
}

// the widest row the block kernel holds in registers
#define GGML_CUDA_TOP_K_BLOCK_MAX_COLS 32768

template<int BLOCK_SIZE>
static void top_k_block_mpt(const float * src, int * dst, int ncols, int64_t nrows, int k, cudaStream_t stream) {
    const int mpt = (ncols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if      (mpt <=  4) { top_k_block_launch<BLOCK_SIZE,  4>(src, dst, ncols, nrows, k, stream); }
    else if (mpt <=  8) { top_k_block_launch<BLOCK_SIZE,  8>(src, dst, ncols, nrows, k, stream); }
    else if (mpt <= 12) { top_k_block_launch<BLOCK_SIZE, 12>(src, dst, ncols, nrows, k, stream); }
    else if (mpt <= 16) { top_k_block_launch<BLOCK_SIZE, 16>(src, dst, ncols, nrows, k, stream); }
    else if (mpt <= 24) { top_k_block_launch<BLOCK_SIZE, 24>(src, dst, ncols, nrows, k, stream); }
    else if (mpt <= 32) { top_k_block_launch<BLOCK_SIZE, 32>(src, dst, ncols, nrows, k, stream); }
    else if (mpt <= 40) { top_k_block_launch<BLOCK_SIZE, 40>(src, dst, ncols, nrows, k, stream); }
    else if (mpt <= 48) { top_k_block_launch<BLOCK_SIZE, 48>(src, dst, ncols, nrows, k, stream); }
    else                { top_k_block_launch<BLOCK_SIZE, 64>(src, dst, ncols, nrows, k, stream); }
}

// a few rows (decode, verify batches) are latency bound and take 1024 threads, so do rows of up to 12288 columns (the pools
// of a 48K context); many wide rows run faster with 512 threads and more keys per thread
static bool top_k_block_cuda(const float * src, int * dst, int ncols, int64_t nrows, int k, cudaStream_t stream) {
    if (k > ncols || ncols > GGML_CUDA_TOP_K_BLOCK_MAX_COLS) {
        return false;
    }
    if (ncols <= 12288 || nrows <= 16) {
        top_k_block_mpt<1024>(src, dst, ncols, nrows, k, stream);
    } else {
        top_k_block_mpt<512>(src, dst, ncols, nrows, k, stream);
    }
    return true;
}
#endif // defined(GGML_USE_HIP)

static void top_k_radix_cuda(
        ggml_cuda_pool & pool,
        const float * src, int * dst, int ncols, int64_t nrows, int k, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int RADIX_BITS = 8;
    constexpr int NBINS = 1 << RADIX_BITS;
    const int blocks_per_row = (int) std::min<int64_t>(((int64_t) ncols + 1023) / 1024, 64);
    // each block of a row reads a contiguous range of whole tiles
    const int block_cols = (int) (((int64_t) ncols + blocks_per_row * BLOCK_SIZE - 1) / (blocks_per_row * BLOCK_SIZE) * BLOCK_SIZE);

    // chunk the rows to bound the histogram memory to 64 MB
    const int64_t chunk_nrows = ggml_cuda_chunk_nrows((size_t) blocks_per_row * NBINS * sizeof(int), nrows);

    ggml_cuda_pool_alloc<top_k_radix_state> states_alloc(pool, chunk_nrows);
    ggml_cuda_pool_alloc<int> histograms_alloc(pool, (size_t) chunk_nrows * blocks_per_row * NBINS);
    top_k_radix_state * states = states_alloc.get();
    int * histograms = histograms_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        const int iter_nrows = std::min(chunk_nrows, nrows - i);

        top_k_radix_init<<<(iter_nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, iter_nrows, k);

        const dim3 row_grid(blocks_per_row * iter_nrows);
        for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
            top_k_radix_histogram<BLOCK_SIZE, RADIX_BITS>
                <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                    src, states, histograms, ncols, block_cols, blocks_per_row, shift);
            top_k_radix_select<BLOCK_SIZE, RADIX_BITS>
                <<<iter_nrows, BLOCK_SIZE, 0, stream>>>(histograms, states, blocks_per_row, shift);
        }

        top_k_radix_gather<BLOCK_SIZE, RADIX_BITS>
            <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                src, dst, histograms, states, ncols, block_cols, k, blocks_per_row);

        src += (size_t) ncols * iter_nrows;
        dst += (size_t) k     * iter_nrows;
    }
}

static void top_k_argsort_cuda(
        ggml_cuda_pool & pool,
        const float * src, int * dst, int ncols, int64_t nrows, int k, bool use_cub, cudaStream_t stream) {
    const int64_t chunk_nrows = ggml_cuda_chunk_nrows((size_t) ncols * sizeof(int), nrows);

    ggml_cuda_pool_alloc<int> tmp_alloc(pool, (size_t) ncols * chunk_nrows);
    int * tmp = tmp_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        const int iter_nrows = std::min(chunk_nrows, nrows - i);

        if (use_cub) {
#ifdef GGML_CUDA_USE_CUB
            argsort_f32_i32_cuda_cub(pool, src, tmp, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
#else
            GGML_ABORT("CUB is not available");
#endif // GGML_CUDA_USE_CUB
        } else {
            argsort_f32_i32_cuda_bitonic(src, tmp, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        }
        CUDA_CHECK(cudaMemcpy2DAsync(dst, k * sizeof(int), tmp, ncols * sizeof(int), k * sizeof(int), iter_nrows,
                                     cudaMemcpyDeviceToDevice, stream));

        src += (size_t) ncols * iter_nrows;
        dst += (size_t) k     * iter_nrows;
    }
}

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();

    const int device = ggml_cuda_get_device();

#ifdef CUB_TOP_K_AVAILABLE
    // a single row always uses DeviceTopK if available
    const bool bitonic_short    = nrows > 1 && ncols <= GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC;
#else
    const bool bitonic_short    = ncols <= GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC;
#endif // CUB_TOP_K_AVAILABLE
    const bool bitonic_few_rows = nrows > GGML_CUDA_TOP_K_NROWS_THRESHOLD &&
                                  ncols <= GGML_CUDA_TOP_K_NCOLS_THRESHOLD_BITONIC_FEW_ROWS &&
                                  nrows <= ggml_cuda_info().devices[device].nsm;

    if (bitonic_short || bitonic_few_rows) {
        // the padded row must fit in shared memory
        const int ncols_pad = next_power_of_2(ncols);
        if (ncols_pad * sizeof(int) <= ggml_cuda_info().devices[device].smpb) {
            top_k_argsort_cuda(pool, src0_d, dst_d, ncols, nrows, k, false, stream);
            return;
        }
    }

#if defined(GGML_USE_HIP)
    // rows of a few thousand columns (e.g. the Qwen QSA indexer scores): one block per row instead of 10 launches
    static const int env_block = ggml_cuda_env_switch("GGML_HIP_TOPK_BLOCK");
    if (ggml_cuda_switch_or_rdna3_5(env_block, ggml_cuda_info().devices[device].cc) &&
            top_k_block_cuda(src0_d, dst_d, ncols, nrows, k, stream)) {
        return;
    }
#endif // defined(GGML_USE_HIP)

    if (nrows > GGML_CUDA_TOP_K_NROWS_THRESHOLD) {
        top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
        return;
    }

#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Assess perf of `DeviceBatchedTopK` for multi-row TopK & CCCL >= 3.5.0, re-running perf sweep of https://github.com/ggml-org/llama.cpp/pull/28713
    for (int64_t i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    if (ncols <= GGML_CUDA_TOP_K_NCOLS_THRESHOLD_ARGSORT) {
        top_k_argsort_cuda(pool, src0_d, dst_d, ncols, nrows, k, true, stream);
    } else {
        top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
    }
#else                             // GGML_CUDA_USE_CUB
    top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
#endif                            // CUB_TOP_K_AVAILABLE
}
