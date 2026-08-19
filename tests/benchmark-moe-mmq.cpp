// Standalone benchmark for the MMQ mul_mat_id (MoE) kernel under different
// expert-routing distributions. The routing histogram (tokens per expert)
// drives the MMQ per-expert tiling, so this tool generates several
// distributions to measure how the kernel behaves as that histogram changes.
//
// Distributions:
//   uniform       - all tokens spread evenly across every expert (balanced router)
//   single        - one expert receives all tokens (worst-case concentration)
//   zipf          - power-law imbalance, expert i gets a share ~ 1/i^alpha
//   concentration - sweep of tokens-per-expert (aligned to 16); all tokens are
//                   shared by a shrinking active set, so different experts
//                   receive all the tokens
//
// Defaults match Qwen3.6-35B-A3B (256 experts, top_k 8, n_embd 2048,
// moe_intermediate 512, Q4_K).

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

struct bench_config {
    int         n_experts = 256;
    int         top_k     = 8;
    int64_t     n_embd    = 2048;
    int64_t     n_ff      = 512;
    ggml_type   type_a    = GGML_TYPE_Q4_K;
    ggml_type   type_b    = GGML_TYPE_F32;
    double      alpha     = 1.0;
    int         rotate    = 0;
    unsigned    seed      = 1234;
    std::string dist      = "all";  // all|uniform|single|zipf|concentration
    std::string proj      = "both"; // both|gateup|down
    std::string csv_path;
    std::vector<int64_t> batches = {1, 4, 8, 32, 64, 128, 256, 512, 2048, 4096};
};

struct run_result {
    std::string proj;
    std::string dist;
    int64_t     n_tokens   = 0;
    int         n_active    = 0;
    int64_t     tpe_target = 0;
    int64_t     tpe_min    = 0;
    int64_t     tpe_max    = 0;
    double      tpe_mean   = 0.0;
    int         n_copies   = 1;
    double      avg_us     = 0.0;
    double      gflops     = 0.0;
    double      ai         = 0.0;   // arithmetic intensity, FLOP/byte
    double      roof_pct   = 0.0;   // measured / roofline ceiling, %
    bool        compute_bound = false;
};

static ggml_type parse_type(const std::string & name) {
    if (name == "q4_K" || name == "q4_k") { return GGML_TYPE_Q4_K; }
    if (name == "q5_K" || name == "q5_k") { return GGML_TYPE_Q5_K; }
    if (name == "q6_K" || name == "q6_k") { return GGML_TYPE_Q6_K; }
    if (name == "q4_0")                   { return GGML_TYPE_Q4_0; }
    if (name == "q8_0")                   { return GGML_TYPE_Q8_0; }
    if (name == "f16")                    { return GGML_TYPE_F16;  }
    if (name == "f32")                    { return GGML_TYPE_F32;  }
    fprintf(stderr, "unknown type '%s', defaulting to q4_K\n", name.c_str());
    return GGML_TYPE_Q4_K;
}

static std::vector<int64_t> parse_int_list(const std::string & s) {
    std::vector<int64_t> out;
    size_t pos = 0;
    while (pos < s.size()) {
        size_t comma = s.find(',', pos);
        std::string tok = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        if (!tok.empty()) {
            out.push_back(std::stoll(tok));
        }
        if (comma == std::string::npos) { break; }
        pos = comma + 1;
    }
    return out;
}

// Target token count per expert (sum == total). Empty entries are experts that
// receive no tokens.
static std::vector<int64_t> counts_uniform(int n_experts, int n_active, int64_t total) {
    n_active = std::max(1, std::min(n_active, n_experts));
    std::vector<int64_t> counts(n_experts, 0);
    const int64_t base = total / n_active;
    const int64_t rem  = total % n_active;
    for (int i = 0; i < n_active; i++) {
        counts[i] = base + (i < rem ? 1 : 0);
    }
    return counts;
}

static std::vector<int64_t> counts_zipf(int n_experts, int n_active, double alpha, int64_t total) {
    n_active = std::max(1, std::min(n_active, n_experts));
    std::vector<double> w(n_active);
    double sum_w = 0.0;
    for (int i = 0; i < n_active; i++) {
        w[i] = 1.0 / std::pow((double) (i + 1), alpha);
        sum_w += w[i];
    }
    std::vector<int64_t> counts(n_experts, 0);
    int64_t assigned = 0;
    for (int i = 0; i < n_active; i++) {
        counts[i] = (int64_t) std::floor(w[i] / sum_w * (double) total);
        assigned += counts[i];
    }
    // hand out the rounding remainder to the heaviest experts
    for (int i = 0; assigned < total; i = (i + 1) % n_active) {
        counts[i]++;
        assigned++;
    }
    return counts;
}

// Realize a target histogram as a per-token expert list, keeping the top_k
// experts within a token distinct. Each token greedily takes the experts with
// the most remaining budget, which spreads heavy experts across many tokens.
static std::vector<int32_t> assign_ids(const std::vector<int64_t> & counts,
                                       int n_experts, int top_k, int64_t n_tokens,
                                       std::mt19937 & rng) {
    std::vector<int64_t> rem = counts;
    std::vector<int32_t> ids((size_t) top_k * n_tokens);
    std::vector<int>     idx(n_experts);
    for (int64_t t = 0; t < n_tokens; t++) {
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), rng);
        std::stable_sort(idx.begin(), idx.end(),
                         [&](int a, int b) { return rem[a] > rem[b]; });
        for (int u = 0; u < top_k; u++) {
            const int e = idx[u];
            ids[(size_t) t * top_k + u] = e;
            if (rem[e] > 0) { rem[e]--; }
        }
    }
    return ids;
}

static void hist_stats(const std::vector<int32_t> & ids, int n_experts, int & n_active,
                       int64_t & tpe_min, int64_t & tpe_max, double & tpe_mean) {
    std::vector<int64_t> hist(n_experts, 0);
    for (int32_t e : ids) { hist[e]++; }
    int active = 0;
    int64_t hi = 0, lo = INT64_MAX, sum = 0;
    for (int64_t c : hist) {
        if (c == 0) { continue; }
        active++;
        sum += c;
        hi = std::max(hi, c);
        lo = std::min(lo, c);
    }
    n_active = active;
    tpe_max  = hi;
    tpe_min  = active ? lo : 0;
    tpe_mean = active ? (double) sum / active : 0.0;
}

static void fill_uniform_data(ggml_tensor * t, std::mt19937 & rng) {
    const size_t nels = ggml_nelements(t);
    std::vector<float> data(nels);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < nels; i++) { data[i] = dist(rng); }

    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(t, data.data(), 0, nels * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> h(nels);
        ggml_fp32_to_fp16_row(data.data(), h.data(), nels);
        ggml_backend_tensor_set(t, h.data(), 0, nels * sizeof(ggml_fp16_t));
    } else if (ggml_is_quantized(t->type)) {
        const int64_t n_per_row = t->ne[0];
        const int64_t nrows     = (int64_t) nels / n_per_row;
        std::vector<uint8_t> q(ggml_row_size(t->type, nels));
        ggml_quantize_chunk(t->type, data.data(), q.data(), 0, nrows, n_per_row, nullptr);
        ggml_backend_tensor_set(t, q.data(), 0, q.size());
    } else {
        GGML_ABORT("unsupported tensor type for benchmark init");
    }
}

static ggml_backend_t create_backend() {
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const int type = (int) ggml_backend_dev_type(dev);
        if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
            if (backend) {
                printf("backend: %s (%s)\n", ggml_backend_name(backend), ggml_backend_dev_description(dev));
                return backend;
            }
        }
    }
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    printf("backend: %s (no GPU found)\n", ggml_backend_name(backend));
    return backend;
}

// gfx1151 (Strix Halo) cache hierarchy: 2 MiB L2 + 32 MiB MALL (L3).
static constexpr size_t L2_BYTES = 2ull  << 20;
static constexpr size_t L3_BYTES = 32ull << 20;

// gfx1151 roofline peaks.
static constexpr double PEAK_GFLOPS = 43.0 * 1000.0; // 43 TFLOP/s
static constexpr double PEAK_BW_BS  = 230.0 * (1024.0 * 1024.0 * 1024.0); // 230 GiB/s in bytes/s

// Pick enough distinct weight copies that the rotated working set is at least
// 2x the last-level cache, so the reused weights get evicted between cycles and
// reads land in DRAM instead of L2/L3.
static int auto_rotate(size_t active_footprint, int n_runs) {
    if (active_footprint == 0) {
        return 1;
    }
    const size_t target = 2 * std::max(L2_BYTES, L3_BYTES);
    const int    k       = (int) ((target + active_footprint - 1) / active_footprint);
    return std::max(1, std::min({ k, n_runs, 64 }));
}

// Build a single mul_mat_id op, feed it the routing described by `counts`, run
// the timing loop and return the measured result. `broadcast` selects the
// gate/up projection (token shared across experts) vs. the down projection.
static run_result run_one(ggml_backend_t backend, const bench_config & cfg,
                          const std::string & proj, bool broadcast,
                          int64_t m, int64_t k, int64_t n_tokens,
                          const std::string & dist_name, int64_t tpe_target,
                          const std::vector<int64_t> & counts, std::mt19937 & rng) {
    const int n_experts = cfg.n_experts;
    const int top_k     = cfg.top_k;

    const uint64_t flops        = 2ull * m * k * n_tokens * top_k;
    const uint64_t target_flops = 100ull * 1000 * 1000 * 1000;
    const int      max_runs     = 4096;
    const int      n_runs       = (int) std::min<int64_t>(max_runs, target_flops / std::max<uint64_t>(flops, 1)) + 1;

    // Resolve the routing first (CPU only) so the number of experts actually
    // touched is known before sizing the weight rotation.
    const std::vector<int32_t> used_ids = assign_ids(counts, n_experts, top_k, n_tokens, rng);

    run_result res;
    res.proj       = proj;
    res.dist       = dist_name;
    res.n_tokens   = n_tokens;
    res.tpe_target = tpe_target;
    hist_stats(used_ids, n_experts, res.n_active, res.tpe_min, res.tpe_max, res.tpe_mean);

    // Rotate through `n_copies` distinct weight buffers so consecutive op
    // instances read different addresses, keeping the expert weights from
    // sitting hot in L2/MALL across iterations. Only the active experts of each
    // copy are read, so size the rotation off that footprint. rotate=0 -> auto.
    const size_t per_expert_bytes = (size_t) m * ggml_row_size(cfg.type_a, k);
    const size_t active_footprint = (size_t) res.n_active * per_expert_bytes;
    const int n_copies = cfg.rotate > 0
        ? std::max(1, std::min(cfg.rotate, n_runs))
        : auto_rotate(active_footprint, n_runs);
    res.n_copies = n_copies;

    const size_t n_tensors = (size_t) n_copies + n_runs + 8;
    ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead() * n_tensors + ggml_graph_overhead_custom(n_runs + 16, false),
        /* .mem_base   = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);

    std::vector<ggml_tensor *> as(n_copies);
    for (int c = 0; c < n_copies; c++) {
        as[c] = ggml_new_tensor_3d(ctx, cfg.type_a, k, m, n_experts);
        ggml_set_name(as[c], "as");
    }

    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_experts, n_tokens);
    ggml_set_name(ids, "ids");
    ggml_tensor * ids_used = ids;
    if (top_k != n_experts) {
        ids_used = ggml_view_2d(ctx, ids, top_k, n_tokens, ids->nb[1], 0);
    }

    ggml_tensor * b = ggml_new_tensor_3d(ctx, cfg.type_b, k, broadcast ? 1 : top_k, n_tokens);
    ggml_set_name(b, "b");

    std::vector<ggml_tensor *> outs(n_runs);
    for (int i = 0; i < n_runs; i++) {
        outs[i] = ggml_mul_mat_id(ctx, as[i % n_copies], b, ids_used);
        ggml_set_name(outs[i], "out");
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buf != nullptr);

    // Quantize the weights once, then mirror them into the other copies: the
    // data is identical but the buffers are distinct, which is all the cache
    // rotation needs (cache is addressed physically, not by content).
    fill_uniform_data(as[0], rng);
    for (int c = 1; c < n_copies; c++) {
        ggml_backend_tensor_copy(as[0], as[c]);
    }
    fill_uniform_data(b, rng);

    // ids storage is [n_experts, n_tokens]; only the first top_k rows of each
    // column are read through the view, the rest stay zeroed.
    std::vector<int32_t> ids_full((size_t) n_experts * n_tokens, 0);
    for (int64_t t = 0; t < n_tokens; t++) {
        for (int u = 0; u < top_k; u++) {
            ids_full[(size_t) t * n_experts + u] = used_ids[(size_t) t * top_k + u];
        }
    }
    ggml_backend_tensor_set(ids, ids_full.data(), 0, ids_full.size() * sizeof(int32_t));

    ggml_cgraph * gf = ggml_new_graph_custom(ctx, n_runs + 16, false);
    for (int i = 0; i < n_runs; i++) {
        ggml_build_forward_expand(gf, outs[i]);
    }

    if (!ggml_backend_supports_op(backend, outs[0])) {
        printf("  op not supported by backend, skipping\n");
        ggml_free(ctx);
        ggml_backend_buffer_free(buf);
        return res;
    }

    ggml_backend_graph_compute(backend, gf); // warmup

    int64_t total_us = 0;
    int64_t total_runs = 0;
    do {
        const int64_t t0 = ggml_time_us();
        ggml_backend_graph_compute(backend, gf);
        total_us += ggml_time_us() - t0;
        total_runs += n_runs;
    } while (total_us < 1000 * 1000);

    res.avg_us = (double) total_us / total_runs;
    res.gflops = (double) flops * total_runs / ((double) total_us / 1e6) / 1e9;

    // Roofline: bytes moved per op = f32 output + active q4_K experts (each read
    // once) + q8_1-quantized input. The gather replicates the input across the
    // top_k experts, so it is read k*top_k*n_tokens times regardless of
    // broadcast. AI drives the min(compute, bw*AI) ceiling.
    const int64_t b_read_elems = k * top_k * n_tokens;
    const double  out_bytes    = (double) m * top_k * n_tokens * sizeof(float);
    const double  input_bytes  = (double) ((b_read_elems + 31) / 32) * 36;
    const double  bytes        = out_bytes + (double) active_footprint + input_bytes;
    res.ai       = (double) flops / bytes;
    const double roof_mem = res.ai * PEAK_BW_BS / 1e9;
    const double roof     = std::min(PEAK_GFLOPS, roof_mem);
    res.compute_bound = roof_mem >= PEAK_GFLOPS;
    res.roof_pct = roof > 0.0 ? res.gflops / roof * 100.0 : 0.0;

    ggml_free(ctx);
    ggml_backend_buffer_free(buf);
    return res;
}

static void print_legend() {
    printf("columns:\n");
    printf("  proj     projection: gateup (k=n_embd, m=n_ff, token broadcast) or down (k=n_ff, m=n_embd)\n");
    printf("  dist     routing distribution: uniform | single | zipf | concentration\n");
    printf("  n_tok    number of tokens in the batch\n");
    printf("  n_act    experts that actually receive tokens (from the generated routing)\n");
    printf("  tpe_tgt  target tokens-per-expert for the distribution (0 = not applicable)\n");
    printf("  tpe_min  min tokens received by an active expert\n");
    printf("  tpe_max  max tokens received by an active expert (worst-case per-expert load)\n");
    printf("  tpe_mean mean tokens-per-expert over active experts\n");
    printf("  rot      distinct weight copies cycled to defeat L2/MALL caching (auto unless --rotate)\n");
    printf("  avg_us   average time per mul_mat_id, microseconds\n");
    printf("  GFLOP/s  measured throughput (2*m*k*n_tok*top_k FLOPs)\n");
    printf("  AI       arithmetic intensity, FLOP/byte (bytes = out f32 + active q4_K experts + q8_1 input)\n");
    printf("  %%roof    measured / roofline ceiling min(peak_compute, AI*peak_bw), %%\n");
    printf("  bound    mem (bandwidth-bound, AI < ridge) or comp (compute-bound, AI > ridge)\n");
    printf("distributions (how tokens are routed to experts):\n");
    printf("  uniform        tokens spread evenly across all experts (balanced router)\n");
    printf("  single         all tokens routed to one expert (worst-case concentration)\n");
    printf("  zipf           power-law imbalance: ranked expert i gets a share ~ 1/i^alpha (--alpha)\n");
    printf("  concentration  sweep of tokens-per-expert aligned to 16; all tokens shared by a\n");
    printf("                 shrinking active set (n_active from n_experts down to top_k)\n\n");
}

static void print_row(const run_result & r) {
    printf("%-7s %-13s %6lld %6d %8lld %8lld %8lld %9.1f %5d %10.2f %10.1f %7.1f %7.1f %5s\n",
           r.proj.c_str(), r.dist.c_str(), (long long) r.n_tokens, r.n_active,
           (long long) r.tpe_target, (long long) r.tpe_min, (long long) r.tpe_max,
           r.tpe_mean, r.n_copies, r.avg_us, r.gflops, r.ai, r.roof_pct,
           r.compute_bound ? "comp" : "mem");
}

static void print_header() {
    printf("%-7s %-13s %6s %6s %8s %8s %8s %9s %5s %10s %10s %7s %7s %5s\n",
           "proj", "dist", "n_tok", "n_act", "tpe_tgt", "tpe_min", "tpe_max",
           "tpe_mean", "rot", "avg_us", "GFLOP/s", "AI", "%roof", "bound");
}

static void collect_runs(ggml_backend_t backend, const bench_config & cfg,
                         const std::string & proj, bool broadcast, int64_t m, int64_t k,
                         std::mt19937 & rng, std::vector<run_result> & results) {
    const bool all = cfg.dist == "all";

    if (all || cfg.dist == "uniform") {
        for (int64_t bs : cfg.batches) {
            const int64_t total = bs * cfg.top_k;
            auto counts = counts_uniform(cfg.n_experts, cfg.n_experts, total);
            auto r = run_one(backend, cfg, proj, broadcast, m, k, bs, "uniform",
                             total / cfg.n_experts, counts, rng);
            print_row(r);
            results.push_back(r);
        }
    }
    if (all || cfg.dist == "single") {
        for (int64_t bs : cfg.batches) {
            const int64_t total = bs * cfg.top_k;
            auto counts = counts_uniform(cfg.n_experts, 1, total);
            auto r = run_one(backend, cfg, proj, broadcast, m, k, bs, "single", total, counts, rng);
            print_row(r);
            results.push_back(r);
        }
    }
    if (all || cfg.dist == "zipf") {
        for (int64_t bs : cfg.batches) {
            const int64_t total = bs * cfg.top_k;
            auto counts = counts_zipf(cfg.n_experts, cfg.n_experts, cfg.alpha, total);
            auto r = run_one(backend, cfg, proj, broadcast, m, k, bs, "zipf", 0, counts, rng);
            print_row(r);
            results.push_back(r);
        }
    }
    if (all || cfg.dist == "concentration") {
        // Sweep tokens-per-expert (aligned to 16) at each batch size. Skip
        // targets that cannot be realized: n_active must divide the routing
        // evenly and stay within [top_k, n_experts] - fewer than top_k experts
        // is impossible with distinct top-k routing, more than n_experts does
        // not exist.
        bool any = false;
        for (int64_t bs : cfg.batches) {
            const int64_t total = bs * cfg.top_k;
            for (int64_t tpe = 16; tpe <= total; tpe *= 2) {
                if (total % tpe != 0) { continue; }
                const int64_t n_active = total / tpe;
                if (n_active < cfg.top_k || n_active > cfg.n_experts) { continue; }
                auto counts = counts_uniform(cfg.n_experts, (int) n_active, total);
                auto r = run_one(backend, cfg, proj, broadcast, m, k, bs, "concentration",
                                 tpe, counts, rng);
                print_row(r);
                results.push_back(r);
                any = true;
            }
        }
        if (!any) {
            printf("# %-7s concentration: no feasible config for the given --m "
                   "(need n_tok*top_k divisible by 16 with n_active in [%d, %d]); "
                   "smallest usable n is 16\n",
                   proj.c_str(), cfg.top_k, cfg.n_experts);
        }
    }
}

static void write_csv(const bench_config & cfg, const std::vector<run_result> & results) {
    FILE * f = fopen(cfg.csv_path.c_str(), "w");
    if (!f) {
        fprintf(stderr, "failed to open %s for writing\n", cfg.csv_path.c_str());
        return;
    }
    fprintf(f, "proj,dtype,n_experts,top_k,dist,n_tokens,n_active,tpe_target,tpe_min,tpe_max,tpe_mean,rotate,avg_us,gflops,ai,roof_pct,bound\n");
    for (const run_result & r : results) {
        fprintf(f, "%s,%s,%d,%d,%s,%lld,%d,%lld,%lld,%lld,%.3f,%d,%.3f,%.3f,%.3f,%.3f,%s\n",
                r.proj.c_str(), ggml_type_name(cfg.type_a), cfg.n_experts, cfg.top_k,
                r.dist.c_str(), (long long) r.n_tokens, r.n_active, (long long) r.tpe_target,
                (long long) r.tpe_min, (long long) r.tpe_max, r.tpe_mean, r.n_copies, r.avg_us, r.gflops,
                r.ai, r.roof_pct, r.compute_bound ? "comp" : "mem");
    }
    fclose(f);
    printf("wrote %zu rows to %s\n", results.size(), cfg.csv_path.c_str());
}

static void usage(const char * prog) {
    printf("usage: %s [options]\n", prog);
    printf("  --experts N        number of experts (default 256)\n");
    printf("  --top-k N          experts used per token (default 8)\n");
    printf("  --n-embd N         model hidden size (default 2048)\n");
    printf("  --n-ff N           moe intermediate size (default 512)\n");
    printf("  --type NAME        expert weight type: q4_K,q5_K,q6_K,q4_0,q8_0,f16,f32 (default q4_K)\n");
    printf("  --dist NAME        all|uniform|single|zipf|concentration (default all)\n");
    printf("  --proj NAME        both|gateup|down (default both)\n");
    printf("  --alpha F          zipf exponent (default 1.0)\n");
    printf("  --rotate K         distinct weight copies cycled per op to defeat L2/MALL caching\n");
    printf("                     (default 0 = auto-size so the working set exceeds L3; K>=1 forces a fixed count)\n");
    printf("  --m a,b,c          token counts to sweep (default 1,4,8,...,4096)\n");
    printf("  --legend           print the legend (columns and distributions) and exit\n");
    printf("  --seed N           RNG seed (default 1234)\n");
    printf("  -o, --output FILE  write results as CSV\n");
}

int main(int argc, char ** argv) {
    bench_config cfg;

    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", arg.c_str()); exit(1); }
            return argv[++i];
        };
        if (arg == "--experts")           { cfg.n_experts  = std::stoi(next()); }
        else if (arg == "--top-k")        { cfg.top_k      = std::stoi(next()); }
        else if (arg == "--n-embd")       { cfg.n_embd     = std::stoll(next()); }
        else if (arg == "--n-ff")         { cfg.n_ff       = std::stoll(next()); }
        else if (arg == "--type")         { cfg.type_a     = parse_type(next()); }
        else if (arg == "--dist")         { cfg.dist       = next(); }
        else if (arg == "--proj")         { cfg.proj       = next(); }
        else if (arg == "--alpha")        { cfg.alpha      = std::stod(next()); }
        else if (arg == "--rotate")       { cfg.rotate     = std::stoi(next()); }
        else if (arg == "--m")            { cfg.batches    = parse_int_list(next()); }
        else if (arg == "--legend")       { print_legend(); return 0; }
        else if (arg == "--seed")         { cfg.seed       = (unsigned) std::stoul(next()); }
        else if (arg == "-o" || arg == "--output") { cfg.csv_path = next(); }
        else if (arg == "-h" || arg == "--help")   { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown argument: %s\n", arg.c_str()); usage(argv[0]); return 1; }
    }
    GGML_ASSERT(cfg.top_k <= cfg.n_experts);

    ggml_time_init();
    ggml_backend_t backend = create_backend();
    printf("config: experts=%d top_k=%d n_embd=%lld n_ff=%lld dtype=%s alpha=%.2f rotate=%d\n",
           cfg.n_experts, cfg.top_k, (long long) cfg.n_embd, (long long) cfg.n_ff,
           ggml_type_name(cfg.type_a), cfg.alpha, cfg.rotate);
    printf("roofline peaks: %.0f TFLOP/s, %.0f GiB/s (ridge AI %.1f FLOP/B)\n\n",
           PEAK_GFLOPS / 1000.0, PEAK_BW_BS / (1024.0 * 1024.0 * 1024.0), PEAK_GFLOPS * 1e9 / PEAK_BW_BS);

    std::mt19937 rng(cfg.seed);
    std::vector<run_result> results;
    print_header();

    // gate/up: k = n_embd, m = n_ff, token broadcast across experts
    // down:    k = n_ff,   m = n_embd
    if (cfg.proj == "both" || cfg.proj == "gateup") {
        collect_runs(backend, cfg, "gateup", true, cfg.n_ff, cfg.n_embd, rng, results);
    }
    if (cfg.proj == "both" || cfg.proj == "down") {
        collect_runs(backend, cfg, "down", false, cfg.n_embd, cfg.n_ff, rng, results);
    }

    if (!cfg.csv_path.empty()) {
        write_csv(cfg, results);
    }

    ggml_backend_free(backend);
    return 0;
}
