// Autotuner for the MMQ (mul_mat_q) tile width. The catalog holds every distinct shape that a
// set of 20 models dispatches to mul_mat_q, as captured from GGML_ROOFLINE_OUT traces.
//
// `weight` is the invocation count per forward pass, so aggregates reflect real GPU time rather
// than treating a shape run 10 times as equal to one run 900 times. Counts come from the
// -ub 2048 capture and are reused as a proxy at the other token counts the sweep covers.
//
// The token count is a sweep parameter, not part of the catalog, and it matters: the dense
// q8_0 mmq_x=64 cliff only shows up around 64 tokens, while the MoE tile-width gap needs 2048.
//
// mmq_x is swept at runtime. mmq_y and the warp count are compile-time constants, so sweeping
// those means rebuilding - see sweep.py.

#include "ggml.h"
#include "ggml-mmq-tune.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct shape {
    const char * model;
    ggml_type    type;
    int64_t      n;
    int64_t      k;
    int          n_experts; // 0 = dense
    int          top_k;
    int          weight;    // invocations per pp128 run
};

static const shape g_shapes[] = {
    { "Qwen3.6-27B_Q4_K_M", GGML_TYPE_Q4_K, 17408, 5120, 0, 0, 128 },  // 1853.7 ms
    { "gemma-3-12b-it_Q4_", GGML_TYPE_Q4_K, 15360, 3840, 0, 0, 188 },  // 1736.0 ms
    { "Qwen2.5-7B-Instruc", GGML_TYPE_Q4_K, 18944, 3584, 0, 0, 54 },  // 603.9 ms
    { "Llama-3.1-8B-Instr", GGML_TYPE_Q4_K, 14336, 4096, 0, 0, 62 },  // 555.2 ms
    { "Qwen3-8B_Q4_K_M", GGML_TYPE_Q4_K, 12288, 4096, 0, 0, 70 },  // 542.7 ms
    { "Qwen2.5-3B-Instruc", GGML_TYPE_Q4_K, 11008, 2048, 0, 0, 140 },  // 479.7 ms
    { "Qwen3.6-27B_Q4_K_M", GGML_TYPE_Q4_K, 5120, 17408, 0, 0, 32 },  // 459.2 ms
    { "Qwen3.5-35B-A3B_Q4", GGML_TYPE_Q4_K, 512, 2048, 256, 8, 160 },  // 452.0 ms
    { "gemma-3-12b-it_Q4_", GGML_TYPE_Q4_K, 3840, 15360, 0, 0, 48 },  // 424.5 ms
    { "Llama-3.1-8B-Instr", GGML_TYPE_Q4_K, 4096, 4096, 0, 0, 136 },  // 337.8 ms
    { "GLM-4.7-Flash_Q4_K", GGML_TYPE_Q4_K, 1536, 2048, 64, 4, 90 },  // 277.2 ms
    { "gemma-3-4b-it_Q4_K", GGML_TYPE_Q4_K, 10240, 2560, 0, 0, 66 },  // 262.8 ms
    { "Qwen3-4B_Q4_K_M", GGML_TYPE_Q4_K, 9728, 2560, 0, 0, 70 },  // 258.6 ms
    { "Qwen3.6-27B_Q4_K_M", GGML_TYPE_Q5_K, 5120, 6144, 0, 0, 48 },  // 254.3 ms
    { "Qwen3.5-35B-A3B_Q4", GGML_TYPE_Q5_K, 2048, 512, 256, 8, 77 },  // 250.0 ms
    { "gemma-3-12b-it_Q4_", GGML_TYPE_Q4_K, 4096, 3840, 0, 0, 96 },  // 242.1 ms
    { "Qwen3.6-27B_Q4_K_M", GGML_TYPE_Q4_K, 6144, 5120, 0, 0, 48 },  // 241.7 ms
    { "gemma-4-26B-A4B-it", GGML_TYPE_Q4_K, 1408, 2816, 128, 8, 30 },  // 232.6 ms
    { "gemma-3-12b-it_Q4_", GGML_TYPE_Q4_K, 3840, 4096, 0, 0, 96 },  // 221.1 ms
    { "Qwen2.5-3B-Instruc", GGML_TYPE_Q4_K, 2048, 2048, 0, 0, 320 },  // 207.2 ms
    { "Qwen3.6-27B_Q4_K_M", GGML_TYPE_Q4_K, 10240, 5120, 0, 0, 24 },  // 206.1 ms
    { "gemma-4-26B-A4B-it", GGML_TYPE_Q5_1, 2816, 704, 128, 8, 29 },  // 195.4 ms
    { "Qwen3.5-35B-A3B_Q4", GGML_TYPE_Q8_0, 8192, 2048, 0, 0, 80 },  // 186.7 ms
    { "gemma-3-12b-it_Q4_", GGML_TYPE_Q4_K, 2048, 3840, 0, 0, 144 },  // 176.3 ms
    { "gemma-2b-it_Q4_K_M", GGML_TYPE_Q4_K, 16384, 2048, 0, 0, 34 },  // 172.6 ms
    { "Qwen3.6-27B_Q4_K_M", GGML_TYPE_Q4_K, 12288, 5120, 0, 0, 16 },  // 165.1 ms
    { "Qwen2.5-7B-Instruc", GGML_TYPE_Q4_K, 3584, 18944, 0, 0, 14 },  // 155.8 ms
    { "Llama-3.1-8B-Instr", GGML_TYPE_Q4_K, 4096, 14336, 0, 0, 16 },  // 142.6 ms
    { "Qwen3-8B_Q4_K_M", GGML_TYPE_Q4_K, 4096, 12288, 0, 0, 18 },  // 140.2 ms
    { "Qwen2.5-7B-Instruc", GGML_TYPE_Q4_K, 3584, 3584, 0, 0, 56 },  // 129.8 ms
    { "Qwen2.5-3B-Instruc", GGML_TYPE_Q4_K, 2048, 11008, 0, 0, 36 },  // 125.2 ms
    { "gemma-4-E2B-it_Q4_", GGML_TYPE_Q4_K, 12288, 1536, 0, 0, 40 },  // 115.6 ms
    { "SmolLM2-1.7B-Instr", GGML_TYPE_Q4_K, 8192, 2048, 0, 0, 46 },  // 111.2 ms
    { "GLM-4.7-Flash_Q4_K", GGML_TYPE_Q6_K, 2048, 1536, 64, 4, 21 },  // 109.6 ms
    { "Qwen3-1.7B_Q4_K_M", GGML_TYPE_Q4_K, 6144, 2048, 0, 0, 54 },  // 97.8 ms
    { "Qwen3.5-35B-A3B_Q4", GGML_TYPE_Q8_0, 2048, 4096, 0, 0, 80 },  // 94.0 ms
    { "Qwen3.6-27B_Q4_K_M", GGML_TYPE_Q4_K, 5120, 6144, 0, 0, 16 },  // 77.3 ms
    { "GLM-4.7-Flash_Q4_K", GGML_TYPE_Q4_K, 2048, 1536, 64, 4, 24 },  // 75.5 ms
    { "Qwen3.5-35B-A3B_Q4", GGML_TYPE_Q8_0, 4096, 2048, 0, 0, 60 },  // 75.0 ms
    { "GLM-4.7-Flash_Q4_K", GGML_TYPE_Q4_K, 2048, 5120, 0, 0, 47 },  // 71.3 ms
    { "Qwen3-4B_Q4_K_M", GGML_TYPE_Q4_K, 2560, 9728, 0, 0, 18 },  // 69.4 ms
    { "gemma-3-4b-it_Q4_K", GGML_TYPE_Q4_K, 2560, 10240, 0, 0, 17 },  // 68.3 ms
    { "Llama-3.1-8B-Instr", GGML_TYPE_Q4_K, 1024, 4096, 0, 0, 102 },  // 64.9 ms
    { "Qwen3-4B_Q4_K_M", GGML_TYPE_Q4_K, 4096, 2560, 0, 0, 36 },  // 60.6 ms
    { "Qwen3-4B_Q4_K_M", GGML_TYPE_Q4_K, 2560, 4096, 0, 0, 36 },  // 55.0 ms
    { "gemma-4-26B-A4B-it", GGML_TYPE_Q8_0, 2112, 2816, 0, 0, 60 },  // 48.3 ms
    { "GLM-4.7-Flash_Q4_K", GGML_TYPE_Q5_K, 1536, 2048, 0, 0, 90 },  // 48.3 ms
    { "Qwen2.5-0.5B-Instr", GGML_TYPE_Q5_0, 4864, 896, 0, 0, 46 },  // 45.2 ms
    { "gemma-2b-it_Q4_K_M", GGML_TYPE_Q4_K, 2048, 16384, 0, 0, 9 },  // 44.4 ms
    { "Qwen3-4B_Q4_K_M", GGML_TYPE_Q4_K, 1024, 2560, 0, 0, 105 },  // 43.6 ms
    { "gemma-4-E2B-it_Q4_", GGML_TYPE_Q4_K, 6144, 1536, 0, 0, 30 },  // 43.2 ms
    { "gemma-4-26B-A4B-it", GGML_TYPE_Q8_0, 2048, 2816, 0, 0, 50 },  // 41.9 ms
    { "gemma-4-26B-A4B-it", GGML_TYPE_Q8_0, 4096, 2816, 0, 0, 25 },  // 41.2 ms
    { "gemma-4-26B-A4B-it", GGML_TYPE_Q8_0, 2816, 4096, 0, 0, 25 },  // 39.7 ms
    { "Qwen3.5-35B-A3B_Q4", GGML_TYPE_Q8_0, 512, 2048, 0, 0, 200 },  // 32.7 ms
    { "GLM-4.7-Flash_Q4_K", GGML_TYPE_Q4_K, 5120, 768, 0, 0, 47 },  // 30.6 ms
    { "SmolLM2-1.7B-Instr", GGML_TYPE_Q4_K, 2048, 8192, 0, 0, 12 },  // 30.0 ms
    { "Qwen3.5-0.8B_Q4_K_", GGML_TYPE_Q4_K, 3584, 1024, 0, 0, 48 },  // 28.8 ms
    { "gemma-4-E2B-it_Q4_", GGML_TYPE_Q4_K, 1536, 12288, 0, 0, 10 },  // 28.5 ms
    { "gemma-3-4b-it_Q4_K", GGML_TYPE_Q4_K, 2048, 2560, 0, 0, 34 },  // 28.3 ms
    { "gemma-3-4b-it_Q4_K", GGML_TYPE_Q4_K, 2560, 2048, 0, 0, 34 },  // 27.8 ms
    { "gemma-4-26B-A4B-it", GGML_TYPE_Q8_0, 2816, 2112, 0, 0, 30 },  // 26.9 ms
    { "Qwen3-1.7B_Q4_K_M", GGML_TYPE_Q4_K, 2048, 6144, 0, 0, 14 },  // 26.3 ms
    { "GLM-4.7-Flash_Q4_K", GGML_TYPE_Q8_0, 512, 192, 0, 0, 47 },  // 23.9 ms
    { "Qwen3.6-27B_Q4_K_M", GGML_TYPE_Q4_K, 1024, 5120, 0, 0, 24 },  // 20.6 ms
    { "Qwen3.5-0.8B_Q4_K_", GGML_TYPE_Q5_K, 6144, 1024, 0, 0, 18 },  // 18.7 ms
    { "GLM-4.7-Flash_Q4_K", GGML_TYPE_Q8_0, 256, 512, 0, 0, 47 },  // 18.5 ms
    { "gemma-4-26B-A4B-it", GGML_TYPE_Q8_0, 8192, 2816, 0, 0, 5 },  // 16.4 ms
    { "Qwen3-1.7B_Q4_K_M", GGML_TYPE_Q4_K, 1024, 2048, 0, 0, 48 },  // 15.7 ms
    { "gemma-4-26B-A4B-it", GGML_TYPE_Q8_0, 2816, 8192, 0, 0, 5 },  // 15.1 ms
    { "Qwen2.5-3B-Instruc", GGML_TYPE_Q4_K, 256, 2048, 0, 0, 135 },  // 15.0 ms
    { "gemma-4-E2B-it_Q4_", GGML_TYPE_Q4_K, 1536, 2048, 0, 0, 28 },  // 14.0 ms
    { "gemma-4-E2B-it_Q4_", GGML_TYPE_Q4_K, 2048, 1536, 0, 0, 28 },  // 13.9 ms
    { "Qwen2.5-7B-Instruc", GGML_TYPE_Q4_K, 512, 3584, 0, 0, 42 },  // 13.6 ms
    { "Qwen3.5-35B-A3B_Q4", GGML_TYPE_Q8_0, 2048, 512, 0, 0, 80 },  // 13.0 ms
    { "Qwen3.6-35B-A3B_UD", GGML_TYPE_Q6_K, 2048, 512, 256, 8, 3 },  // 12.8 ms
    { "GLM-4.7-Flash_Q4_K", GGML_TYPE_Q4_K, 768, 2048, 0, 0, 47 },  // 12.0 ms
    { "gemma-4-E2B-it_Q4_", GGML_TYPE_Q4_K, 1536, 6144, 0, 0, 8 },  // 11.8 ms
    { "Qwen2.5-0.5B-Instr", GGML_TYPE_Q5_0, 896, 896, 0, 0, 48 },  // 9.4 ms
    { "GLM-4.7-Flash_Q4_K", GGML_TYPE_Q8_0, 576, 2048, 0, 0, 47 },  // 9.2 ms
    { "Qwen2.5-0.5B-Instr", GGML_TYPE_Q4_K, 896, 4864, 0, 0, 11 },  // 8.7 ms
    { "gemma-4-E2B-it_Q4_", GGML_TYPE_Q4_K, 4096, 1536, 0, 0, 7 },  // 7.0 ms
    { "Qwen3.5-0.8B_Q4_K_", GGML_TYPE_Q4_K, 1024, 3584, 0, 0, 12 },  // 6.9 ms
    { "GLM-4.7-Flash_Q4_K", GGML_TYPE_Q4_K, 10240, 2048, 0, 0, 2 },  // 6.7 ms
    { "Qwen3.5-0.8B_Q4_K_", GGML_TYPE_Q5_K, 1024, 2048, 0, 0, 18 },  // 6.7 ms
    { "gemma-4-E2B-it_Q4_", GGML_TYPE_Q4_K, 1536, 4096, 0, 0, 7 },  // 6.5 ms
    { "Qwen3.5-0.8B_Q4_K_", GGML_TYPE_Q4_K, 2048, 1024, 0, 0, 18 },  // 6.2 ms
    { "gemma-4-26B-A4B-it", GGML_TYPE_Q8_0, 2816, 704, 128, 8, 1 },  // 4.3 ms
    { "Qwen3.5-0.8B_Q4_K_", GGML_TYPE_Q4_K, 4096, 1024, 0, 0, 6 },  // 4.1 ms
    { "Qwen3.5-35B-A3B_Q4", GGML_TYPE_Q8_0, 32, 2048, 0, 0, 60 },  // 2.5 ms
    { "gemma-4-26B-A4B-it", GGML_TYPE_Q8_0, 1024, 2816, 0, 0, 5 },  // 2.2 ms
    { "gemma-4-E2B-it_Q4_", GGML_TYPE_Q4_K, 256, 1536, 0, 0, 18 },  // 1.5 ms
    { "Qwen2.5-0.5B-Instr", GGML_TYPE_Q5_0, 128, 896, 0, 0, 36 },  // 1.5 ms
    { "Qwen3.5-0.8B_Q4_K_", GGML_TYPE_Q8_0, 16, 1024, 0, 0, 36 },  // 0.9 ms
    { "Qwen3.5-0.8B_Q4_K_", GGML_TYPE_Q4_K, 512, 1024, 0, 0, 8 },  // 0.7 ms
    { "gemma-4-E2B-it_Q4_", GGML_TYPE_Q4_K, 512, 1536, 0, 0, 5 },  // 0.7 ms
    { "Qwen2.5-0.5B-Instr", GGML_TYPE_Q8_0, 128, 896, 0, 0, 12 },  // 0.4 ms
    // MMQ only at smaller batches - at -ub 2048 these two are dispatched to hipBLASLt instead,
    // so the capture above does not see them. Kept because the sweep also covers small M.
    { "GLM-4.7-Flash", GGML_TYPE_Q6_K, 2048, 1536, 0, 0, 450 },
    { "GLM-4.7-Flash", GGML_TYPE_Q6_K, 2048, 10240, 0, 0, 10 },
};


static const int g_n_shapes = sizeof(g_shapes)/sizeof(g_shapes[0]);

static std::vector<int64_t> parse_i64_list(const char * s) {
    std::vector<int64_t> out;
    for (const char * p = s; *p; ) {
        char * end = nullptr;
        const int64_t v = strtoll(p, &end, 10);
        if (end == p) {
            break;
        }
        out.push_back(v);
        p = (*end == ',') ? end + 1 : end;
    }
    return out;
}

static void usage(const char * argv0) {
    printf("usage: %s [options]\n", argv0);
    printf("  -s <list>     shape indices to run (default: all)\n");
    printf("  -m <list>     token counts to sweep (default: 64,128,512,2048)\n");
    printf("  -x <list>     mmq_x values to sweep (default: 8,16,...,128)\n");
    printf("  -r <n>        timed iterations per config (default: 11)\n");
    printf("  -w <n>        warmup iterations per config (default: 3)\n");
    printf("  --zipf <s>    MoE routing skew (default: 2.0, fitted from real traces; 0 = uniform)\n");
    printf("  --dense-only  skip MoE shapes\n");
    printf("  --moe-only    skip dense shapes\n");
    printf("  --output <console|csv>\n");
    printf("  --list        print the shape catalog and exit\n");
}

int main(int argc, char ** argv) {
    std::vector<int64_t> shape_ids;
    std::vector<int64_t> ms = { 64, 128, 512, 2048 };
    std::vector<int>     mmq_xs;
    int   niter      = 11;
    int   nwarmup    = 3;
    float zipf       = 2.0f;
    bool  csv        = false;
    bool  dense_only = false;
    bool  moe_only   = false;

    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "-s" && i + 1 < argc) {
            shape_ids = parse_i64_list(argv[++i]);
        } else if (a == "-m" && i + 1 < argc) {
            ms = parse_i64_list(argv[++i]);
        } else if (a == "-x" && i + 1 < argc) {
            for (int64_t v : parse_i64_list(argv[++i])) {
                mmq_xs.push_back((int) v);
            }
        } else if (a == "-r" && i + 1 < argc) {
            niter = atoi(argv[++i]);
        } else if (a == "-w" && i + 1 < argc) {
            nwarmup = atoi(argv[++i]);
        } else if (a == "--zipf" && i + 1 < argc) {
            zipf = atof(argv[++i]);
        } else if (a == "--dense-only") {
            dense_only = true;
        } else if (a == "--moe-only") {
            moe_only = true;
        } else if (a == "--output" && i + 1 < argc) {
            csv = std::string(argv[++i]) == "csv";
        } else if (a == "--list") {
            for (int s = 0; s < g_n_shapes; s++) {
                const shape & sh = g_shapes[s];
                printf("%2d  %-12s %-5s %-5s N=%-6" PRId64 " K=%-6" PRId64 " E=%-4d topk=%-3d w=%d\n",
                       s, sh.model, ggml_type_name(sh.type), sh.n_experts ? "moe" : "dense",
                       sh.n, sh.k, sh.n_experts, sh.top_k, sh.weight);
            }
            return 0;
        } else {
            usage(argv[0]);
            return a == "-h" || a == "--help" ? 0 : 1;
        }
    }

    if (shape_ids.empty()) {
        for (int s = 0; s < g_n_shapes; s++) {
            const bool is_moe = g_shapes[s].n_experts > 0;
            if ((dense_only && is_moe) || (moe_only && !is_moe)) {
                continue;
            }
            shape_ids.push_back(s);
        }
    }
    if (mmq_xs.empty()) {
        for (int x = 8; x <= 128; x += 8) {
            mmq_xs.push_back(x);
        }
    }
    // The heuristic's choice is the baseline every other config is compared against.
    mmq_xs.insert(mmq_xs.begin(), GGML_MMQ_TUNE_MMQ_X_AUTO);

    const int mmq_y  = ggml_mmq_tune_mmq_y();
    const int nwarps = ggml_mmq_tune_nwarps();

    if (csv) {
        printf("model,kind,type,n,k,m,n_experts,top_k,zipf_s,weight,mmq_y,nwarps,mmq_x,"
               "us_median,us_min,tflops,checksum\n");
    } else {
        printf("mmq_y=%d nwarps=%d zipf_s=%.2f\n\n", mmq_y, nwarps, zipf);
        printf("%-12s %-5s %-5s %7s %7s %6s | %10s | %5s %10s %8s\n",
               "model", "kind", "type", "N", "K", "M", "auto us", "best", "us", "speedup");
    }

    std::vector<ggml_mmq_tune_point> pts(mmq_xs.size());

    for (int64_t sid : shape_ids) {
        if (sid < 0 || sid >= g_n_shapes) {
            fprintf(stderr, "bad shape index %" PRId64 "\n", sid);
            return 1;
        }
        const shape & sh = g_shapes[sid];
        const bool is_moe = sh.n_experts > 0;

        for (int64_t m : ms) {
            const ggml_mmq_tune_case tc = {
                sh.type, sh.n, sh.k, m, sh.n_experts, sh.top_k, is_moe ? zipf : 0.0f, nwarmup, niter
            };

            if (!ggml_mmq_tune_sweep(&tc, mmq_xs.data(), (int) mmq_xs.size(), pts.data())) {
                fprintf(stderr, "sweep failed for shape %" PRId64 " m=%" PRId64 "\n", sid, m);
                return 1;
            }

            // MoE does top_k times the work of a dense GEMM of the same N,K per token.
            const double flop = 2.0*m*sh.n*sh.k*(is_moe ? sh.top_k : 1);

            if (csv) {
                for (const auto & p : pts) {
                    if (!p.valid) {
                        continue;
                    }
                    printf("%s,%s,%s,%" PRId64 ",%" PRId64 ",%" PRId64 ",%d,%d,%.2f,%d,%d,%d,%d,"
                           "%.3f,%.3f,%.4f,%.9e\n",
                           sh.model, is_moe ? "moe" : "dense", ggml_type_name(sh.type),
                           sh.n, sh.k, m, sh.n_experts, sh.top_k, is_moe ? zipf : 0.0f, sh.weight,
                           mmq_y, nwarps, p.mmq_x, p.us_median, p.us_min,
                           flop/(p.us_median*1e6), p.checksum);
                }
                fflush(stdout);
                continue;
            }

            const double us_auto = pts[0].us_median;

            int    best_x  = 0;
            double best_us = 0.0;
            for (size_t i = 1; i < pts.size(); i++) {
                if (pts[i].valid && (best_x == 0 || pts[i].us_median < best_us)) {
                    best_x  = pts[i].mmq_x;
                    best_us = pts[i].us_median;
                }
            }

            printf("%-12s %-5s %-5s %7" PRId64 " %7" PRId64 " %6" PRId64 " | %10.2f | %5d %10.2f %7.2fx\n",
                   sh.model, is_moe ? "moe" : "dense", ggml_type_name(sh.type), sh.n, sh.k, m,
                   us_auto, best_x, best_us, best_us > 0.0 ? us_auto/best_us : 0.0);
            fflush(stdout);
        }
    }

    return 0;
}
