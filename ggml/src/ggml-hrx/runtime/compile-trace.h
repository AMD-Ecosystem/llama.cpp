#pragma once

// Compilation requests copy these IDs to preserve their origin on JIT workers.
// All output is opt-in; execution IDs describe backend calls, not token steps.

#include <cstdint>
#include <string>

namespace ggml::hrx {

struct Dispatch;
struct KernelDefinition;
struct LoomKernelCompileRequest;

struct CompileTraceIds {
    uint64_t cache_id       = 0;
    uint64_t execution_id   = 0;
    uint64_t compilation_id = 0;
};

bool     compile_trace_enabled();
uint64_t compile_trace_next_id();
uint64_t compile_trace_now_ns();

struct CompileTraceExecution {
    CompileTraceExecution(uint64_t cache_id, uint64_t graph_uid);
    ~CompileTraceExecution();

    CompileTraceExecution(const CompileTraceExecution &) = delete;
    CompileTraceExecution & operator=(const CompileTraceExecution &) = delete;

    CompileTraceIds ids;
    uint64_t       compilations = 0;
    bool           success      = false;

  private:
    uint64_t start_ns_ = 0;
};

void trace_compile_request(const std::string & key, const KernelDefinition & definition,
                           const Dispatch & dispatch, const LoomKernelCompileRequest & request);
uint64_t trace_compile_begin(const CompileTraceIds & ids);
void trace_compile_end(const CompileTraceIds & ids, uint64_t start_ns, bool success, const std::string & error);
void trace_compile_wait(const CompileTraceIds & ids, uint64_t waiting_execution_id, uint64_t start_ns, bool success);
void trace_compile_cache_clear(uint64_t cache_id);

}  // namespace ggml::hrx
