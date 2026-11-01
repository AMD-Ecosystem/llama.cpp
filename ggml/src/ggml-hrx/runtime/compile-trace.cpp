#include "compile-trace.h"

#include "dispatch/dispatch.h"
#include "kernel-corpus/kernel-corpus.h"
#include "loom-kernel-jit.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace ggml::hrx {
namespace {

using json = nlohmann::ordered_json;

void write_event(const char * event, const CompileTraceIds & ids, json fields) {
#ifdef _WIN32
    const int pid = _getpid();
#else
    const int pid = getpid();
#endif
    json record = {
        { "event", event },
        { "pid", pid },
        { "time_ns", compile_trace_now_ns() },
        { "cache_id", ids.cache_id },
        { "execution_id", ids.execution_id },
        { "compilation_id", ids.compilation_id },
    };
    record.update(fields);
    // One stdio call keeps records from compiler workers on separate lines.
    const std::string line = record.dump();
    std::fprintf(stderr, "hrx compile: %s\n", line.c_str());
}

}  // namespace

bool compile_trace_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_HRX_TRACE_COMPILE");
        return value != nullptr ? std::strcmp(value, "1") == 0 : false;
    }();
    return enabled;
}

uint64_t compile_trace_next_id() {
    static std::atomic<uint64_t> next{ 1 };
    return next.fetch_add(1, std::memory_order_relaxed);
}

uint64_t compile_trace_now_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

CompileTraceExecution::CompileTraceExecution(uint64_t cache_id, uint64_t graph_uid) {
    if (cache_id == 0) {
        return;
    }
    ids = { cache_id, compile_trace_next_id(), 0 };
    write_event("graph_begin", ids, { { "graph_uid", graph_uid } });
    start_ns_ = compile_trace_now_ns();
}

CompileTraceExecution::~CompileTraceExecution() {
    if (ids.cache_id == 0) {
        return;
    }
    const uint64_t duration_ns = compile_trace_now_ns() - start_ns_;
    write_event("graph_end", ids, {
        { "duration_ns", duration_ns }, { "compilations", compilations }, { "success", success },
    });
}

void trace_compile_request(const std::string & key, const KernelDefinition & definition,
                           const Dispatch & dispatch, const LoomKernelCompileRequest & request) {
    json configs = json::object();
    for (const auto & config : request.config_storage) {
        configs[config.first] = config.second;
    }
    write_event("compile_request", request.trace, {
        { "key", key },
        { "kernel", kernel_definition_name(definition) },
        { "workload_specialization", dispatch.kernel.workload_specialization == WorkloadSpecialization::Exact ?
                                         "exact" : "dynamic" },
        { "integer_parameters", dispatch.kernel.integer_parameters },
        { "compile_parameters", configs },
    });
}

uint64_t trace_compile_begin(const CompileTraceIds & ids) {
    write_event("compile_begin", ids, json::object());
    return compile_trace_now_ns();
}

void trace_compile_end(const CompileTraceIds & ids, uint64_t start_ns, bool success, const std::string & error) {
    const uint64_t duration_ns = start_ns != 0 ? compile_trace_now_ns() - start_ns : 0;
    write_event("compile_end", ids, {
        { "duration_ns", duration_ns }, { "compiler_started", start_ns != 0 },
        { "success", success }, { "error", error },
    });
}

void trace_compile_wait(const CompileTraceIds & ids, uint64_t waiting_execution_id, uint64_t start_ns, bool success) {
    const uint64_t duration_ns = compile_trace_now_ns() - start_ns;
    write_event("compile_wait", ids, {
        { "waiting_execution_id", waiting_execution_id }, { "duration_ns", duration_ns }, { "success", success },
    });
}

void trace_compile_cache_clear(uint64_t cache_id) {
    write_event("cache_clear", { cache_id, 0, 0 }, json::object());
}

}  // namespace ggml::hrx
