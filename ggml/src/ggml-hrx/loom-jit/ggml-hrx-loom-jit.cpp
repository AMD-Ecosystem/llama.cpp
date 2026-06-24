// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include "ggml-hrx-loom-jit.h"

#include "loomc/loomc.h"
#include "loomc/sanitizer.h"
#include "loomc/target/amdgpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <string>

struct ggml_hrx_loom_jit_amdgpu_s {
    loomc_target_environment_t *        target_environment = nullptr;
    loomc_context_t *                   context            = nullptr;
    loomc_target_profile_t *            target_profile     = nullptr;
    loomc_target_selection_t *          target_selection   = nullptr;
    loomc_compiler_t *                  compiler           = nullptr;
    loomc_pass_program_t *              pass_program       = nullptr;
    loomc_amdgpu_runtime_global_flags_t runtime_globals    = LOOMC_AMDGPU_RUNTIME_GLOBAL_NONE;
};

namespace {

template <typename T, void (*Release)(T *)> class LoomHandle {
  public:
    LoomHandle()                               = default;
    LoomHandle(const LoomHandle &)             = delete;
    LoomHandle & operator=(const LoomHandle &) = delete;

    ~LoomHandle() { reset(); }

    T * get() const { return value_; }

    T ** out() {
        reset();
        return &value_;
    }

    void reset(T * value = nullptr) {
        if (value_) {
            Release(value_);
        }
        value_ = value;
    }

  private:
    T * value_ = nullptr;
};

using LoomWorkspace        = LoomHandle<loomc_workspace_t, loomc_workspace_release>;
using LoomSource           = LoomHandle<loomc_source_t, loomc_source_release>;
using LoomModule           = LoomHandle<loomc_module_t, loomc_module_release>;
using LoomResult           = LoomHandle<loomc_result_t, loomc_result_release>;
using LoomLinkIndexBuilder = LoomHandle<loomc_link_index_builder_t, loomc_link_index_builder_release>;
using LoomLinkIndex        = LoomHandle<loomc_link_index_t, loomc_link_index_release>;
using LoomLinker           = LoomHandle<loomc_linker_t, loomc_linker_release>;

struct HrxLoomJitDeleter {
    void operator()(ggml_hrx_loom_jit_amdgpu_s * jit) const { ggml_hrx_loom_jit_amdgpu_release(jit); }
};

hrx_status_t ggml_hrx_loom_jit_make_status(hrx_status_code_t code, const char * message) {
    return hrx_make_status(code, message ? message : "GGML HRX Loom JIT failure");
}

hrx_status_code_t ggml_hrx_loom_jit_status_code_from_loom(loomc_status_code_t code) {
    switch (code) {
        case LOOMC_STATUS_OK:
            return HRX_STATUS_OK;
        case LOOMC_STATUS_CANCELLED:
            return HRX_STATUS_CANCELLED;
        case LOOMC_STATUS_UNKNOWN:
            return HRX_STATUS_UNKNOWN;
        case LOOMC_STATUS_INVALID_ARGUMENT:
            return HRX_STATUS_INVALID_ARGUMENT;
        case LOOMC_STATUS_DEADLINE_EXCEEDED:
            return HRX_STATUS_DEADLINE_EXCEEDED;
        case LOOMC_STATUS_NOT_FOUND:
            return HRX_STATUS_NOT_FOUND;
        case LOOMC_STATUS_ALREADY_EXISTS:
            return HRX_STATUS_ALREADY_EXISTS;
        case LOOMC_STATUS_PERMISSION_DENIED:
            return HRX_STATUS_PERMISSION_DENIED;
        case LOOMC_STATUS_RESOURCE_EXHAUSTED:
            return HRX_STATUS_OUT_OF_MEMORY;
        case LOOMC_STATUS_FAILED_PRECONDITION:
            return HRX_STATUS_FAILED_PRECONDITION;
        case LOOMC_STATUS_ABORTED:
            return HRX_STATUS_ABORTED;
        case LOOMC_STATUS_OUT_OF_RANGE:
            return HRX_STATUS_OUT_OF_RANGE;
        case LOOMC_STATUS_UNIMPLEMENTED:
            return HRX_STATUS_UNIMPLEMENTED;
        case LOOMC_STATUS_INTERNAL:
            return HRX_STATUS_INTERNAL;
        case LOOMC_STATUS_UNAVAILABLE:
            return HRX_STATUS_UNAVAILABLE;
        case LOOMC_STATUS_DATA_LOSS:
            return HRX_STATUS_DATA_LOSS;
        case LOOMC_STATUS_UNAUTHENTICATED:
            return HRX_STATUS_PERMISSION_DENIED;
        case LOOMC_STATUS_DEFERRED:
            return HRX_STATUS_UNAVAILABLE;
        case LOOMC_STATUS_INCOMPATIBLE:
            return HRX_STATUS_FAILED_PRECONDITION;
        case LOOMC_STATUS_CODE_MASK:
            return HRX_STATUS_INTERNAL;
    }
    return HRX_STATUS_INTERNAL;
}

std::string ggml_hrx_loom_jit_format_status(loomc_status_t status) {
    if (loomc_status_is_ok(status)) {
        return "OK";
    }
    loomc_host_size_t length = 0;
    loomc_status_format(status, 0, nullptr, &length);
    std::unique_ptr<char[]> buffer(new (std::nothrow) char[length + 1]());
    if (!buffer) {
        char fallback[4096] = { 0 };
        loomc_host_size_t fallback_length = 0;
        loomc_status_format(status, sizeof(fallback), fallback, &fallback_length);
        if (fallback_length >= sizeof(fallback)) {
            fallback_length = sizeof(fallback) - 1;
        }
        return std::string(fallback, fallback_length);
    }
    loomc_host_size_t actual_length = 0;
    loomc_status_format(status, length + 1, buffer.get(), &actual_length);
    return std::string(buffer.get(), actual_length);
}

void ggml_hrx_loom_jit_spam_failure(const char * context, const std::string & message) {
    std::fprintf(stderr, "HRX Loom JIT %s failed: %s\n", context ? context : "operation", message.c_str());
    std::fflush(stderr);
}

hrx_status_t ggml_hrx_loom_jit_status_from_loom(loomc_status_t status, const char * context) {
    if (loomc_status_is_ok(status)) {
        return hrx_ok_status();
    }
    const hrx_status_code_t hrx_code = ggml_hrx_loom_jit_status_code_from_loom(loomc_status_code(status));
    const std::string       formatted_status = ggml_hrx_loom_jit_format_status(status);
    loomc_status_free(status);
    std::string message = std::string(context ? context : "loomc") + ": " + formatted_status;
    ggml_hrx_loom_jit_spam_failure(context, message);
    return ggml_hrx_loom_jit_make_status(hrx_code, message.c_str());
}

hrx_status_t ggml_hrx_loom_jit_status_from_result(const loomc_result_t * result, const char * context) {
    if (result && loomc_result_succeeded(result)) {
        return hrx_ok_status();
    }
    std::string message = context ? context : "loomc result failed";
    if (result) {
        const loomc_host_size_t diagnostic_count = loomc_result_diagnostic_count(result);
        for (loomc_host_size_t i = 0; i < diagnostic_count; ++i) {
            const loomc_diagnostic_t * diagnostic = loomc_result_diagnostic_at(result, i);
            if (!diagnostic) {
                continue;
            }
            message += "\n  diagnostic[";
            message += std::to_string(static_cast<unsigned long long>(i));
            message += "] ";
            message.append(diagnostic->code.data, diagnostic->code.size);
            message += ": ";
            message.append(diagnostic->message.data, diagnostic->message.size);
            if (diagnostic->range.start_line || diagnostic->range.start_column) {
                message += " @ ";
                message += std::to_string(diagnostic->range.start_line);
                message += ":";
                message += std::to_string(diagnostic->range.start_column);
            }
        }
    }
    ggml_hrx_loom_jit_spam_failure(context, message);
    return ggml_hrx_loom_jit_make_status(HRX_STATUS_FAILED_PRECONDITION, message.c_str());
}

void * ggml_hrx_loom_jit_malloc_copy(const void * data, size_t size, bool nul_terminate) {
    if (!data || size == 0) {
        return nullptr;
    }
    const size_t alloc_size = nul_terminate ? size + 1 : size;
    void *       result     = nullptr;
    hrx_status_t status     = hrx_host_allocator_malloc_uninitialized(hrx_host_allocator_system(), alloc_size, &result);
    if (!hrx_status_is_ok(status)) {
        hrx_status_ignore(status);
        return nullptr;
    }
    std::memcpy(result, data, size);
    if (nul_terminate) {
        static_cast<char *>(result)[size] = 0;
    }
    return result;
}

const loomc_artifact_t * ggml_hrx_loom_jit_find_artifact(const loomc_result_t * result,
                                                          loomc_artifact_kind_t  kind,
                                                          loomc_string_view_t    format) {
    for (loomc_host_size_t i = 0; i < loomc_result_artifact_count(result); ++i) {
        const loomc_artifact_t * artifact = loomc_result_artifact_at(result, i);
        if (!artifact) {
            continue;
        }
        if (artifact->kind == kind && loomc_string_view_equal(artifact->format, format)) {
            return artifact;
        }
    }
    return nullptr;
}

hrx_status_t ggml_hrx_loom_jit_copy_artifact_bytes(const loomc_artifact_t * artifact,
                                                    void **                  out_data,
                                                    size_t *                 out_size,
                                                    bool                     nul_terminate) {
    if (out_data) {
        *out_data = nullptr;
    }
    if (out_size) {
        *out_size = 0;
    }
    if (!artifact || !out_data || !out_size) {
        return hrx_ok_status();
    }
    void * copy =
        ggml_hrx_loom_jit_malloc_copy(artifact->contents.data, artifact->contents.data_length, nul_terminate);
    if (!copy) {
        return ggml_hrx_loom_jit_make_status(HRX_STATUS_OUT_OF_MEMORY, "failed to copy Loom artifact");
    }
    *out_data = copy;
    *out_size = artifact->contents.data_length;
    return hrx_ok_status();
}

hrx_status_t ggml_hrx_loom_jit_parse_sanitizer_checks(const char * value, loomc_sanitizer_checks_t * out_checks) {
    *out_checks = 0;
    if (!value || !value[0] || std::strcmp(value, "0") == 0 || std::strcmp(value, "none") == 0) {
        return hrx_ok_status();
    }
    if (std::strcmp(value, "access") == 0 || std::strcmp(value, "asan") == 0) {
        *out_checks = LOOMC_SANITIZER_CHECKS_ASAN_LIKE;
        return hrx_ok_status();
    }
    if (std::strcmp(value, "value") == 0) {
        *out_checks = LOOMC_SANITIZER_CHECK_VALUE;
        return hrx_ok_status();
    }
    if (std::strcmp(value, "operation") == 0) {
        *out_checks = LOOMC_SANITIZER_CHECK_OPERATION;
        return hrx_ok_status();
    }
    if (std::strcmp(value, "ubsan") == 0) {
        *out_checks = LOOMC_SANITIZER_CHECKS_UBSAN_LIKE;
        return hrx_ok_status();
    }
    if (std::strcmp(value, "all") == 0) {
        *out_checks = LOOMC_SANITIZER_CHECK_ACCESS | LOOMC_SANITIZER_CHECK_VALUE | LOOMC_SANITIZER_CHECK_OPERATION;
        return hrx_ok_status();
    }
    char message[256] = { 0 };
    std::snprintf(message, sizeof(message), "unsupported GGML_HRX_LOOM_SANITIZER '%s'", value);
    return ggml_hrx_loom_jit_make_status(HRX_STATUS_INVALID_ARGUMENT, message);
}

hrx_status_t ggml_hrx_loom_jit_parse_sanitizer_reporting(const char *                       value,
                                                          loomc_sanitizer_reporting_mode_t * out_reporting_mode) {
    *out_reporting_mode = LOOMC_SANITIZER_REPORTING_MODE_REPORT_ONLY;
    if (!value || !value[0] || std::strcmp(value, "report-only") == 0 || std::strcmp(value, "report_only") == 0 ||
        std::strcmp(value, "report") == 0) {
        return hrx_ok_status();
    }
    if (std::strcmp(value, "default") == 0) {
        *out_reporting_mode = LOOMC_SANITIZER_REPORTING_MODE_DEFAULT;
        return hrx_ok_status();
    }
    if (std::strcmp(value, "trap") == 0) {
        *out_reporting_mode = LOOMC_SANITIZER_REPORTING_MODE_TRAP;
        return hrx_ok_status();
    }
    char message[256] = { 0 };
    std::snprintf(message, sizeof(message), "unsupported GGML_HRX_LOOM_SANITIZER_REPORTING '%s'", value);
    return ggml_hrx_loom_jit_make_status(HRX_STATUS_INVALID_ARGUMENT, message);
}

loomc_amdgpu_runtime_global_flags_t ggml_hrx_loom_jit_runtime_globals(loomc_sanitizer_checks_t sanitizer_checks) {
    if (!sanitizer_checks) {
        return LOOMC_AMDGPU_RUNTIME_GLOBAL_NONE;
    }
    loomc_amdgpu_runtime_global_flags_t runtime_globals = LOOMC_AMDGPU_RUNTIME_GLOBAL_FEEDBACK_CONFIG;
    if (sanitizer_checks & LOOMC_SANITIZER_CHECK_ACCESS) {
        runtime_globals |= LOOMC_AMDGPU_RUNTIME_GLOBAL_ASAN_CONFIG;
    }
    return runtime_globals;
}

}  // namespace

hrx_status_t ggml_hrx_loom_jit_amdgpu_create(const ggml_hrx_loom_jit_amdgpu_options_t * options,
                                              ggml_hrx_loom_jit_amdgpu_t *               out_jit) {
    if (!out_jit) {
        return ggml_hrx_loom_jit_make_status(HRX_STATUS_INVALID_ARGUMENT, "out_jit must not be NULL");
    }
    *out_jit = nullptr;
    if (!options || options->structure_size < sizeof(*options) || !options->processor || options->processor[0] == 0) {
        return ggml_hrx_loom_jit_make_status(HRX_STATUS_INVALID_ARGUMENT,
                                              "valid ggml_hrx_loom_jit_amdgpu_options_t with processor is required");
    }

    std::unique_ptr<ggml_hrx_loom_jit_amdgpu_s, HrxLoomJitDeleter> jit(new (std::nothrow)
                                                                            ggml_hrx_loom_jit_amdgpu_s());
    if (!jit) {
        return ggml_hrx_loom_jit_make_status(HRX_STATUS_OUT_OF_MEMORY, "failed to allocate GGML HRX Loom JIT");
    }

    LoomResult     result;
    loomc_status_t status = loomc_target_environment_create_amdgpu(loomc_allocator_system(), &jit->target_environment);
    if (!loomc_status_is_ok(status)) {
        return ggml_hrx_loom_jit_status_from_loom(status, "create AMDGPU target environment");
    }

    loomc_context_target_options_t target_options = {};
    target_options.type                           = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS;
    target_options.structure_size                 = sizeof(target_options);
    target_options.target_environment             = jit->target_environment;
    loomc_context_options_t context_options       = {};
    context_options.type                          = LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS;
    context_options.structure_size                = sizeof(context_options);
    context_options.next                          = &target_options;
    status = loomc_context_create(&context_options, loomc_allocator_system(), &jit->context);
    if (!loomc_status_is_ok(status)) {
        return ggml_hrx_loom_jit_status_from_loom(status, "create Loom context");
    }

    loomc_amdgpu_profile_options_t profile_options = {};
    profile_options.type                           = LOOMC_STRUCTURE_TYPE_AMDGPU_PROFILE_OPTIONS;
    profile_options.structure_size                 = sizeof(profile_options);
    profile_options.identifier                     = loomc_make_cstring_view(options->identifier);
    profile_options.processor                      = loomc_make_cstring_view(options->processor);
    status = loomc_target_profile_create_amdgpu(jit->target_environment, &profile_options, loomc_allocator_system(),
                                                &jit->target_profile);
    if (!loomc_status_is_ok(status)) {
        return ggml_hrx_loom_jit_status_from_loom(status, "create AMDGPU target profile");
    }
    status = loomc_target_selection_create_from_profile(jit->target_profile, loomc_allocator_system(),
                                                        &jit->target_selection);
    if (!loomc_status_is_ok(status)) {
        return ggml_hrx_loom_jit_status_from_loom(status, "create target selection");
    }
    status = loomc_compiler_create(jit->context, nullptr, loomc_allocator_system(), &jit->compiler);
    if (!loomc_status_is_ok(status)) {
        return ggml_hrx_loom_jit_status_from_loom(status, "create Loom compiler");
    }

    loomc_target_selection_options_t pipeline_target_options = {};
    pipeline_target_options.type                             = LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS;
    pipeline_target_options.structure_size                   = sizeof(pipeline_target_options);
    pipeline_target_options.target_selection                 = jit->target_selection;
    loomc_sanitizer_options_t sanitizer_options              = {};
    sanitizer_options.type                                   = LOOMC_STRUCTURE_TYPE_SANITIZER_OPTIONS;
    sanitizer_options.structure_size                         = sizeof(sanitizer_options);
    sanitizer_options.next                                   = &pipeline_target_options;
    hrx_status_t sanitizer_status =
        ggml_hrx_loom_jit_parse_sanitizer_checks(options->sanitizer, &sanitizer_options.checks);
    if (!hrx_status_is_ok(sanitizer_status)) {
        return sanitizer_status;
    }
    if (sanitizer_options.checks) {
        hrx_status_t sanitizer_reporting_status = ggml_hrx_loom_jit_parse_sanitizer_reporting(
            options->sanitizer_reporting, &sanitizer_options.reporting_mode);
        if (!hrx_status_is_ok(sanitizer_reporting_status)) {
            return sanitizer_reporting_status;
        }
    }
    jit->runtime_globals                             = ggml_hrx_loom_jit_runtime_globals(sanitizer_options.checks);
    loomc_target_pipeline_options_t pipeline_options = {};
    pipeline_options.type                            = LOOMC_STRUCTURE_TYPE_TARGET_PIPELINE_OPTIONS;
    pipeline_options.structure_size                  = sizeof(pipeline_options);
    pipeline_options.next       = sanitizer_options.checks ? static_cast<const void *>(&sanitizer_options) :
                                                             static_cast<const void *>(&pipeline_target_options);
    pipeline_options.identifier = loomc_make_cstring_view("ggml-hrx-amdgpu-jit-prepared-low");
    pipeline_options.kind       = LOOMC_TARGET_PIPELINE_KIND_PREPARED_LOW;
    pipeline_options.control_flow_lowering    = LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG;
    pipeline_options.source_to_low_max_errors = 20;
    status = loomc_pass_program_create_from_target_pipeline(jit->context, &pipeline_options, loomc_allocator_system(),
                                                            &jit->pass_program, result.out());
    if (!loomc_status_is_ok(status)) {
        return ggml_hrx_loom_jit_status_from_loom(status, "create target pass program");
    }
    if (!loomc_result_succeeded(result.get())) {
        return ggml_hrx_loom_jit_status_from_result(result.get(), "target pass program failed");
    }

    *out_jit = jit.release();
    return hrx_ok_status();
}

void ggml_hrx_loom_jit_amdgpu_release(ggml_hrx_loom_jit_amdgpu_t jit) {
    if (!jit) {
        return;
    }
    loomc_pass_program_release(jit->pass_program);
    loomc_compiler_release(jit->compiler);
    loomc_target_selection_release(jit->target_selection);
    loomc_target_profile_release(jit->target_profile);
    loomc_context_release(jit->context);
    loomc_target_environment_release(jit->target_environment);
    delete jit;
}

hrx_status_t ggml_hrx_loom_jit_amdgpu_compile(ggml_hrx_loom_jit_amdgpu_t                  jit,
                                               const ggml_hrx_loom_jit_compile_options_t * options,
                                               ggml_hrx_loom_jit_compile_result_t *        out_result) {
    if (!out_result) {
        return ggml_hrx_loom_jit_make_status(HRX_STATUS_INVALID_ARGUMENT, "out_result must not be NULL");
    }
    std::memset(out_result, 0, sizeof(*out_result));
    if (!jit || !options || options->structure_size < sizeof(*options) || !options->source_data ||
        options->source_size == 0 || !options->root_symbol || options->root_symbol[0] == 0) {
        return ggml_hrx_loom_jit_make_status(
            HRX_STATUS_INVALID_ARGUMENT, "valid GGML HRX Loom JIT compile options with source and root are required");
    }
    if (options->config_binding_count > 0 && !options->config_bindings) {
        return ggml_hrx_loom_jit_make_status(HRX_STATUS_INVALID_ARGUMENT,
                                              "GGML HRX Loom JIT config binding count requires config bindings");
    }
    for (size_t i = 0; i < options->config_binding_count; ++i) {
        if (!options->config_bindings[i].key || !options->config_bindings[i].value) {
            return ggml_hrx_loom_jit_make_status(HRX_STATUS_INVALID_ARGUMENT,
                                                  "GGML HRX Loom JIT config binding keys and values must not be NULL");
        }
    }

    std::unique_ptr<loomc_config_binding_t[]> config_bindings;
    if (options->config_binding_count > 0) {
        config_bindings.reset(new (std::nothrow) loomc_config_binding_t[options->config_binding_count]());
        if (!config_bindings) {
            return ggml_hrx_loom_jit_make_status(HRX_STATUS_OUT_OF_MEMORY,
                                                  "failed to allocate GGML HRX Loom JIT config bindings");
        }
        for (size_t i = 0; i < options->config_binding_count; ++i) {
            config_bindings[i].key   = loomc_make_cstring_view(options->config_bindings[i].key);
            config_bindings[i].value = loomc_make_cstring_view(options->config_bindings[i].value);
        }
    }

    LoomWorkspace workspace;
    LoomSource    source;
    LoomModule    module;
    LoomResult    result;

    loomc_status_t status = loomc_workspace_create(nullptr, loomc_allocator_system(), workspace.out());
    if (!loomc_status_is_ok(status)) {
        return ggml_hrx_loom_jit_status_from_loom(status, "create Loom workspace");
    }

    loomc_source_options_t source_options = {};
    source_options.type                   = LOOMC_STRUCTURE_TYPE_SOURCE_OPTIONS;
    source_options.structure_size         = sizeof(source_options);
    source_options.format                 = options->source_format == GGML_HRX_LOOM_JIT_SOURCE_FORMAT_BYTECODE ?
                                                LOOMC_SOURCE_FORMAT_BYTECODE :
                                                LOOMC_SOURCE_FORMAT_TEXT;
    source_options.identifier             = loomc_make_cstring_view(options->source_identifier);
    source_options.contents               = loomc_make_byte_span(options->source_data, options->source_size);
    source_options.storage                = LOOMC_SOURCE_STORAGE_BORROWED;
    status = loomc_source_create(&source_options, loomc_allocator_system(), source.out());
    if (!loomc_status_is_ok(status)) {
        return ggml_hrx_loom_jit_status_from_loom(status, "create Loom source");
    }

    LoomLinkIndexBuilder link_index_builder;
    status = loomc_link_index_builder_create(jit->context, nullptr, loomc_allocator_system(), link_index_builder.out());
    if (!loomc_status_is_ok(status)) {
        return ggml_hrx_loom_jit_status_from_loom(status, "create Loom link index builder");
    }
    loomc_link_index_source_options_t link_source_options = {};
    link_source_options.provider_name                     = loomc_make_cstring_view(options->source_identifier);
    link_source_options.role                              = LOOMC_LINK_PROVIDER_ROLE_INPUT;
    status = loomc_link_index_builder_add_source(link_index_builder.get(), source.get(), &link_source_options, nullptr);
    if (!loomc_status_is_ok(status)) {
        return ggml_hrx_loom_jit_status_from_loom(status, "index Loom source");
    }
    LoomLinkIndex link_index;
    status = loomc_link_index_builder_finish(link_index_builder.get(), link_index.out(), result.out());
    if (!loomc_status_is_ok(status)) {
        return ggml_hrx_loom_jit_status_from_loom(status, "finish Loom link index");
    }
    if (!loomc_result_succeeded(result.get())) {
        return ggml_hrx_loom_jit_status_from_result(result.get(), "Loom source indexing failed");
    }
    result.reset();

    LoomLinker linker;
    status = loomc_linker_create(jit->context, nullptr, loomc_allocator_system(), linker.out());
    if (!loomc_status_is_ok(status)) {
        return ggml_hrx_loom_jit_status_from_loom(status, "create Loom linker");
    }

    loomc_target_selection_options_t link_target_options = {};
    link_target_options.type                             = LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS;
    link_target_options.structure_size                   = sizeof(link_target_options);
    link_target_options.target_selection                 = jit->target_selection;
    const loomc_string_view_t root_symbols[]             = {
        loomc_make_cstring_view(options->root_symbol),
    };
    loomc_link_options_t link_options = {};
    link_options.type                 = LOOMC_STRUCTURE_TYPE_LINK_OPTIONS;
    link_options.structure_size       = sizeof(link_options);
    link_options.next                 = &link_target_options;
    link_options.link_index           = link_index.get();
    link_options.module_name          = loomc_make_cstring_view(options->module_name);
    link_options.root_symbols         = root_symbols;
    link_options.root_symbol_count    = 1;
    link_options.flags                = LOOMC_LINK_FLAG_STRIP_CHECK_SYMBOLS;
    status = loomc_link_module(linker.get(), workspace.get(), &link_options, module.out(), result.out());
    if (!loomc_status_is_ok(status)) {
        return ggml_hrx_loom_jit_status_from_loom(status, "link Loom root");
    }
    if (!loomc_result_succeeded(result.get())) {
        return ggml_hrx_loom_jit_status_from_result(result.get(), "Loom root linking failed");
    }
    result.reset();

    loomc_target_selection_options_t compile_target_options = {};
    compile_target_options.type                             = LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS;
    compile_target_options.structure_size                   = sizeof(compile_target_options);
    compile_target_options.target_selection                 = jit->target_selection;
    loomc_compile_options_t compile_options                 = {};
    compile_options.type                                    = LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS;
    compile_options.structure_size                          = sizeof(compile_options);
    compile_options.next                                    = &compile_target_options;
    compile_options.module_name                             = loomc_make_cstring_view(options->module_name);
    compile_options.config.bindings                         = config_bindings.get();
    compile_options.config.binding_count                    = options->config_binding_count;
    compile_options.config.flags                            = LOOMC_CONFIG_POLICY_FLAG_REQUIRE_RESOLVED;
    status = loomc_compile_module(jit->compiler, workspace.get(), jit->pass_program, module.get(), &compile_options,
                                  loomc_allocator_system(), result.out());
    if (!loomc_status_is_ok(status)) {
        return ggml_hrx_loom_jit_status_from_loom(status, "compile Loom module");
    }
    if (!loomc_result_succeeded(result.get())) {
        return ggml_hrx_loom_jit_status_from_result(result.get(), "Loom compilation failed");
    }

    const loomc_artifact_t * compile_report = ggml_hrx_loom_jit_find_artifact(
        result.get(), LOOMC_ARTIFACT_KIND_REPORT, loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_JSON));
    hrx_status_t hrx_status = ggml_hrx_loom_jit_copy_artifact_bytes(
        compile_report, reinterpret_cast<void **>(&out_result->compile_report_json),
        &out_result->compile_report_json_size, true);
    if (!hrx_status_is_ok(hrx_status)) {
        return hrx_status;
    }
    result.reset();

    loomc_target_selection_options_t emit_target_options = {};
    emit_target_options.type                             = LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS;
    emit_target_options.structure_size                   = sizeof(emit_target_options);
    emit_target_options.target_selection                 = jit->target_selection;
    loomc_amdgpu_emit_options_t amdgpu_options           = {};
    amdgpu_options.type                                  = LOOMC_STRUCTURE_TYPE_AMDGPU_EMIT_OPTIONS;
    amdgpu_options.structure_size                        = sizeof(amdgpu_options);
    amdgpu_options.next                                  = &emit_target_options;
    amdgpu_options.runtime_globals                       = jit->runtime_globals;
    const loomc_option_entry_t emit_entries[]            = {
        {
         loomc_make_cstring_view(LOOMC_EMIT_OPTION_KEY_IDENTIFIER),
         loomc_make_cstring_view(options->artifact_identifier),
         },
    };
    loomc_option_dict_t option_dict                    = {};
    option_dict.type                                   = LOOMC_STRUCTURE_TYPE_OPTION_DICT;
    option_dict.structure_size                         = sizeof(option_dict);
    option_dict.next                                   = &amdgpu_options;
    option_dict.entries                                = emit_entries;
    option_dict.entry_count                            = options->artifact_identifier ? 1 : 0;
    loomc_artifact_manifest_options_t manifest_options = {};
    manifest_options.type                              = LOOMC_STRUCTURE_TYPE_ARTIFACT_MANIFEST_OPTIONS;
    manifest_options.structure_size                    = sizeof(manifest_options);
    manifest_options.next                              = &option_dict;
    manifest_options.mode                              = LOOMC_ARTIFACT_MANIFEST_MODE_DETAILS;
    loomc_compile_report_options_t report_options      = {};
    report_options.type                                = LOOMC_STRUCTURE_TYPE_COMPILE_REPORT_OPTIONS;
    report_options.structure_size                      = sizeof(report_options);
    report_options.next                                = &manifest_options;
    report_options.mode                                = LOOMC_COMPILE_REPORT_MODE_DETAILS;
    loomc_emit_options_t emit_options                  = {};
    emit_options.type                                  = LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS;
    emit_options.structure_size                        = sizeof(emit_options);
    emit_options.next                                  = &report_options;
    emit_options.artifact_format                       = loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO);
    emit_options.identifier                            = loomc_make_cstring_view(options->artifact_identifier);
    emit_options.artifact_flags                        = LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY;
    status = loomc_emit_module(jit->target_environment, workspace.get(), module.get(), &emit_options,
                               loomc_allocator_system(), result.out());
    if (!loomc_status_is_ok(status)) {
        ggml_hrx_loom_jit_compile_result_deinitialize(out_result);
        return ggml_hrx_loom_jit_status_from_loom(status, "emit AMDGPU HSACO");
    }
    if (!loomc_result_succeeded(result.get())) {
        hrx_status = ggml_hrx_loom_jit_status_from_result(result.get(), "AMDGPU HSACO emission failed");
        ggml_hrx_loom_jit_compile_result_deinitialize(out_result);
        return hrx_status;
    }

    const loomc_artifact_t * hsaco = ggml_hrx_loom_jit_find_artifact(
        result.get(), LOOMC_ARTIFACT_KIND_EXECUTABLE, loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_AMDGPU_HSACO));
    if (!hsaco) {
        ggml_hrx_loom_jit_compile_result_deinitialize(out_result);
        return ggml_hrx_loom_jit_make_status(HRX_STATUS_NOT_FOUND, "Loom did not return an AMDGPU HSACO artifact");
    }
    hrx_status = ggml_hrx_loom_jit_copy_artifact_bytes(hsaco, &out_result->hsaco_data, &out_result->hsaco_size, false);
    if (hrx_status_is_ok(hrx_status)) {
        const loomc_artifact_t * report =
            ggml_hrx_loom_jit_find_artifact(result.get(), LOOMC_ARTIFACT_KIND_REPORT,
                                             loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_COMPILE_REPORT_JSON));
        hrx_status =
            ggml_hrx_loom_jit_copy_artifact_bytes(report, reinterpret_cast<void **>(&out_result->compile_report_json),
                                                   &out_result->compile_report_json_size, true);
    }
    if (hrx_status_is_ok(hrx_status)) {
        const loomc_artifact_t * manifest =
            ggml_hrx_loom_jit_find_artifact(result.get(), LOOMC_ARTIFACT_KIND_REPORT,
                                             loomc_make_cstring_view(LOOMC_ARTIFACT_FORMAT_ARTIFACT_MANIFEST_JSON));
        hrx_status = ggml_hrx_loom_jit_copy_artifact_bytes(
            manifest, reinterpret_cast<void **>(&out_result->manifest_json), &out_result->manifest_json_size, true);
    }

    if (!hrx_status_is_ok(hrx_status)) {
        ggml_hrx_loom_jit_compile_result_deinitialize(out_result);
    }
    return hrx_status;
}

void ggml_hrx_loom_jit_compile_result_deinitialize(ggml_hrx_loom_jit_compile_result_t * result) {
    if (!result) {
        return;
    }
    hrx_host_allocator_t allocator = hrx_host_allocator_system();
    hrx_host_allocator_free(allocator, result->hsaco_data);
    hrx_host_allocator_free(allocator, result->manifest_json);
    hrx_host_allocator_free(allocator, result->compile_report_json);
    std::memset(result, 0, sizeof(*result));
}
