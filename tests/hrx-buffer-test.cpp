#include "backend-buffer-binding.h"
#include "backend-context.h"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-hrx.h"
#include "ggml-impl.h"
#include "ggml.h"
#include "hrx-interop-utils.h"
#include "runtime/host-memory.h"
#include "runtime/command-program-executor.h"
#include "testing_suite.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#if defined(__linux__)
#include <sys/mman.h>
#endif

#define REQUIRE(condition)                                                                           \
    do {                                                                                             \
        if (!(condition)) {                                                                          \
            std::fprintf(stderr, "%s:%d: requirement failed: %s\n", __FILE__, __LINE__, #condition); \
            std::abort();                                                                            \
        }                                                                                            \
    } while (false)

static void require_hrx_status(hrx_status_t status) {
    if (ggml::hrx::ErrorResult error = ggml::hrx::take_status(status)) {
        std::fprintf(stderr, "HRX status failed: %s\n", error->c_str());
        std::abort();
    }
}

static ggml_backend_hrx_context * backend_context(ggml_backend_t backend) {
    auto * context = static_cast<ggml_backend_hrx_context *>(backend->context);
    REQUIRE(context != nullptr);
    REQUIRE(context->device != nullptr);
    REQUIRE(context->device->device != nullptr);
    REQUIRE(context->stream != nullptr);
    return context;
}

static void run_backend_buffer_checks(ggml_backend_t backend) {
    ggml_backend_hrx_context * hrx = backend_context(backend);

    ggml_init_params params = {};
    params.mem_size         = 16 * 1024;
    params.no_alloc         = true;
    ggml_context * context  = ggml_init(params);
    REQUIRE(context != nullptr);
    ggml_tensor *         tensor      = ggml_new_tensor_1d(context, GGML_TYPE_I32, 64);
    ggml_tensor *         copy        = ggml_new_tensor_1d(context, GGML_TYPE_I32, 64);
    ggml_backend_buffer_t buffer      = ggml_backend_alloc_buffer(backend, 4096);
    ggml_backend_buffer_t copy_buffer = ggml_backend_alloc_buffer(backend, 4096);
    REQUIRE(buffer != nullptr);
    REQUIRE(copy_buffer != nullptr);
    tensor->buffer = buffer;
    tensor->data   = ggml_backend_buffer_get_base(buffer);
    copy->buffer   = copy_buffer;
    copy->data     = ggml_backend_buffer_get_base(copy_buffer);
    REQUIRE(ggml_backend_buffer_init_tensor(buffer, tensor) == GGML_STATUS_SUCCESS);
    REQUIRE(ggml_backend_buffer_init_tensor(copy_buffer, copy) == GGML_STATUS_SUCCESS);

    std::array<uint32_t, 64> input = {};
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<uint32_t>(i * 17 + 3);
    }
    ggml_backend_tensor_set(tensor, input.data(), 0, sizeof(input));
    std::array<uint32_t, 64> output = {};
    ggml_backend_tensor_get(tensor, output.data(), 0, sizeof(output));
    REQUIRE(output == input);
    ggml_backend_tensor_copy(tensor, copy);
    output.fill(0);
    ggml_backend_tensor_get(copy, output.data(), 0, sizeof(output));
    REQUIRE(output == input);

    input[0] = 0x12345678;
    ggml_backend_tensor_set_async(backend, tensor, input.data(), 0, sizeof(input));
    ggml_backend_synchronize(backend);
    output.fill(0);
    ggml_backend_tensor_get_async(backend, tensor, output.data(), 0, sizeof(output));
    ggml_backend_synchronize(backend);
    REQUIRE(output == input);
    REQUIRE(hrx->device->synchronous_upload_fallbacks.load(std::memory_order_relaxed) == 1);
    REQUIRE(hrx->device->synchronous_download_fallbacks.load(std::memory_order_relaxed) == 1);

    ggml_backend_tensor_memset(tensor, 0x5a, 16, 32);
    ggml_backend_tensor_get(tensor, output.data(), 0, sizeof(output));
    const uint8_t * bytes = reinterpret_cast<const uint8_t *>(output.data());
    for (size_t i = 16; i < 48; ++i) {
        REQUIRE(bytes[i] == 0x5a);
    }

    ggml_backend_buffer_clear(buffer, 0);
    ggml_backend_tensor_get(tensor, output.data(), 0, sizeof(output));
    for (uint32_t value : output) {
        REQUIRE(value == 0);
    }

    ggml_backend_buffer_free(buffer);
    ggml_backend_buffer_free(copy_buffer);
    ggml_free(context);
    ggml_backend_synchronize(backend);
    REQUIRE(hrx->device->device != nullptr);
}

static void run_host_buffer_checks(ggml_backend_t backend) {
    ggml_backend_hrx_context * context = backend_context(backend);
    ggml_backend_buffer_type_t buft    = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(backend));
    REQUIRE(buft != nullptr);
    REQUIRE(ggml_backend_buft_is_host(buft));

    const bool original_direct_host_bindings = context->device->use_direct_host_bindings;
    context->device->use_direct_host_bindings = false;
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, 4096);
    context->device->use_direct_host_bindings = original_direct_host_bindings;
    REQUIRE(buffer != nullptr);
    REQUIRE(ggml_backend_buffer_is_host(buffer));
    auto * buffer_context = ggml_backend_hrx_buffer_context_from_buffer(buffer);
    REQUIRE(buffer_context != nullptr);
    REQUIRE(buffer_context->buffer != nullptr);
    REQUIRE(buffer_context->base == ggml_backend_buffer_get_base(buffer));
    REQUIRE(!buffer_context->direct_host_binding);

    const uint32_t pattern = 0x12345678;
    require_hrx_status(
        hrx_stream_fill_buffer(context->stream, buffer_context->buffer, 0, 4096, &pattern, sizeof(pattern)));
    require_hrx_status(hrx_stream_synchronize(context->stream));
    const auto * words = static_cast<const uint32_t *>(ggml_backend_buffer_get_base(buffer));
    for (size_t i = 0; i < 4096 / sizeof(uint32_t); ++i) {
        REQUIRE(words[i] == pattern);
    }

    ggml_init_params params = {};
    params.mem_size         = 4096;
    params.no_alloc         = true;
    ggml_context * ggml     = ggml_init(params);
    REQUIRE(ggml != nullptr);
    ggml_tensor * host_tensor = ggml_new_tensor_1d(ggml, GGML_TYPE_I32, 64);
    host_tensor->buffer       = buffer;
    host_tensor->data         = ggml_backend_buffer_get_base(buffer);
    REQUIRE(ggml_backend_buffer_init_tensor(buffer, host_tensor) == GGML_STATUS_SUCCESS);
    ggml::hrx::ValueBufferBinding staged_binding;
    REQUIRE(ggml_backend_hrx_resolve_value_buffer(host_tensor, staged_binding));
    REQUIRE(staged_binding.buffer == nullptr);
    REQUIRE(staged_binding.host_data == ggml_backend_buffer_get_base(buffer));
    REQUIRE(staged_binding.offset == 0);
    REQUIRE(staged_binding.length == ggml_nbytes(host_tensor));

    context->device->use_direct_host_bindings = true;
    ggml_backend_buffer_t direct_buffer = ggml_backend_buft_alloc_buffer(buft, 4096);
    context->device->use_direct_host_bindings = original_direct_host_bindings;
    REQUIRE(direct_buffer != nullptr);
    auto * direct_buffer_context = ggml_backend_hrx_buffer_context_from_buffer(direct_buffer);
    REQUIRE(direct_buffer_context->direct_host_binding);
    ggml_tensor * direct_tensor = ggml_new_tensor_1d(ggml, GGML_TYPE_I32, 64);
    direct_tensor->buffer       = direct_buffer;
    direct_tensor->data         = ggml_backend_buffer_get_base(direct_buffer);
    REQUIRE(ggml_backend_buffer_init_tensor(direct_buffer, direct_tensor) == GGML_STATUS_SUCCESS);
    ggml::hrx::ValueBufferBinding direct_binding;
    REQUIRE(ggml_backend_hrx_resolve_value_buffer(direct_tensor, direct_binding));
    REQUIRE(direct_binding.buffer == direct_buffer_context->buffer);
    REQUIRE(direct_binding.host_data == nullptr);
    REQUIRE(direct_binding.offset == 0);
    REQUIRE(direct_binding.length == ggml_nbytes(direct_tensor));
    ggml_backend_hrx_imported_region retained_region = {};
    REQUIRE(ggml_backend_hrx_get_retained_region(host_tensor, retained_region));
    REQUIRE(!ggml_backend_hrx_get_imported_region(host_tensor, retained_region));
    const uint64_t before_host_set = retained_region.generation;
    const uint32_t host_replacement = 0xabcdef01;
    ggml_backend_tensor_set(host_tensor, &host_replacement, 0, sizeof(host_replacement));
    REQUIRE(ggml_backend_hrx_get_retained_region(host_tensor, retained_region));
    REQUIRE(retained_region.generation != before_host_set);
    REQUIRE(ggml_backend_hrx_get_retained_region(direct_tensor, retained_region));
    const uint64_t before_direct_set = retained_region.generation;
    ggml_backend_tensor_set_async(backend, direct_tensor, &host_replacement, 0, sizeof(host_replacement));
    ggml_backend_synchronize(backend);
    REQUIRE(ggml_backend_hrx_get_retained_region(direct_tensor, retained_region));
    REQUIRE(retained_region.generation != before_direct_set);
    const uint64_t before_host_clear = retained_region.generation;
    ggml_backend_buffer_clear(direct_buffer, 0);
    REQUIRE(ggml_backend_hrx_get_retained_region(direct_tensor, retained_region));
    REQUIRE(retained_region.generation != before_host_clear);

    ggml_tensor *         tensor = ggml_new_tensor_1d(ggml, GGML_TYPE_I32, 64);
    ggml_backend_buffer_t local  = ggml_backend_alloc_buffer(backend, 4096);
    REQUIRE(local != nullptr);
    tensor->buffer = local;
    tensor->data   = ggml_backend_buffer_get_base(local);
    REQUIRE(ggml_backend_buffer_init_tensor(local, tensor) == GGML_STATUS_SUCCESS);

    const uint64_t upload_fallbacks   = context->device->synchronous_upload_fallbacks.load(std::memory_order_relaxed);
    const uint64_t download_fallbacks = context->device->synchronous_download_fallbacks.load(std::memory_order_relaxed);
    auto * host_words = static_cast<uint32_t *>(ggml_backend_buffer_get_base(buffer));
    for (size_t i = 0; i < 64; ++i) {
        host_words[i] = static_cast<uint32_t>(i * 13 + 7);
    }
    ggml_backend_tensor_set_async(backend, tensor, host_words, 0, 64 * sizeof(uint32_t));
    ggml_backend_synchronize(backend);
    std::memset(host_words, 0, 64 * sizeof(uint32_t));
    ggml_backend_tensor_get_async(backend, tensor, host_words, 0, 64 * sizeof(uint32_t));
    ggml_backend_synchronize(backend);
    for (size_t i = 0; i < 64; ++i) {
        REQUIRE(host_words[i] == static_cast<uint32_t>(i * 13 + 7));
    }
    REQUIRE(context->device->synchronous_upload_fallbacks.load(std::memory_order_relaxed) == upload_fallbacks);
    REQUIRE(context->device->synchronous_download_fallbacks.load(std::memory_order_relaxed) == download_fallbacks);

    REQUIRE(ggml_backend_hrx_get_retained_region(direct_tensor, retained_region));
    const uint64_t before_host_copy = retained_region.generation;
    ggml_backend_tensor_copy_async(backend, backend, tensor, direct_tensor);
    ggml_backend_synchronize(backend);
    REQUIRE(ggml_backend_hrx_get_retained_region(direct_tensor, retained_region));
    REQUIRE(retained_region.generation != before_host_copy);
    const uint64_t before_host_memset = retained_region.generation;
    ggml_backend_tensor_memset(direct_tensor, 0x5a, 0, sizeof(uint32_t));
    REQUIRE(ggml_backend_hrx_get_retained_region(direct_tensor, retained_region));
    REQUIRE(retained_region.generation != before_host_memset);

    ggml_backend_buffer_free(local);
    ggml_backend_buffer_free(direct_buffer);
    ggml_backend_buffer_free(buffer);
    ggml_free(ggml);
}

static void run_imported_buffer_checks(ggml_backend_t backend) {
    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    ggml_backend_dev_props props = {};
    ggml_backend_dev_get_props(device, &props);
    REQUIRE(props.caps.buffer_from_host_ptr);

    std::array<uint32_t, 128> source = {};
    for (size_t i = 0; i < source.size(); ++i) {
        source[i] = static_cast<uint32_t>(i + 1);
    }
    ggml_backend_buffer_t imported = ggml_backend_dev_buffer_from_host_ptr(device, source.data(), sizeof(source),
                                                                            sizeof(source) / 2);
    REQUIRE(imported != nullptr);
    REQUIRE(ggml_backend_buffer_get_base(imported) == source.data());
    REQUIRE(!ggml_backend_buffer_is_host(imported));
    auto * context = ggml_backend_hrx_buffer_context_from_buffer(imported);
    REQUIRE(context->imported);
    REQUIRE(context->buffer == nullptr);

    ggml_init_params params = {};
    params.mem_size = 4096;
    params.no_alloc = true;
    ggml_context * ggml = ggml_init(params);
    REQUIRE(ggml != nullptr);
    ggml_tensor * first = ggml_new_tensor_1d(ggml, GGML_TYPE_I32, 32);
    ggml_tensor * second = ggml_new_tensor_1d(ggml, GGML_TYPE_I32, 32);
    REQUIRE(ggml_backend_tensor_alloc(imported, first, source.data()) == GGML_STATUS_SUCCESS);
    REQUIRE(ggml_backend_tensor_alloc(imported, second, source.data() + 32) == GGML_STATUS_SUCCESS);
    ggml_backend_buffer_set_usage(imported, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    ggml_backend_hrx_imported_region first_region = {};
    ggml_backend_hrx_imported_region second_region = {};
    REQUIRE(ggml_backend_hrx_get_imported_region(first, first_region));
    REQUIRE(ggml_backend_hrx_get_imported_region(second, second_region));
    REQUIRE(first_region.identity == second_region.identity);
    REQUIRE(first_region.offset == 0);
    REQUIRE(second_region.offset == 32 * sizeof(uint32_t));
    REQUIRE(first_region.type == GGML_TYPE_I32);
    REQUIRE(first_region.ne[0] == 32);
    REQUIRE(first_region.nb[0] == sizeof(uint32_t));

    ggml_tensor * view = ggml_view_1d(ggml, second, 8, 4 * sizeof(uint32_t));
    REQUIRE(ggml_backend_view_init(view) == GGML_STATUS_SUCCESS);
    ggml_backend_hrx_imported_region view_region = {};
    REQUIRE(ggml_backend_hrx_get_imported_region(view, view_region));
    REQUIRE(view_region.identity == second_region.identity);
    REQUIRE(view_region.offset == second_region.offset + 4 * sizeof(uint32_t));
    REQUIRE(view_region.length == 8 * sizeof(uint32_t));
    REQUIRE(view_region.ne[0] == 8);

    ggml::hrx::ValueBufferBinding binding;
    REQUIRE(ggml_backend_hrx_resolve_value_buffer(second, binding));
    REQUIRE(binding.buffer == nullptr);
    REQUIRE(binding.host_data == source.data());
    REQUIRE(binding.offset == second_region.offset);
    REQUIRE(binding.weight);

    std::array<uint32_t, 32> readback = {};
    ggml_backend_tensor_get(second, readback.data(), 0, sizeof(readback));
    REQUIRE(readback[0] == source[32]);
    const uint64_t old_generation = second_region.generation;
    const uint32_t replacement = 999;
    ggml_backend_tensor_set(second, &replacement, 0, sizeof(replacement));
    REQUIRE(source[32] == replacement);
    REQUIRE(ggml_backend_hrx_get_imported_region(second, second_region));
    REQUIRE(second_region.generation != old_generation);

    const uint32_t async_replacement = 1234;
    ggml_backend_tensor_set_async(backend, second, &async_replacement, 0, sizeof(async_replacement));
    ggml_backend_synchronize(backend);
    ggml_backend_tensor_get_async(backend, second, readback.data(), 0, sizeof(readback));
    ggml_backend_synchronize(backend);
    REQUIRE(readback[0] == async_replacement);
    REQUIRE(ggml_backend_hrx_get_imported_region(second, second_region));
    REQUIRE(second_region.generation != old_generation);

    ggml_backend_buffer_t owned = ggml_backend_alloc_buffer(backend, sizeof(readback));
    REQUIRE(owned != nullptr);
    ggml_tensor * copy = ggml_new_tensor_1d(ggml, GGML_TYPE_I32, 32);
    REQUIRE(ggml_backend_tensor_alloc(owned, copy, ggml_backend_buffer_get_base(owned)) == GGML_STATUS_SUCCESS);
    ggml_backend_tensor_copy_async(backend, backend, second, copy);
    ggml_backend_synchronize(backend);
    ggml_backend_tensor_copy_async(backend, backend, copy, first);
    ggml_backend_synchronize(backend);
    REQUIRE(source[0] == async_replacement);

    ggml_backend_buffer_free(owned);
    ggml_backend_buffer_free(imported);
    REQUIRE(source[0] == async_replacement);
    ggml_free(ggml);

    auto * hrx_device = backend_context(backend)->device;
    const bool previous_direct = hrx_device->use_direct_host_bindings;
    hrx_device->use_direct_host_bindings = true;
    ggml_backend_buffer_t coherent = ggml_backend_buft_alloc_buffer(ggml_backend_dev_host_buffer_type(device), 1024);
    hrx_device->use_direct_host_bindings = previous_direct;
    REQUIRE(coherent != nullptr);
    auto * coherent_context = ggml_backend_hrx_buffer_context_from_buffer(coherent);
    uint8_t * subregion = static_cast<uint8_t *>(ggml_backend_buffer_get_base(coherent)) + 256;
    ggml_backend_buffer_t imported_coherent = ggml_backend_dev_buffer_from_host_ptr(device, subregion, 256, 256);
    REQUIRE(imported_coherent != nullptr);
    REQUIRE(!ggml_backend_buffer_is_host(imported_coherent));
    auto * imported_context = ggml_backend_hrx_buffer_context_from_buffer(imported_coherent);
    REQUIRE(imported_context->imported);
    REQUIRE(imported_context->direct_host_binding);
    REQUIRE(imported_context->buffer == coherent_context->buffer);
    REQUIRE(imported_context->physical_offset == 256);
    ggml_init_params subparams = params;
    subparams.mem_size = 128 * 1024;
    ggml_context * subctx = ggml_init(subparams);
    REQUIRE(subctx != nullptr);
    ggml_tensor * subtensor = ggml_new_tensor_1d(subctx, GGML_TYPE_I32, 32);
    REQUIRE(ggml_backend_tensor_alloc(imported_coherent, subtensor, subregion) == GGML_STATUS_SUCCESS);
    ggml::hrx::ValueBufferBinding direct_binding;
    REQUIRE(ggml_backend_hrx_resolve_value_buffer(subtensor, direct_binding));
    REQUIRE(direct_binding.buffer == coherent_context->buffer);
    REQUIRE(direct_binding.host_data == nullptr);
    REQUIRE(direct_binding.offset == 256);
    const uint32_t queued_value = 0x456789ab;
    require_hrx_status(hrx_stream_fill_buffer(backend_context(backend)->stream, coherent_context->buffer,
                                              256, 32 * sizeof(uint32_t), &queued_value, sizeof(queued_value)));
    std::array<uint32_t, 32> ordered_readback = {};
    ggml_backend_tensor_get_async(backend, subtensor, ordered_readback.data(), 0, sizeof(ordered_readback));
    for (uint32_t value : ordered_readback) {
        REQUIRE(value == queued_value);
    }
    const uint32_t ordered_write = 0x13579bdf;
    require_hrx_status(hrx_stream_fill_buffer(backend_context(backend)->stream, coherent_context->buffer,
                                              256, 32 * sizeof(uint32_t), &queued_value, sizeof(queued_value)));
    ggml_backend_tensor_set_async(backend, subtensor, &ordered_write, 0, sizeof(ordered_write));
    ggml_backend_synchronize(backend);
    REQUIRE(*reinterpret_cast<uint32_t *>(subregion) == ordered_write);

    ggml_backend_t cpu = ggml_backend_cpu_init();
    REQUIRE(cpu != nullptr);
    ggml_backend_buffer_t cpu_buffer = ggml_backend_alloc_buffer(cpu, sizeof(ordered_readback));
    REQUIRE(cpu_buffer != nullptr);
    ggml_context * cpu_ctx = ggml_init(params);
    REQUIRE(cpu_ctx != nullptr);
    ggml_tensor * cpu_tensor = ggml_new_tensor_1d(cpu_ctx, GGML_TYPE_I32, 32);
    REQUIRE(ggml_backend_tensor_alloc(cpu_buffer, cpu_tensor, ggml_backend_buffer_get_base(cpu_buffer)) == GGML_STATUS_SUCCESS);
    std::array<uint32_t, 32> cpu_values = {};
    cpu_values.fill(0x2468ace0);
    ggml_backend_tensor_set(cpu_tensor, cpu_values.data(), 0, sizeof(cpu_values));
    require_hrx_status(hrx_stream_fill_buffer(backend_context(backend)->stream, coherent_context->buffer,
                                              256, sizeof(cpu_values), &queued_value, sizeof(queued_value)));
    ggml_backend_tensor_copy_async(cpu, backend, cpu_tensor, subtensor);
    ggml_backend_synchronize(backend);
    REQUIRE(*reinterpret_cast<uint32_t *>(subregion) == cpu_values[0]);

    ggml_tensor * coherent_output = ggml_new_tensor_1d(subctx, GGML_TYPE_F32, 32);
    REQUIRE(ggml_backend_tensor_alloc(imported_coherent, coherent_output, subregion) == GGML_STATUS_SUCCESS);
    ggml_tensor * coherent_input = ggml_new_tensor_1d(subctx, GGML_TYPE_F32, 32);
    ggml_backend_buffer_t coherent_input_buffer = ggml_backend_alloc_buffer(backend, 32 * sizeof(float));
    REQUIRE(coherent_input_buffer != nullptr);
    REQUIRE(ggml_backend_tensor_alloc(coherent_input_buffer, coherent_input,
                                      ggml_backend_buffer_get_base(coherent_input_buffer)) == GGML_STATUS_SUCCESS);
    std::array<float, 32> coherent_values = {};
    coherent_values.fill(3.25f);
    ggml_backend_tensor_set(coherent_input, coherent_values.data(), 0, sizeof(coherent_values));
    ggml_tensor * coherent_copy = ggml_cpy(subctx, coherent_input, coherent_output);
    ggml_cgraph * coherent_graph = ggml_new_graph(subctx);
    REQUIRE(coherent_graph != nullptr);
    ggml_build_forward_expand(coherent_graph, coherent_copy);
    coherent_graph->uid = 950005;
    ggml_backend_hrx_imported_region coherent_region = {};
    REQUIRE(ggml_backend_hrx_get_imported_region(coherent_output, coherent_region));
    const uint64_t coherent_generation = coherent_region.generation;
    REQUIRE(ggml_backend_graph_compute(backend, coherent_graph) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);
    REQUIRE(ggml_backend_hrx_get_imported_region(coherent_output, coherent_region));
    REQUIRE(coherent_region.generation != coherent_generation);
    REQUIRE(*reinterpret_cast<float *>(subregion) == coherent_values[0]);
    ggml_backend_buffer_free(coherent_input_buffer);
    ggml_backend_buffer_free(cpu_buffer);
    ggml_free(cpu_ctx);
    ggml_backend_free(cpu);
    ggml_backend_buffer_free(imported_coherent);
    ggml_backend_buffer_free(coherent);
    ggml_free(subctx);

#if defined(__linux__)
    // Imported file mappings can be read-only. Import, tensor initialization and buffer release must only borrow it.
    constexpr size_t page_size = 4096;
    void * protected_page = mmap(nullptr, page_size, PROT_READ | PROT_WRITE,
                                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    REQUIRE(protected_page != MAP_FAILED);
    auto * protected_bytes = static_cast<uint8_t *>(protected_page);
    protected_bytes[0] = 0x5a;
    REQUIRE(mprotect(protected_page, page_size, PROT_READ) == 0);
    ggml_backend_buffer_t read_only_import = ggml_backend_dev_buffer_from_host_ptr(
        device, protected_page, page_size, page_size);
    REQUIRE(read_only_import != nullptr);
    REQUIRE(!ggml_backend_buffer_is_host(read_only_import));
    ggml_context * read_only_ctx = ggml_init(params);
    REQUIRE(read_only_ctx != nullptr);
    ggml_tensor * read_only_tensor = ggml_new_tensor_1d(read_only_ctx, GGML_TYPE_I8, 1);
    REQUIRE(ggml_backend_tensor_alloc(read_only_import, read_only_tensor, protected_page) == GGML_STATUS_SUCCESS);
    uint8_t protected_readback = 0;
    ggml_backend_tensor_get(read_only_tensor, &protected_readback, 0, sizeof(protected_readback));
    REQUIRE(protected_readback == 0x5a);
    ggml_backend_buffer_free(read_only_import);
    REQUIRE(protected_bytes[0] == 0x5a);
    ggml_free(read_only_ctx);
    REQUIRE(munmap(protected_page, page_size) == 0);
#endif
}

static void run_host_transfer_checks(ggml_backend_hrx_context * context) {
    ggml::hrx::HostTransferManager transfers;
    ggml::hrx::HostStagingBuffer   staging;
    REQUIRE(ggml::hrx::allocate_host_staging_buffer(context->device->device, 64, staging).success());

    const std::array<uint8_t, 64> zero = {};
    require_hrx_status(hrx_synchronous_h2d(context->device->device, zero.data(), staging.buffer, 0, zero.size()));

    std::array<uint8_t, 64> host = {};
    for (size_t i = 0; i < host.size(); ++i) {
        host[i] = static_cast<uint8_t>(i + 1);
    }

    REQUIRE(transfers.upload_synchronous(context->stream, host.data(), staging.buffer, 0, 0).success());
    REQUIRE(transfers.upload_synchronous(context->stream, host.data() + 8, staging.buffer, 16, 24).success());
    ggml::hrx::HostTransferStats stats = transfers.stats();
    REQUIRE(stats.uploads == 1);
    REQUIRE(stats.upload_bytes == 24);
    require_hrx_status(hrx_stream_synchronize(context->stream));

    std::array<uint8_t, 64> upload_result = {};
    require_hrx_status(
        hrx_synchronous_d2h(context->device->device, staging.buffer, 0, upload_result.data(), upload_result.size()));
    for (size_t i = 0; i < upload_result.size(); ++i) {
        const uint8_t expected = i >= 16 && i < 40 ? host[i - 8] : 0;
        REQUIRE(upload_result[i] == expected);
    }

    std::array<uint8_t, 64> device_values = {};
    for (size_t i = 0; i < device_values.size(); ++i) {
        device_values[i] = static_cast<uint8_t>(0xa0 + i);
    }
    require_hrx_status(
        hrx_synchronous_h2d(context->device->device, device_values.data(), staging.buffer, 0, device_values.size()));

    std::array<uint8_t, 48> download_result = {};
    REQUIRE(
        transfers.download_synchronous(context->stream, staging.buffer, 12, download_result.data() + 4, 20).success());
    stats = transfers.stats();
    REQUIRE(stats.downloads == 1);
    REQUIRE(stats.download_bytes == 20);
    require_hrx_status(hrx_stream_synchronize(context->stream));
    for (size_t i = 0; i < download_result.size(); ++i) {
        const uint8_t expected = i >= 4 && i < 24 ? device_values[i + 8] : 0;
        REQUIRE(download_result[i] == expected);
    }

    REQUIRE(!transfers.upload_synchronous(nullptr, host.data(), staging.buffer, 0, 4).success());
    REQUIRE(!transfers.upload_synchronous(context->stream, nullptr, staging.buffer, 0, 4).success());
    REQUIRE(!transfers.upload_synchronous(context->stream, host.data(), nullptr, 0, 4).success());
    REQUIRE(!transfers.download_synchronous(nullptr, staging.buffer, 0, download_result.data(), 4).success());
    REQUIRE(!transfers.download_synchronous(context->stream, nullptr, 0, download_result.data(), 4).success());
    REQUIRE(!transfers.download_synchronous(context->stream, staging.buffer, 0, nullptr, 4).success());

    transfers.clear();
    stats = transfers.stats();
    REQUIRE(stats.uploads == 0);
    REQUIRE(stats.downloads == 0);
    REQUIRE(stats.upload_bytes == 0);
    REQUIRE(stats.download_bytes == 0);
}

static void run_host_staging_checks(ggml_backend_hrx_context * context) {
    ggml::hrx::HostStagingBuffer staging;
    REQUIRE(ggml::hrx::allocate_host_staging_buffer(context->device->device, 32, staging).success());
    REQUIRE(staging.buffer != nullptr);
    REQUIRE(staging.length == 32);

    hrx_buffer_t                 original = staging.buffer;
    ggml::hrx::HostStagingBuffer moved(std::move(staging));
    REQUIRE(moved.buffer == original);
    REQUIRE(moved.length == 32);
    REQUIRE(staging.buffer == nullptr);
    REQUIRE(staging.length == 0);

    ggml::hrx::HostStagingBuffer assigned;
    assigned = std::move(moved);
    REQUIRE(assigned.buffer == original);
    REQUIRE(assigned.length == 32);
    REQUIRE(moved.buffer == nullptr);
    REQUIRE(moved.length == 0);

    assigned.clear();
    REQUIRE(assigned.buffer == nullptr);
    REQUIRE(assigned.length == 0);
    assigned.clear();
    REQUIRE(assigned.buffer == nullptr);
}

static void run_host_weight_cache_checks(ggml_backend_hrx_context * context) {
    ggml::hrx::HostTransferManager transfers;
    ggml::hrx::HostWeightCache     weights;

    std::array<uint8_t, 128> host = {};
    for (size_t i = 0; i < host.size(); ++i) {
        host[i] = static_cast<uint8_t>(i);
    }

    ggml::hrx::HostWeightSource source;
    source.host_data  = host.data();
    source.identity   = 0x1234;
    source.generation = 1;
    source.capacity   = host.size();
    source.offset     = 16;
    source.length     = 32;
    source.layout     = "ggml-native";

    ggml::hrx::HostWeightAcquireResult first =
        weights.acquire(context->device->device, context->stream, transfers, source);
    REQUIRE(first.valid());

    std::array<uint8_t, 32> first_bytes = {};
    require_hrx_status(
        hrx_synchronous_d2h(context->device->device, first.lease.buffer(), 0, first_bytes.data(), first_bytes.size()));
    for (size_t i = 0; i < first_bytes.size(); ++i) {
        REQUIRE(first_bytes[i] == host[source.offset + i]);
    }

    ggml::hrx::HostWeightAcquireResult second =
        weights.acquire(context->device->device, context->stream, transfers, source);
    REQUIRE(second.valid());
    REQUIRE(second.lease.buffer() == first.lease.buffer());

    source.offset = 32;
    ggml::hrx::HostWeightAcquireResult slice =
        weights.acquire(context->device->device, context->stream, transfers, source);
    REQUIRE(slice.valid());
    REQUIRE(slice.lease.buffer() != first.lease.buffer());

    source.offset     = 16;
    source.generation = 2;
    ggml::hrx::HostWeightAcquireResult next_generation =
        weights.acquire(context->device->device, context->stream, transfers, source);
    REQUIRE(next_generation.valid());
    REQUIRE(next_generation.lease.buffer() != first.lease.buffer());

    source.generation = 1;
    source.layout     = "alternate-layout";
    ggml::hrx::HostWeightAcquireResult conflict =
        weights.acquire(context->device->device, context->stream, transfers, source);
    REQUIRE(!conflict.valid());

    ggml::hrx::HostWeightCacheStats weight_stats = weights.stats();
    REQUIRE(weight_stats.hits == 1);
    REQUIRE(weight_stats.misses == 3);
    REQUIRE(weight_stats.allocation_count == 3);
    REQUIRE(weight_stats.resident_bytes == 96);

    const ggml::hrx::HostTransferStats transfer_stats = transfers.stats();
    REQUIRE(transfer_stats.uploads == 3);
    REQUIRE(transfer_stats.upload_bytes == 96);

    ggml::hrx::RecordedCommandGraph recorded;
    hrx_buffer_retain(first.lease.buffer());
    recorded.retained_buffers.push_back(first.lease.buffer());
    recorded.resident_host_weights.push_back(first.lease);
    weights.clear();
    weight_stats = weights.stats();
    REQUIRE(weight_stats.hits == 1);
    REQUIRE(weight_stats.misses == 3);
    REQUIRE(weight_stats.allocation_count == 3);
    REQUIRE(weight_stats.resident_bytes == 96);
    first.lease = {};
    second.lease = {};
    slice.lease = {};
    next_generation.lease = {};
    weight_stats = weights.stats();
    REQUIRE(weight_stats.allocation_count == 1);
    REQUIRE(weight_stats.resident_bytes == 32);
    recorded = {};
    weight_stats = weights.stats();
    REQUIRE(weight_stats.allocation_count == 0);
    REQUIRE(weight_stats.resident_bytes == 0);

    source.layout = ggml::hrx::kNativeWeightLayout;
    source.generation = 3;
    source.offset = 16;
    source.source_type = GGML_TYPE_F32;
    source.source_ne = { 8, 1, 1, 1 };
    source.source_nb = { 4, 32, 32, 32 };
    auto region = weights.acquire(context->device->device, context->stream, transfers, source);
    auto alias = weights.acquire(context->device->device, context->stream, transfers, source);
    REQUIRE(region.valid() && alias.valid());
    REQUIRE(region.lease.buffer() == alias.lease.buffer());
    source.source_type = GGML_TYPE_I32;
    auto other_type = weights.acquire(context->device->device, context->stream, transfers, source);
    REQUIRE(other_type.valid());
    REQUIRE(other_type.lease.buffer() != region.lease.buffer());
    source.source_type = GGML_TYPE_F32;
    source.source_ne = { 4, 2, 1, 1 };
    auto other_shape = weights.acquire(context->device->device, context->stream, transfers, source);
    REQUIRE(other_shape.valid());
    REQUIRE(other_shape.lease.buffer() != region.lease.buffer());
    source.source_ne = { 8, 1, 1, 1 };
    source.source_nb = { 4, 64, 64, 64 };
    auto other_stride = weights.acquire(context->device->device, context->stream, transfers, source);
    REQUIRE(other_stride.valid());
    REQUIRE(other_stride.lease.buffer() != region.lease.buffer());
    source.source_nb = { 4, 32, 32, 32 };
    source.offset = 32;
    auto other_slice = weights.acquire(context->device->device, context->stream, transfers, source);
    REQUIRE(other_slice.valid());
    REQUIRE(other_slice.lease.buffer() != region.lease.buffer());
    weight_stats = weights.stats();
    REQUIRE(weight_stats.hits == 2);
    REQUIRE(weight_stats.misses == 8);
    REQUIRE(weight_stats.allocation_count == 5);
    REQUIRE(weight_stats.peak_resident_bytes == 160);
}

static void run_q6_pack_scratch_checks(ggml_backend_hrx_context * context) {
    constexpr size_t weight_bytes = 64 * 210;
    std::vector<uint8_t> host(weight_bytes, 0x5a);
    ggml::hrx::HostTransferManager transfers;
    ggml::hrx::HostWeightCache weights;
    ggml::hrx::HostWeightSource source;
    source.host_data = host.data();
    source.identity = 0x5678;
    source.generation = 1;
    source.capacity = weight_bytes;
    source.length = weight_bytes;
    source.materialized_length = weight_bytes;
    source.layout = ggml::hrx::kQ6KPackedK256Row64ScaleRowLayout;
    source.source_type = GGML_TYPE_Q6_K;
    source.input_size = 256;
    source.output_size = 64;

    auto first = weights.acquire(context->device->device, context->stream, transfers, source);
    REQUIRE(first.valid());
    std::vector<uint8_t> downloaded(weight_bytes);
    require_hrx_status(hrx_synchronous_d2h(context->device->device, first.lease.buffer(), 0,
                                           downloaded.data(), downloaded.size()));
    for (uint8_t byte : downloaded) {
        REQUIRE(byte == 0x5a);
    }

    first.lease = {};
    weights.evict_except({}, {});
    REQUIRE(weights.stats().allocation_count == 0);
    std::fill(host.begin(), host.end(), 0xa5);
    source.generation = 2;
    auto second = weights.acquire(context->device->device, context->stream, transfers, source);
    REQUIRE(second.valid());
    require_hrx_status(hrx_synchronous_d2h(context->device->device, second.lease.buffer(), 0,
                                           downloaded.data(), downloaded.size()));
    for (uint8_t byte : downloaded) {
        REQUIRE(byte == 0xa5);
    }
    REQUIRE(weights.stats().peak_resident_bytes == weight_bytes);
}

template <typename F>
static void run_with_hrx_backend(F && f) {
    ggml_backend_t backend = ggml_backend_hrx_init(0);
    REQUIRE(backend != nullptr);
    ggml_backend_hrx_context * context = backend_context(backend);
    f(backend, context);
    ggml_backend_free(backend);
}

static void register_hrx_buffer_cases(test_runner::Suite & suite) {
    suite.device_case("backend_buffer", [] {
        run_with_hrx_backend([](ggml_backend_t backend, ggml_backend_hrx_context *) {
            run_backend_buffer_checks(backend);
        });
    });
    suite.device_case("host_buffer", [] {
        run_with_hrx_backend([](ggml_backend_t backend, ggml_backend_hrx_context *) {
            run_host_buffer_checks(backend);
        });
    });
    suite.device_case("imported_buffer", [] {
        run_with_hrx_backend([](ggml_backend_t backend, ggml_backend_hrx_context *) {
            run_imported_buffer_checks(backend);
        });
    });
    suite.device_case("host_transfer", [] {
        run_with_hrx_backend([](ggml_backend_t, ggml_backend_hrx_context * context) {
            run_host_transfer_checks(context);
        });
    });
    suite.device_case("host_staging", [] {
        run_with_hrx_backend([](ggml_backend_t, ggml_backend_hrx_context * context) {
            run_host_staging_checks(context);
        });
    });
    suite.device_case("host_weight_cache", [] {
        run_with_hrx_backend([](ggml_backend_t, ggml_backend_hrx_context * context) {
            run_host_weight_cache_checks(context);
        });
    });
    suite.device_case("q6_pack_scratch", [] {
        run_with_hrx_backend([](ggml_backend_t, ggml_backend_hrx_context * context) {
            run_q6_pack_scratch_checks(context);
        });
    });
}

int main(int argc, char ** argv) {
    test_runner::Suite suite(test_runner::Config::with_prefix(
        "HRX buffer test", "hrx-buffer", "GGML_HRX_BUFFER_TEST", 1));
    register_hrx_buffer_cases(suite);

    const bool has_device = ggml_backend_hrx_get_device_count() != 0;
    return suite.run(argc, argv, has_device);
}
