#pragma once

#include "../dispatch-registry.h"

namespace ggml::hrx {

bool qwen_accepts_q6k_q8_input(const DispatchMatchContext & context,
                               const GraphNode &            consumer,
                               const Value &                input);

// The endpoint Q8 projection is a single-token path; prefill uses common matmul dispatch.
Dispatch make_qwen_q6k_q8_projection_dispatch(const Value & weight, const Value & output,
                                              const DispatchBinding & input, bool row64_weight_policy);

void register_qwen_matmul_dispatches(DispatchRegistryBuilder & registry);

}  // namespace ggml::hrx
