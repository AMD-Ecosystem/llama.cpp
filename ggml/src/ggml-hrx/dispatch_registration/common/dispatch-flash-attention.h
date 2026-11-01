#pragma once

#include "../dispatch-registry.h"

namespace ggml::hrx {

class Graph;
struct GraphNode;

bool flash_attention_accepts_maskless_f16_handoff(const Graph & graph, const GraphNode & node);

void register_flash_attention_dispatches(DispatchRegistryBuilder & registry);

}  // namespace ggml::hrx
