#pragma once

#include "../dispatch-registry.h"

namespace ggml::hrx {

void register_mrope_text_dispatch(DispatchRegistryBuilder & registry);

}  // namespace ggml::hrx
