#pragma once

#include "../dispatch-registry.h"

namespace ggml::hrx {

void register_rmsnorm_dispatches(DispatchRegistryBuilder & registry);

}  // namespace ggml::hrx
