#pragma once

#include "dispatch_registration/dispatch-registry.h"

namespace ggml::hrx {

void register_soft_max_dispatch(DispatchRegistryBuilder & registry);

}  // namespace ggml::hrx
