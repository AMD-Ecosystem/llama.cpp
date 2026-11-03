#pragma once

#include "dispatch_registration/dispatch-registry.h"

namespace ggml::hrx {

void register_pool_1d_dispatch(DispatchRegistryBuilder & registry);

}  // namespace ggml::hrx
