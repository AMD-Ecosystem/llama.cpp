#pragma once

#include "dispatch_registration/dispatch-registry.h"

namespace ggml::hrx {

void register_amax_scale_dispatch(DispatchRegistryBuilder & registry);

}  // namespace ggml::hrx
