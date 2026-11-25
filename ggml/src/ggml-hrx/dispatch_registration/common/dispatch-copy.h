#pragma once

#include "dispatch_registration/dispatch-registry.h"

namespace ggml::hrx {

bool make_copy_f32_dispatch(const Value & source, const Value & output, Dispatch & dispatch);

void register_copy_dispatch(DispatchRegistryBuilder & registry);

}  // namespace ggml::hrx
