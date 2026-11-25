#pragma once
#include "dispatch_registration/dispatch-registry.h"

namespace ggml::hrx {
void register_reduction_dispatches(DispatchRegistryBuilder & registry);
}
