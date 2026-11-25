#pragma once

#include "../dispatch-registry.h"

#include <string>

namespace ggml::hrx {

std::string float_config_value(float value);

void register_common_dispatches(DispatchRegistryBuilder & registry);

}  // namespace ggml::hrx
