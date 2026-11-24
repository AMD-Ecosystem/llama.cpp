#pragma once

#include "../dispatch-registry.h"

namespace ggml::hrx {

void register_lightning_indexer_dispatch(DispatchRegistryBuilder & registry);

}  // namespace ggml::hrx
