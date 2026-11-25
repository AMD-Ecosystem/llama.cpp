#pragma once

#include <cstdlib>

#define GGML_CUDA_TRAFFIC_OPT_ENABLED(name, fallback) \
    ([] { static const bool enabled = [] { const char * v = std::getenv(#name); return v ? std::atoi(v) != 0 : (fallback); }(); return enabled; }())
