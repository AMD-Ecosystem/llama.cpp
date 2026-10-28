#define GGML_COMMON_IMPL_CPP
#include "iq-codebook-resource.h"

#include "ggml-common.h"

#include <cstddef>
#include <cstdlib>

namespace ggml::hrx {

namespace {

const std::vector<uint8_t> & iq1_grid_bytes() {
    static_assert(sizeof(iq1s_grid) == 16384);
    static const std::vector<uint8_t> data(reinterpret_cast<const uint8_t *>(iq1s_grid),
                                           reinterpret_cast<const uint8_t *>(iq1s_grid) + sizeof(iq1s_grid));
    return data;
}

const std::vector<uint8_t> & iq2_xxs_grid_bytes() {
    static_assert(sizeof(iq2xxs_grid) == 2048);
    static const std::vector<uint8_t> data(reinterpret_cast<const uint8_t *>(iq2xxs_grid),
                                           reinterpret_cast<const uint8_t *>(iq2xxs_grid) + sizeof(iq2xxs_grid));
    return data;
}

const std::vector<uint8_t> & iq2_xs_grid_bytes() {
    static_assert(sizeof(iq2xs_grid) == 4096);
    static const std::vector<uint8_t> data(reinterpret_cast<const uint8_t *>(iq2xs_grid),
                                           reinterpret_cast<const uint8_t *>(iq2xs_grid) + sizeof(iq2xs_grid));
    return data;
}

const std::vector<uint8_t> & iq3_xxs_grid_bytes() {
    static_assert(sizeof(iq3xxs_grid) == 1024);
    static const std::vector<uint8_t> data(reinterpret_cast<const uint8_t *>(iq3xxs_grid),
                                           reinterpret_cast<const uint8_t *>(iq3xxs_grid) + sizeof(iq3xxs_grid));
    return data;
}

const std::vector<uint8_t> & iq2_s_grid_bytes() {
    static_assert(sizeof(iq2s_grid) == 8192);
    static const std::vector<uint8_t> data(reinterpret_cast<const uint8_t *>(iq2s_grid),
                                           reinterpret_cast<const uint8_t *>(iq2s_grid) + sizeof(iq2s_grid));
    return data;
}

const std::vector<uint8_t> & iq3_s_grid_bytes() {
    static_assert(sizeof(iq3s_grid) == 2048);
    static const std::vector<uint8_t> data(reinterpret_cast<const uint8_t *>(iq3s_grid),
                                           reinterpret_cast<const uint8_t *>(iq3s_grid) + sizeof(iq3s_grid));
    return data;
}

}  // namespace

const IQCodebookResourceSpec & iq_codebook_resource_spec(IQCodebookResource resource) {
    static const IQCodebookResourceSpec iq1    = { "common.iq1.grid.v1", iq1_grid_bytes(), 256 };
    static const IQCodebookResourceSpec iq2_xxs = { "common.iq2_xxs.grid.v1", iq2_xxs_grid_bytes(), 256 };
    static const IQCodebookResourceSpec iq2_xs  = { "common.iq2_xs.grid.v1", iq2_xs_grid_bytes(), 256 };
    static const IQCodebookResourceSpec iq3_xxs = { "common.iq3_xxs.grid.v1", iq3_xxs_grid_bytes(), 256 };
    static const IQCodebookResourceSpec iq2_s = { "common.iq2_s.grid.v1", iq2_s_grid_bytes(), 256 };
    static const IQCodebookResourceSpec iq3_s = { "common.iq3_s.grid.v1", iq3_s_grid_bytes(), 256 };
    switch (resource) {
        case IQCodebookResource::IQ1:
            return iq1;
        case IQCodebookResource::IQ2XXS:
            return iq2_xxs;
        case IQCodebookResource::IQ2XS:
            return iq2_xs;
        case IQCodebookResource::IQ3XXS:
            return iq3_xxs;
        case IQCodebookResource::IQ2S:
            return iq2_s;
        case IQCodebookResource::IQ3S:
            return iq3_s;
    }
    std::abort();
}

}  // namespace ggml::hrx
