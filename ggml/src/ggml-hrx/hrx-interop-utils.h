#pragma once

#include "hrx_runtime.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace ggml::hrx {

// Success has no payload; failure carries the diagnostic produced by HRX or
// by the caller. Keeping this distinct from an empty string makes status tests
// explicit at API boundaries.
using ErrorResult = std::optional<std::string>;

inline ErrorResult take_status(hrx_status_t status) {
    if (hrx_status_is_ok(status)) {
        return std::nullopt;
    }
    char *       message       = nullptr;
    size_t       length        = 0;
    hrx_status_t format_status = hrx_status_to_string(status, &message, &length);
    if (!hrx_status_is_ok(format_status)) {
        hrx_status_ignore(format_status);
    }
    std::string result = message != nullptr ? std::string(message, length) : std::string("unknown HRX error");
    hrx_status_free_message(message);
    hrx_status_ignore(status);
    return result;
}

// Streams and direct transfers must name a provisioned queue with the roles they use.
inline ErrorResult find_provisioned_queue(hrx_device_t device, hrx_queue_role_flags_t roles, hrx_queue_t & queue) {
    queue = nullptr;
    const size_t family_count = hrx_device_queue_family_count(device);
    for (size_t i = 0; i < family_count; ++i) {
        hrx_queue_family_info_t family = {};
        if (ErrorResult error = take_status(hrx_device_query_queue_family(device, static_cast<uint32_t>(i), &family))) {
            return error;
        }
        if ((family.roles & roles) == roles && family.provisioned_queue_count > 0) {
            queue = hrx_device_get_queue(device, static_cast<uint32_t>(i), 0);
            if (queue != nullptr) {
                return std::nullopt;
            }
        }
    }
    return std::string("no provisioned HRX queue with required roles");
}

inline ErrorResult find_transfer_queue(hrx_device_t device, hrx_queue_t & queue) {
    return find_provisioned_queue(device, HRX_QUEUE_ROLE_TRANSFER, queue);
}

inline ErrorResult find_stream_queue(hrx_device_t device, hrx_queue_t & queue) {
    return find_provisioned_queue(device, HRX_QUEUE_ROLE_DISPATCH | HRX_QUEUE_ROLE_TRANSFER, queue);
}

}  // namespace ggml::hrx
