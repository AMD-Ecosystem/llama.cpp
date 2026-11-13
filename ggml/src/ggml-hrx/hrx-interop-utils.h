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

// Direct transfers must name a queue that can execute them; a device no longer selects one for the caller.
inline ErrorResult find_transfer_queue(hrx_device_t device, hrx_queue_t & queue) {
    queue = nullptr;
    const size_t family_count = hrx_device_queue_family_count(device);
    for (size_t i = 0; i < family_count; ++i) {
        hrx_queue_family_info_t family = {};
        if (ErrorResult error = take_status(hrx_device_query_queue_family(device, static_cast<uint32_t>(i), &family))) {
            return error;
        }
        if ((family.roles & HRX_QUEUE_ROLE_TRANSFER) != 0 && family.provisioned_queue_count > 0) {
            queue = hrx_device_get_queue(device, static_cast<uint32_t>(i), 0);
            if (queue != nullptr) {
                return std::nullopt;
            }
        }
    }
    return std::string("no provisioned HRX transfer queue");
}

}  // namespace ggml::hrx
