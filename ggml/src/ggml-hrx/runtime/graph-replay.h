#pragma once

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace ggml::hrx {

enum class HrxGraphReplayEvent {
    Disabled,
    Ineligible,
    MissBuild,
    Hit,
    RebuildTransient,
    BuildFailed,
    LaunchFailed,
};

inline const char * hrx_graph_replay_event_name(HrxGraphReplayEvent event) {
    switch (event) {
        case HrxGraphReplayEvent::Disabled:
            return "disabled";
        case HrxGraphReplayEvent::Ineligible:
            return "ineligible";
        case HrxGraphReplayEvent::MissBuild:
            return "miss_build";
        case HrxGraphReplayEvent::Hit:
            return "hit";
        case HrxGraphReplayEvent::RebuildTransient:
            return "rebuild_transient";
        case HrxGraphReplayEvent::BuildFailed:
            return "build_failed";
        case HrxGraphReplayEvent::LaunchFailed:
            return "launch_failed";
    }
    return "unknown";
}

inline bool graph_replay_disabled_by_environment() {
    const char * value = std::getenv("GGML_HRX_DISABLE_GRAPH_REPLAY");
    const bool set = value != nullptr;
    const bool nonempty = set && value[0] != '\0';
    const bool enabled = nonempty && std::strcmp(value, "0") != 0;
    return enabled;
}

inline uint64_t hrx_graph_replay_now_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

}  // namespace ggml::hrx
