#pragma once

#include "runtime/command-program-executor.h"
#include "runtime/host-memory.h"
#include "status.h"

#include <memory>
#include <unordered_map>
#include <vector>

typedef struct hrx_event_s * hrx_event_t;
typedef struct hrx_semaphore_s * hrx_semaphore_t;

namespace ggml::hrx {

// Keeps submitted bindings alive until their stream timeline point or failure-path event completes.
// The current demand is retained separately so completed work can stay warm for the next call.
class WeightSubmissionTracker {
  public:
    ~WeightSubmissionTracker();

    WeightSubmissionTracker(const WeightSubmissionTracker &)             = delete;
    WeightSubmissionTracker & operator=(const WeightSubmissionTracker &) = delete;
    WeightSubmissionTracker() = default;

    Status poll_completed(bool * completed = nullptr);
    Status record(hrx_device_t device, hrx_stream_t stream,
                  std::shared_ptr<CachedCommandProgramResources> resources,
                  const std::vector<HostWeightSource> & demand, bool admitted, bool demand_changed);
    void mark_stream_synchronized();
    bool has_obsolete_pending() const;
    void fail_completion_record_for_test() { fail_next_record_ = true; }

    const std::vector<HostWeightSource> & current_demand() const { return current_demand_; }
    bool demand_matches(const std::vector<HostWeightSource> & demand) const;
    std::vector<HostWeightSource> merged_demand_for(const std::vector<HostWeightSource> & demand) const;
    void release_current_leases() { current_leases_.clear(); }
    const std::vector<HostWeightLease> & current_leases() const { return current_leases_; }
    std::vector<HostWeightLease> outstanding_leases() const;

  private:
    struct RegionKey {
        uint64_t identity = 0;
        size_t offset = 0;
        size_t length = 0;
        ggml_type type = GGML_TYPE_COUNT;
        std::array<int64_t, GGML_MAX_DIMS> ne = {};
        std::array<size_t, GGML_MAX_DIMS> nb = {};

        bool operator==(const RegionKey & other) const;
    };
    struct RegionKeyHash {
        size_t operator()(const RegionKey & key) const;
    };
    using RegionDemands = std::unordered_map<RegionKey, std::vector<HostWeightSource>, RegionKeyHash>;
    static RegionKey region_of(const HostWeightSource & source);
    static RegionDemands group_demands(const std::vector<HostWeightSource> & demand);
    void publish_demand(const std::vector<HostWeightSource> & demand);

    struct Pending {
        hrx_event_t event = nullptr;
        hrx_semaphore_t semaphore = nullptr;
        uint64_t value = 0;
        std::shared_ptr<CachedCommandProgramResources> resources;
        bool may_hold_obsolete_demand = false;
    };

    static void release_completion(Pending & pending);

    std::vector<HostWeightSource> current_demand_;
    RegionDemands retained_regions_;
    std::vector<HostWeightLease> current_leases_;
    std::vector<Pending> pending_;
    bool fail_next_record_ = false;
};

}  // namespace ggml::hrx
