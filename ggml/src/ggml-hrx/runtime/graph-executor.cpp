#include "graph-executor.h"

#include "backend-buffer-binding.h"
#include "ggml-impl.h"
#include "hrx-interop-utils.h"
#include "hrx_runtime.h"
#include "runtime/kernel-executable-cache.h"
#include "runtime/prepared-command-program-cache.h"
#include "runtime/transient-arena.h"

#include <algorithm>
#include <optional>
#include <utility>
#include <vector>

namespace ggml::hrx {

WeightSubmissionTracker::~WeightSubmissionTracker() {
    mark_stream_synchronized();
}

bool WeightSubmissionTracker::RegionKey::operator==(const RegionKey & other) const {
    return identity == other.identity && offset == other.offset &&
           length == other.length && type == other.type && ne == other.ne && nb == other.nb;
}

size_t WeightSubmissionTracker::RegionKeyHash::operator()(const RegionKey & key) const {
    uint64_t hash = UINT64_C(1469598103934665603);
    auto mix = [&](uint64_t value) {
        hash ^= value;
        hash *= UINT64_C(1099511628211);
    };
    mix(key.identity);
    mix(key.offset);
    mix(key.length);
    mix(static_cast<uint64_t>(key.type));
    for (size_t dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
        mix(static_cast<uint64_t>(key.ne[dimension]));
        mix(key.nb[dimension]);
    }
    return static_cast<size_t>(hash);
}

WeightSubmissionTracker::RegionKey WeightSubmissionTracker::region_of(const HostWeightSource & source) {
    return { source.identity, source.offset, source.region_length, source.region_type,
             source.source_ne, source.source_nb };
}

static bool same_weight_representation(const HostWeightSource & left, const HostWeightSource & right) {
    return left.generation == right.generation && left.capacity == right.capacity &&
           left.length == right.length &&
           left.materialized_length == right.materialized_length && left.layout == right.layout &&
           left.source_type == right.source_type && left.input_size == right.input_size &&
           left.output_size == right.output_size;
}

WeightSubmissionTracker::RegionDemands WeightSubmissionTracker::group_demands(
    const std::vector<HostWeightSource> & demand) {
    RegionDemands grouped;
    for (const HostWeightSource & source : demand) {
        auto & layouts = grouped[region_of(source)];
        const bool duplicate = std::any_of(layouts.begin(), layouts.end(), [&](const HostWeightSource & existing) {
            return same_weight_representation(existing, source);
        });
        if (!duplicate) {
            layouts.push_back(source);
        }
    }
    return grouped;
}

bool WeightSubmissionTracker::demand_matches(const std::vector<HostWeightSource> & demand) const {
    bool needs_grouped_comparison = false;
    for (const HostWeightSource & source : demand) {
        const auto retained = retained_regions_.find(region_of(source));
        if (retained == retained_regions_.end()) {
            return false;
        }
        const bool singleton_region = retained->second.size() == 1;
        if (!singleton_region) {
            needs_grouped_comparison = true;
            break;
        }
        if (!same_weight_representation(retained->second.front(), source)) {
            return false;
        }
    }
    if (!needs_grouped_comparison) {
        return true;
    }

    const RegionDemands touched = group_demands(demand);
    for (const auto & [region, layouts] : touched) {
        const auto retained = retained_regions_.find(region);
        const bool region_is_retained = retained != retained_regions_.end();
        if (!region_is_retained) {
            return false;
        }
        const bool same_layout_count = layouts.size() == retained->second.size();
        if (!same_layout_count) {
            return false;
        }
        for (const HostWeightSource & source : layouts) {
            const bool present = std::any_of(retained->second.begin(), retained->second.end(),
                [&](const HostWeightSource & existing) {
                    return same_weight_representation(existing, source);
                });
            if (!present) {
                return false;
            }
        }
    }
    return true;
}

std::vector<HostWeightSource> WeightSubmissionTracker::merged_demand_for(
    const std::vector<HostWeightSource> & demand) const {
    RegionDemands merged = retained_regions_;
    for (auto & [region, layouts] : group_demands(demand)) {
        merged[region] = std::move(layouts);
    }
    std::vector<HostWeightSource> result;
    for (const auto & [region, layouts] : merged) {
        result.insert(result.end(), layouts.begin(), layouts.end());
    }
    return result;
}

void WeightSubmissionTracker::publish_demand(const std::vector<HostWeightSource> & demand) {
    for (auto & [region, layouts] : group_demands(demand)) {
        retained_regions_[region] = std::move(layouts);
    }
    current_demand_.clear();
    for (const auto & [region, layouts] : retained_regions_) {
        current_demand_.insert(current_demand_.end(), layouts.begin(), layouts.end());
    }
}

Status WeightSubmissionTracker::poll_completed(bool * completed) {
    Status status;
    if (completed != nullptr) {
        *completed = false;
    }
    for (auto pending = pending_.begin(); pending != pending_.end();) {
        const bool has_event = pending->event != nullptr;
        const bool has_timeline = pending->semaphore != nullptr;
        const bool has_completion = has_event || has_timeline;
        if (!has_completion) {
            ++pending;
            continue;
        }
        bool complete = false;
        if (has_event) {
            if (ErrorResult error = take_status(hrx_event_query(pending->event, &complete))) {
                status.log("query HRX weight submission completion: %s", error->c_str());
                return status;
            }
        } else {
            uint64_t completed_value = 0;
            if (ErrorResult error = take_status(hrx_semaphore_query(pending->semaphore, &completed_value))) {
                status.log("query HRX weight submission timeline: %s", error->c_str());
                return status;
            }
            complete = completed_value >= pending->value;
        }
        if (complete) {
            const bool report_obsolete_completion = completed != nullptr && pending->may_hold_obsolete_demand;
            if (report_obsolete_completion) {
                *completed = true;
            }
            release_completion(*pending);
            pending = pending_.erase(pending);
        } else {
            ++pending;
        }
    }
    return status;
}

void WeightSubmissionTracker::release_completion(Pending & pending) {
    if (pending.event != nullptr) {
        hrx_event_release(pending.event);
        pending.event = nullptr;
    }
    if (pending.semaphore != nullptr) {
        hrx_semaphore_release(pending.semaphore);
        pending.semaphore = nullptr;
    }
}

Status WeightSubmissionTracker::record(hrx_device_t device, hrx_stream_t stream,
                                       std::shared_ptr<CachedCommandProgramResources> resources,
                                       const std::vector<HostWeightSource> & demand, bool admitted,
                                       bool demand_changed) {
    Status status;
    if (resources == nullptr) {
        if (admitted) {
            if (demand_changed) {
                publish_demand(demand);
            }
            current_leases_.clear();
        }
        return status;
    }
    const bool admitted_new_demand = admitted && demand_changed;
    if (admitted_new_demand) {
        for (Pending & pending : pending_) {
            pending.may_hold_obsolete_demand = true;
        }
    }
    hrx_event_t event = nullptr;
    hrx_semaphore_t semaphore = nullptr;
    uint64_t value = 0;
    if (fail_next_record_) {
        fail_next_record_ = false;
        status.log("injected HRX weight completion capture failure");
    } else if (admitted) {
        // Capture after all queued work, including a possible host-download tail.
        hrx_timeline_point_t position = {};
        if (ErrorResult error = take_status(hrx_stream_flush(stream))) {
            status.log("flush HRX weight submission: %s", error->c_str());
        } else if (ErrorResult error = take_status(hrx_stream_get_timeline_position(stream, &position))) {
            status.log("capture HRX weight submission timeline: %s", error->c_str());
        } else {
            semaphore = position.semaphore;
            value = position.value;
            hrx_semaphore_retain(semaphore);
        }
    } else if (ErrorResult error = take_status(hrx_event_create(device, HRX_EVENT_FLAG_DISABLE_TIMING, &event))) {
        status.log("create HRX weight submission completion: %s", error->c_str());
    } else if (ErrorResult error = take_status(hrx_event_record(event, stream))) {
        status.log("record HRX weight submission completion: %s", error->c_str());
    }
    if (!status.success()) {
        if (event != nullptr) {
            hrx_event_release(event);
        }
        if (semaphore != nullptr) {
            hrx_semaphore_release(semaphore);
        }
        // A capture failure must not let queued commands outlive their bindings.
        if (ErrorResult error = take_status(hrx_stream_synchronize(stream))) {
            status.log("synchronize after HRX completion capture failure: %s", error->c_str());
            pending_.push_back({ nullptr, nullptr, 0, std::move(resources), true });
            return status;
        }
        mark_stream_synchronized();
        return status;
    }
    pending_.push_back({ event, semaphore, value, std::move(resources), !admitted });
    if (admitted) {
        if (admitted_new_demand) {
            publish_demand(demand);
        }
        current_leases_ = pending_.back().resources->prepared.resident_host_weights;
    }
    return status;
}

void WeightSubmissionTracker::mark_stream_synchronized() {
    for (Pending & pending : pending_) {
        release_completion(pending);
    }
    pending_.clear();
}

bool WeightSubmissionTracker::has_obsolete_pending() const {
    return std::any_of(pending_.begin(), pending_.end(),
                       [](const Pending & pending) { return pending.may_hold_obsolete_demand; });
}

std::vector<HostWeightLease> WeightSubmissionTracker::outstanding_leases() const {
    std::vector<HostWeightLease> leases;
    for (const Pending & pending : pending_) {
        if (pending.resources == nullptr) {
            continue;
        }
        const auto & resident = pending.resources->prepared.resident_host_weights;
        leases.insert(leases.end(), resident.begin(), resident.end());
    }
    return leases;
}

GraphExecutor::GraphExecutor(ggml_backend_hrx_context & context) : context_(context) {}

Status GraphExecutor::context_valid_for_graph_programs() const {
    Status status;
    if (context_.device == nullptr) {
        status.log("missing HRX device context");
    } else if (context_.device->architecture.empty()) {
        status.log("missing HRX target");
    }
    return status;
}

Status GraphExecutor::context_valid_for_execution() const {
    return context_valid_for_graph_programs();
}

GraphSupportResult GraphExecutor::can_execute(const ggml_cgraph & graph) const {
    GraphSupportResult result;
    if (graph.n_nodes == 0) {
        result.supported = true;
        return result;
    }
    result.status = context_valid_for_graph_programs();
    if (!result.status.success()) {
        return result;
    }
    const KernelCorpus &            corpus = get_qwen_kernel_corpus();
    const GraphProgramSupportResult support =
        context_.graph_programs.check_support(graph, corpus, context_.device->architecture);
    result.supported = support.supported;
    result.status.append(support.status);
    return result;
}

CommandProgramBindings GraphExecutor::bind_external_value_buffers(const GraphProgramMatch & match) const {
    std::vector<CommandProgramBinding> bindings;
    Status                             status;
    bindings.reserve(match.external_bindings.size());
    for (const GraphProgramExternalBinding & external : match.external_bindings) {
        ValueBufferBinding    value_binding;
        CommandProgramBinding binding;
        binding.value = external.value;
        if (ggml_backend_hrx_resolve_value_buffer(external.tensor, value_binding)) {
            std::array<int64_t, GGML_MAX_DIMS> ne = {};
            std::array<size_t, GGML_MAX_DIMS> nb = {};
            for (size_t dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
                ne[dimension] = external.tensor->ne[dimension];
                nb[dimension] = external.tensor->nb[dimension];
            }
            binding = make_command_program_binding(external.value, value_binding, external.tensor->type, ne, nb,
                                                   ggml_nbytes(external.tensor) == 0);
        } else {
            status.log("external value %d is not bound", external.value.value);
        }
        bindings.push_back(binding);
    }
    return CommandProgramBindings::from_bindings(std::move(bindings), status);
}

GraphExecutionResult GraphExecutor::execute(const ggml_cgraph & graph) const {
    GraphExecutionResult result;
    if (graph.n_nodes == 0) {
        result.code = GGML_STATUS_SUCCESS;
        return result;
    }
    result.status = context_valid_for_execution();
    if (!result.status.success()) {
        return result;
    }

    const KernelCorpus & corpus = get_qwen_kernel_corpus();
    GraphProgramLookup   lookup = context_.graph_programs.get_or_build(graph, corpus, context_.device->architecture);
    if (!lookup.valid()) {
        result.status.append(lookup.status);
        result.status.append(lookup.match.status);
        if (result.status.success()) {
            result.status.log("build HRX graph program failed");
        }
        return result;
    }

    const bool        use_graph_prepared = lookup.program->can_use_prepared_fast_path(graph);
    GraphProgramMatch binding_match = std::move(lookup.match);
    const bool has_prepared = lookup.program->has_prepared_program();
    const bool may_use_partial_bindings = use_graph_prepared && has_prepared;
    bool full_bindings = true;
    if (may_use_partial_bindings) {
        const bool imported_sources_current = lookup.program->imported_sources_current();
        binding_match = imported_sources_current ? lookup.program->match_host_staging_graph(graph) :
                                                   lookup.program->match_trusted_graph(graph, true);
        full_bindings = !imported_sources_current;
    }
    if (!binding_match.valid()) {
        result.status.append(binding_match.status);
        return result;
    }

    CommandProgramBindings bindings = bind_external_value_buffers(binding_match);
    if (!bindings.valid()) {
        result.status.append(bindings.status);
        return result;
    }
    std::shared_ptr<const std::vector<HostWeightSource>> cached_demand;
    if (!full_bindings) {
        const GraphProgramMatch candidate_match = lookup.program->match_weight_demand_graph(graph);
        if (!candidate_match.valid()) {
            result.status.append(candidate_match.status);
            return result;
        }
        CommandProgramBindings candidate_bindings = bind_external_value_buffers(candidate_match);
        if (!candidate_bindings.valid()) {
            result.status.append(candidate_bindings.status);
            return result;
        }
        cached_demand = lookup.program->cached_weight_demand_for(candidate_bindings);
    }
    std::optional<CommandProgramBindings> full_demand_bindings;
    const CommandProgramBindings * demand_bindings = &bindings;
    const bool needs_full_demand_bindings = !full_bindings && cached_demand == nullptr;
    if (needs_full_demand_bindings) {
        const GraphProgramMatch demand_match = lookup.program->match_trusted_graph(graph, true);
        if (!demand_match.valid()) {
            result.status.append(demand_match.status);
            return result;
        }
        full_demand_bindings.emplace(bind_external_value_buffers(demand_match));
        if (!full_demand_bindings->valid()) {
            result.status.append(full_demand_bindings->status);
            return result;
        }
        demand_bindings = &*full_demand_bindings;
    }
    std::vector<HostWeightSource> collected_demand;
    if (cached_demand == nullptr) {
        const CommandProgramWeightDemands weight_demands =
            bind_command_program_weight_demands(lookup.program->weight_requests(), *demand_bindings);
        if (!weight_demands.valid()) {
            result.status.append(weight_demands.status);
            return result;
        }
        for (const CommandProgramWeightDemand & demand : weight_demands.weights) {
            collected_demand.push_back(demand.source);
        }
    }
    const std::vector<HostWeightSource> & current_demand =
        cached_demand != nullptr ? *cached_demand : collected_demand;
    const bool demand_changed = !context_.weight_submissions.demand_matches(current_demand);
    std::vector<HostWeightSource> merged_demand;
    if (demand_changed) {
        merged_demand = context_.weight_submissions.merged_demand_for(current_demand);
    }
    bool completed_before = false;
    result.status = context_.weight_submissions.poll_completed(&completed_before);
    if (!result.status.success()) {
        return result;
    }
    if (demand_changed) {
        context_.weight_submissions.release_current_leases();
        context_.graph_programs.invalidate_obsolete_weights(merged_demand);
        context_.prepared_programs.invalidate_obsolete_weights(merged_demand);
    }
    const bool prepared_program_missing = !lookup.program->has_prepared_program();
    const bool restore_full_bindings = use_graph_prepared && prepared_program_missing;
    if (restore_full_bindings) {
        if (full_demand_bindings) {
            bindings = std::move(*full_demand_bindings);
            full_bindings = true;
        } else if (!full_bindings) {
            const GraphProgramMatch full_match = lookup.program->match_trusted_graph(graph, true);
            if (!full_match.valid()) {
                result.status.append(full_match.status);
                return result;
            }
            bindings = bind_external_value_buffers(full_match);
            if (!bindings.valid()) {
                result.status.append(bindings.status);
                return result;
            }
            full_bindings = true;
        }
    }
    const bool reconcile_before = demand_changed || completed_before;
    if (reconcile_before) {
        const auto & retained = demand_changed ? merged_demand : context_.weight_submissions.current_demand();
        context_.host_weights.evict_except(retained, context_.weight_submissions.outstanding_leases());
    }
    const CommandProgramExecutionContext execution_context = {
        context_.device->device,
        context_.stream,
        context_.device->architecture.c_str(),
        &corpus,
        &context_.kernel_executables,
        &context_.transient_arena,
        &context_.host_transfers,
        &context_.host_weights,
        &context_.device->host_buffers,
        &context_.graph_replay_state,
    };
    const PreparedCommandProgramCacheExecutionResult execution =
        use_graph_prepared ? lookup.program->execute_with_result(execution_context, bindings, full_bindings) :
                             context_.prepared_programs.execute_with_result(execution_context, lookup.program->uid(),
                                                                            lookup.program->command_shape_hash(),
                                                                            lookup.program->commands(), bindings);
    result.graph_replay_event = execution.graph_replay_event;
    const Status completion_status = context_.weight_submissions.record(
        context_.device->device, context_.stream, execution.execution_resources, current_demand, execution.success,
        demand_changed);
    result.status.append(completion_status);
    const bool admitted = execution.success && completion_status.success();
    const bool cache_needs_refresh = cached_demand == nullptr || restore_full_bindings;
    const bool cache_can_refresh = admitted && use_graph_prepared && cache_needs_refresh;
    if (cache_can_refresh) {
        const CommandProgramBindings & cache_bindings =
            full_bindings ? bindings : *full_demand_bindings;
        lookup.program->remember_weight_demand(cache_bindings, current_demand);
    }
    if (!admitted) {
        const auto & retained_demand = context_.weight_submissions.current_demand();
        context_.graph_programs.invalidate_obsolete_weights(retained_demand);
        context_.prepared_programs.invalidate_obsolete_weights(retained_demand);
    }
    bool completed_after = false;
    const Status retirement_status = context_.weight_submissions.poll_completed(&completed_after);
    result.status.append(retirement_status);
    const bool reconcile_after = demand_changed || completed_after || !admitted;
    if (reconcile_after) {
        const std::vector<HostWeightLease> outstanding = context_.weight_submissions.outstanding_leases();
        context_.host_weights.evict_except(context_.weight_submissions.current_demand(), outstanding);
        const auto & current = context_.weight_submissions.current_leases();
        const bool outstanding_only_allocation = std::any_of(outstanding.begin(), outstanding.end(),
            [&](const HostWeightLease & lease) {
                return std::none_of(current.begin(), current.end(),
                    [&](const HostWeightLease & active) { return active.buffer() == lease.buffer(); });
            });
        const bool report_overlap = admitted && outstanding_only_allocation;
        if (report_overlap) {
            std::vector<HostWeightLease> coexisting = current;
            coexisting.insert(coexisting.end(), outstanding.begin(), outstanding.end());
            context_.host_weights.report_coexisting_representations(coexisting,
                                                                    "current_vs_outstanding");
        }
    }
    if (execution.execution_resources != nullptr) {
        // A failed return can still follow queued writes or a successful fallback synchronization.
        lookup.program->mark_imported_writes(graph);
    }
    if (!result.status.success()) {
        return result;
    }
    if (!execution.success) {
        result.status.append(execution.status);
        if (result.status.success()) {
            result.status.log("execute HRX command program failed");
        }
        return result;
    }

    result.code = GGML_STATUS_SUCCESS;
    return result;
}

}  // namespace ggml::hrx
