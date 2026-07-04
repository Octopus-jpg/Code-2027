#include "htc/rtw.hpp"
#include "htc/runtime_clock.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace htc {
namespace {

using SteadyClock = std::chrono::steady_clock;
constexpr Count kGainInitProgressPercentStep = 1U;
constexpr Count kGainInitChunkProgressPercentStep = 5U;
constexpr Count kGainInitLargeSampleFormulaNodesThreshold = 20000U;
constexpr Count kGainInitLargeSampleHyperedgesThreshold = 2000U;
constexpr Count kGainInitLargeSampleCandidatesThreshold = 20000U;
constexpr Count kLocalCandidateFallbackPercent = 85U;
constexpr Count kLocalCandidateFallbackMinSavings = 64U;
constexpr std::uint64_t kBatchScratchBudgetBytes = 20ULL * 1024ULL * 1024ULL * 1024ULL;
constexpr std::uint64_t kBatchStateBytesEstimate = 32ULL;
constexpr long double kBatchBudgetUtilization = 0.60L;

double elapsed_seconds(SteadyClock::time_point begin, SteadyClock::time_point end) {
    return std::chrono::duration<double>(end - begin).count();
}

Count ceil_to_count(long double value) {
    if (!(value > 0.0L)) {
        return 1U;
    }
    const long double ceil_value = std::ceil(value);
    const long double count_max = static_cast<long double>(std::numeric_limits<Count>::max());
    if (!std::isfinite(static_cast<double>(ceil_value)) || ceil_value >= count_max) {
        return std::numeric_limits<Count>::max();
    }
    return static_cast<Count>(ceil_value);
}

std::vector<Count> build_progress_checkpoints(Count total, Count percent_step) {
    std::vector<Count> checkpoints;
    if (total == 0U || percent_step == 0U) {
        return checkpoints;
    }

    const Count step = std::min<Count>(percent_step, 100U);
    for (Count pct = step; pct <= 100U; pct += step) {
        Count checkpoint = ceil_to_count((static_cast<long double>(total) *
                                          static_cast<long double>(pct)) /
                                         100.0L);
        checkpoint = std::max<Count>(1U, std::min<Count>(checkpoint, total));
        if (checkpoints.empty() || checkpoints.back() != checkpoint) {
            checkpoints.push_back(checkpoint);
        }
    }
    if (checkpoints.empty() || checkpoints.back() != total) {
        checkpoints.push_back(total);
    }
    return checkpoints;
}

template <typename T>
RTWRange<T> make_range(const std::vector<T>& values, std::size_t begin, std::size_t end) {
    if (end <= begin) {
        return RTWRange<T>{};
    }
    if (end > values.size()) {
        throw std::logic_error("CSR range out of bounds");
    }
    return RTWRange<T>{values.data() + begin, end - begin};
}

std::vector<NodeId> build_local_candidate_priority_set(
    const RTWSample& sample,
    const std::vector<NodeId>& activated_nodes,
    const std::vector<GateId>& affected_gates) {
    if (sample.formula_nodes.empty()) {
        return {};
    }
    if (activated_nodes.empty() && affected_gates.empty()) {
        return {};
    }

    std::vector<std::uint8_t> visited(sample.formula_nodes.size(), 0U);
    std::queue<std::size_t> queue;
    auto enqueue_node = [&](NodeId node) {
        std::size_t idx = 0U;
        if (!sample.try_get_formula_index(node, &idx)) {
            return;
        }
        if (visited[idx] != 0U) {
            return;
        }
        visited[idx] = 1U;
        queue.push(idx);
    };

    for (NodeId node : activated_nodes) {
        enqueue_node(node);
    }
    for (GateId gid : affected_gates) {
        if (gid >= sample.hyperedges.size()) {
            throw std::logic_error("affected gate id out of RTW sample range");
        }
        for (NodeId member : sample.hyperedge_nodes(gid)) {
            enqueue_node(member);
        }
    }

    while (!queue.empty()) {
        const std::size_t node_idx = queue.front();
        queue.pop();
        const NodeId node = sample.formula_nodes[node_idx];

        for (NodeId src : sample.in_dep_sources(node)) {
            enqueue_node(src);
        }

        for (GateId gid : sample.incident_hyperedges(node)) {
            if (gid >= sample.hyperedges.size()) {
                throw std::logic_error("incident gate id out of RTW sample range");
            }
            for (NodeId member : sample.hyperedge_nodes(gid)) {
                enqueue_node(member);
            }
        }
    }

    std::vector<NodeId> prioritized_candidates;
    prioritized_candidates.reserve(sample.formula_nodes.size());
    for (std::size_t idx = 0; idx < sample.formula_nodes.size(); ++idx) {
        if (visited[idx] != 0U) {
            prioritized_candidates.push_back(sample.formula_nodes[idx]);
        }
    }
    return prioritized_candidates;
}

}  // namespace

void RTWSample::add_formula_node(NodeId v) {
    if (formula_pos_.find(v) != formula_pos_.end()) {
        return;
    }
    formula_pos_[v] = formula_nodes.size();
    formula_nodes.push_back(v);
}

void RTWSample::add_dependency(NodeId src, NodeId dst) {
    add_formula_node(src);
    add_formula_node(dst);
    deps.push_back(RTWDependency{src, dst});
}

void RTWSample::add_hyperedge(HyperedgeId h,
                              std::uint32_t threshold,
                              const std::vector<NodeId>& nodes) {
    for (NodeId u : nodes) {
        add_formula_node(u);
    }

    if (hyperedges.size() >= static_cast<std::size_t>(kInvalidHyperedgeId)) {
        throw std::overflow_error("too many RTW hyperedges in one sample");
    }

    const std::uint64_t offset64 = static_cast<std::uint64_t>(hyperedge_nodes_flat_.size());
    const std::uint64_t count64 = static_cast<std::uint64_t>(nodes.size());
    const std::uint64_t max32 = static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max());
    if (offset64 > max32 || count64 > max32 || offset64 + count64 > max32) {
        throw std::overflow_error("RTW hyperedge node flat index exceeds uint32 range");
    }

    hyperedges.push_back(RTWHyperedge{
        h,
        threshold,
        static_cast<std::uint32_t>(offset64),
        static_cast<std::uint32_t>(count64),
    });
    hyperedge_nodes_flat_.insert(hyperedge_nodes_flat_.end(), nodes.begin(), nodes.end());
}

bool RTWSample::contains_formula(NodeId v) const {
    return formula_pos_.find(v) != formula_pos_.end();
}

bool RTWSample::try_get_formula_index(NodeId v, std::size_t* index) const {
    const auto it = formula_pos_.find(v);
    if (it == formula_pos_.end()) {
        return false;
    }
    if (index != nullptr) {
        *index = it->second;
    }
    return true;
}

Count RTWSample::gate_input_incidence_count() const {
    Count total = 0U;
    for (const auto& h : hyperedges) {
        const Count s = static_cast<Count>(h.node_count);
        total += s * (s - 1U);
    }
    return total;
}

Count RTWSample::gate_count() const {
    Count total = 0U;
    for (const auto& h : hyperedges) {
        total += static_cast<Count>(h.node_count);
    }
    return total;
}

Count RTWSample::witness_size() const {
    return static_cast<Count>(formula_nodes.size()) +
           static_cast<Count>(deps.size()) +
           gate_count() +
           gate_input_incidence_count();
}

void RTWSample::build_indices() {
    const std::size_t formula_n = formula_nodes.size();
    out_dep_offsets_.assign(formula_n + 1U, 0U);
    in_dep_offsets_.assign(formula_n + 1U, 0U);
    incident_hyperedge_offsets_.assign(formula_n + 1U, 0U);

    auto checked_increment = [](std::uint32_t* value, const char* label) {
        if (*value == std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error(std::string(label) + " count exceeds uint32 range");
        }
        ++(*value);
    };

    for (const auto& dep : deps) {
        std::size_t src_idx = 0;
        std::size_t dst_idx = 0;
        if (!try_get_formula_index(dep.src, &src_idx) ||
            !try_get_formula_index(dep.dst, &dst_idx)) {
            throw std::logic_error("RTW dependency endpoint missing in formula nodes");
        }
        checked_increment(&out_dep_offsets_[src_idx + 1U], "RTW out dependency");
        checked_increment(&in_dep_offsets_[dst_idx + 1U], "RTW in dependency");
    }

    for (std::size_t i = 0; i < hyperedges.size(); ++i) {
        const auto nodes = hyperedge_nodes(static_cast<HyperedgeId>(i));
        for (NodeId u : nodes) {
            std::size_t input_idx = 0;
            if (!try_get_formula_index(u, &input_idx)) {
                throw std::logic_error("RTW hyperedge node missing in formula nodes");
            }
            checked_increment(&incident_hyperedge_offsets_[input_idx + 1U], "RTW incident hyperedge");
        }
    }

    auto prefix_sum_offsets = [](std::vector<std::uint32_t>* offsets, const char* label) {
        for (std::size_t i = 1; i < offsets->size(); ++i) {
            const std::uint64_t prefix = static_cast<std::uint64_t>((*offsets)[i - 1U]) +
                                         static_cast<std::uint64_t>((*offsets)[i]);
            if (prefix > std::numeric_limits<std::uint32_t>::max()) {
                throw std::overflow_error(std::string(label) + " prefix exceeds uint32 range");
            }
            (*offsets)[i] = static_cast<std::uint32_t>(prefix);
        }
    };

    prefix_sum_offsets(&out_dep_offsets_, "RTW out dependency");
    prefix_sum_offsets(&in_dep_offsets_, "RTW in dependency");
    prefix_sum_offsets(&incident_hyperedge_offsets_, "RTW incident hyperedge");

    out_dep_values_.assign(out_dep_offsets_.empty() ? 0U
                                                    : static_cast<std::size_t>(out_dep_offsets_.back()),
                           kInvalidNodeId);
    in_dep_values_.assign(in_dep_offsets_.empty() ? 0U
                                                  : static_cast<std::size_t>(in_dep_offsets_.back()),
                          kInvalidNodeId);
    incident_hyperedge_values_.assign(
        incident_hyperedge_offsets_.empty()
            ? 0U
            : static_cast<std::size_t>(incident_hyperedge_offsets_.back()),
        kInvalidHyperedgeId);

    std::vector<std::uint32_t> out_cursor = out_dep_offsets_;
    std::vector<std::uint32_t> in_cursor = in_dep_offsets_;
    std::vector<std::uint32_t> incident_hyperedge_cursor = incident_hyperedge_offsets_;

    for (const auto& dep : deps) {
        std::size_t src_idx = 0;
        std::size_t dst_idx = 0;
        if (!try_get_formula_index(dep.src, &src_idx) ||
            !try_get_formula_index(dep.dst, &dst_idx)) {
            throw std::logic_error("RTW dependency endpoint missing in formula nodes");
        }
        const std::size_t out_pos = static_cast<std::size_t>(out_cursor[src_idx]++);
        const std::size_t in_pos = static_cast<std::size_t>(in_cursor[dst_idx]++);
        out_dep_values_[out_pos] = dep.dst;
        in_dep_values_[in_pos] = dep.src;
    }

    for (std::size_t i = 0; i < hyperedges.size(); ++i) {
        const HyperedgeId hid = static_cast<HyperedgeId>(i);
        for (NodeId u : hyperedge_nodes(hid)) {
            std::size_t input_idx = 0;
            if (!try_get_formula_index(u, &input_idx)) {
                throw std::logic_error("RTW hyperedge node missing in formula nodes");
            }
            const std::size_t input_pos = static_cast<std::size_t>(incident_hyperedge_cursor[input_idx]++);
            incident_hyperedge_values_[input_pos] = hid;
        }
    }
}

RTWRange<NodeId> RTWSample::node_range_from_csr(const std::vector<std::uint32_t>& offsets,
                                                const std::vector<NodeId>& values,
                                                std::size_t idx) {
    if (idx + 1U >= offsets.size()) {
        return RTWRange<NodeId>{};
    }
    const std::size_t begin = static_cast<std::size_t>(offsets[idx]);
    const std::size_t end = static_cast<std::size_t>(offsets[idx + 1U]);
    return make_range(values, begin, end);
}

RTWRange<HyperedgeId> RTWSample::hyperedge_range_from_csr(
    const std::vector<std::uint32_t>& offsets,
    const std::vector<HyperedgeId>& values,
    std::size_t idx) {
    if (idx + 1U >= offsets.size()) {
        return RTWRange<HyperedgeId>{};
    }
    const std::size_t begin = static_cast<std::size_t>(offsets[idx]);
    const std::size_t end = static_cast<std::size_t>(offsets[idx + 1U]);
    return make_range(values, begin, end);
}

RTWRange<NodeId> RTWSample::out_dep_targets(NodeId v) const {
    std::size_t idx = 0;
    return try_get_formula_index(v, &idx)
               ? node_range_from_csr(out_dep_offsets_, out_dep_values_, idx)
               : RTWRange<NodeId>{};
}

RTWRange<NodeId> RTWSample::in_dep_sources(NodeId v) const {
    std::size_t idx = 0;
    return try_get_formula_index(v, &idx)
               ? node_range_from_csr(in_dep_offsets_, in_dep_values_, idx)
               : RTWRange<NodeId>{};
}

RTWRange<HyperedgeId> RTWSample::incident_hyperedges(NodeId v) const {
    std::size_t idx = 0;
    return try_get_formula_index(v, &idx)
               ? hyperedge_range_from_csr(incident_hyperedge_offsets_, incident_hyperedge_values_, idx)
               : RTWRange<HyperedgeId>{};
}

RTWRange<NodeId> RTWSample::hyperedge_nodes(HyperedgeId hid) const {
    if (hid >= hyperedges.size()) {
        throw std::out_of_range("RTW hyperedge id out of range");
    }
    const auto& h = hyperedges[hid];
    const std::size_t begin = static_cast<std::size_t>(h.node_offset);
    const std::size_t end = begin + static_cast<std::size_t>(h.node_count);
    return make_range(hyperedge_nodes_flat_, begin, end);
}

void RTWStats::add_sample(const RTWSample& sample) {
    ++witness_count;
    const Count formula_nodes = static_cast<Count>(sample.formula_nodes.size());
    const Count dependencies = static_cast<Count>(sample.deps.size());
    const Count gates = sample.gate_count();
    const Count gate_inputs = sample.gate_input_incidence_count();
    total_formula_nodes += formula_nodes;
    total_dependencies += dependencies;
    total_gates += gates;
    total_gate_input_incidences += gate_inputs;
    total_witness_size += formula_nodes + dependencies + gates + gate_inputs;
}

double RTWStats::avg_formula_nodes() const {
    return witness_count == 0U ? 0.0
                               : static_cast<double>(total_formula_nodes) /
                                     static_cast<double>(witness_count);
}

double RTWStats::avg_dependencies() const {
    return witness_count == 0U ? 0.0
                               : static_cast<double>(total_dependencies) /
                                     static_cast<double>(witness_count);
}

double RTWStats::avg_gates() const {
    return witness_count == 0U ? 0.0
                               : static_cast<double>(total_gates) /
                                     static_cast<double>(witness_count);
}

double RTWStats::avg_gate_input_incidences() const {
    return witness_count == 0U ? 0.0
                               : static_cast<double>(total_gate_input_incidences) /
                                     static_cast<double>(witness_count);
}

double RTWStats::avg_witness_size() const {
    return witness_count == 0U ? 0.0
                               : static_cast<double>(total_witness_size) /
                                     static_cast<double>(witness_count);
}

double RTWCollectionStats::avg_dirty_candidates_per_seed() const {
    return seed_additions == 0U ? 0.0
                                : static_cast<double>(dirty_candidates) /
                                      static_cast<double>(seed_additions);
}

double RTWCollectionStats::avg_affected_samples_per_seed() const {
    return seed_additions == 0U ? 0.0
                                : static_cast<double>(affected_samples) /
                                      static_cast<double>(seed_additions);
}

RTWSampler::RTWSampler(const HybridHypergraph& graph) : graph_(graph) {}

RTWSample RTWSampler::sample(Random& random) const {
    if (graph_.num_nodes() == 0U) {
        throw std::runtime_error("cannot sample RTW from empty graph");
    }
    const NodeId root = random.uniform_node(0U, graph_.num_nodes() - 1U);
    return sample_with_root(root, random);
}

RTWSample RTWSampler::sample_with_root(NodeId root, Random& random) const {
    if (root >= graph_.num_nodes()) {
        throw std::out_of_range("RTW root out of graph range");
    }

    RTWSample witness;
    witness.root = root;
    witness.add_formula_node(root);

    std::vector<std::uint8_t> expanded(graph_.num_nodes(), 0U);
    std::vector<std::uint8_t> included_hyperedge(graph_.num_hyperedges(), 0U);
    std::queue<NodeId> queue;
    queue.push(root);

    while (!queue.empty()) {
        const NodeId v = queue.front();
        queue.pop();

        if (expanded[v] != 0U) {
            continue;
        }
        expanded[v] = 1U;

        for (EdgeId eid : graph_.in_edges(v)) {
            const auto& e = graph_.edge(eid);
            if (random.bernoulli(e.prob)) {
                witness.add_dependency(e.src, v);
                queue.push(e.src);
            }
        }

        for (HyperedgeId hid : graph_.incident_hyperedges(v)) {
            if (included_hyperedge[hid] != 0U) {
                continue;
            }
            included_hyperedge[hid] = 1U;
            const auto& h = graph_.hyperedge(hid);
            witness.add_hyperedge(hid, h.threshold, h.nodes);
            for (NodeId u : h.nodes) {
                queue.push(u);
            }
        }
    }

    witness.build_indices();
    return witness;
}

RTWEvaluator::EvalState RTWEvaluator::run_closure(const RTWSample& sample,
                                                  const std::vector<NodeId>& seeds) const {
    EvalState state;
    state.node_true.assign(sample.formula_nodes.size(), 0U);
    state.hyperedge_count.assign(sample.hyperedges.size(), 0U);
    state.hyperedge_triggered.assign(sample.hyperedges.size(), 0U);

    std::queue<std::size_t> queue;
    for (NodeId seed : seeds) {
        std::size_t idx = 0;
        if (!sample.try_get_formula_index(seed, &idx)) {
            continue;
        }
        if (state.node_true[idx] != 0U) {
            continue;
        }
        state.node_true[idx] = 1U;
        queue.push(idx);
    }

    while (!queue.empty()) {
        const std::size_t idx = queue.front();
        queue.pop();
        const NodeId node = sample.formula_nodes[idx];

        for (NodeId dst : sample.out_dep_targets(node)) {
            std::size_t dst_idx = 0;
            if (!sample.try_get_formula_index(dst, &dst_idx)) {
                continue;
            }
            if (state.node_true[dst_idx] == 0U) {
                state.node_true[dst_idx] = 1U;
                queue.push(dst_idx);
            }
        }

        for (HyperedgeId hid : sample.incident_hyperedges(node)) {
            if (hid >= sample.hyperedges.size()) {
                throw std::logic_error("RTW hyperedge id out of range");
            }
            if (state.hyperedge_triggered[hid] != 0U) {
                continue;
            }

            const auto& hyperedge = sample.hyperedges[hid];
            if (state.hyperedge_count[hid] < hyperedge.node_count) {
                ++state.hyperedge_count[hid];
            }
            if (state.hyperedge_count[hid] < hyperedge.threshold) {
                continue;
            }

            state.hyperedge_triggered[hid] = 1U;
            for (NodeId target : sample.hyperedge_nodes(hid)) {
                std::size_t target_idx = 0;
                if (!sample.try_get_formula_index(target, &target_idx)) {
                    throw std::logic_error("RTW hyperedge node missing in formula nodes");
                }
                if (state.node_true[target_idx] == 0U) {
                    state.node_true[target_idx] = 1U;
                    queue.push(target_idx);
                }
            }
        }
    }

    return state;
}

bool RTWEvaluator::is_satisfied(const RTWSample& sample, const std::vector<NodeId>& seeds) const {
    if (sample.root == kInvalidNodeId) {
        throw std::invalid_argument("RTW sample root is invalid");
    }

    const EvalState state = run_closure(sample, seeds);
    std::size_t root_idx = 0;
    if (!sample.try_get_formula_index(sample.root, &root_idx)) {
        return false;
    }
    return state.node_true[root_idx] != 0U;
}

Count RTWEvaluator::total_gate_deficit(const RTWSample& sample,
                                       const std::vector<NodeId>& seeds) const {
    const EvalState state = run_closure(sample, seeds);

    Count deficit = 0;
    for (std::size_t i = 0; i < sample.hyperedges.size(); ++i) {
        const auto& hyperedge = sample.hyperedges[i];
        const Count s = static_cast<Count>(hyperedge.node_count);
        if (s == 0U) {
            continue;
        }
        const Count c = static_cast<Count>(state.hyperedge_count[i]);
        if (c >= s || c >= static_cast<Count>(hyperedge.threshold)) {
            continue;
        }
        deficit += (s - c) * (static_cast<Count>(hyperedge.threshold) - c);
    }
    return deficit;
}

void RTWCollection::add_sample(RTWSample sample) {
    if (samples_.size() >= static_cast<std::size_t>(kInvalidSampleId)) {
        throw std::overflow_error("too many RTW samples");
    }
    sample.build_indices();
    samples_.push_back(std::move(sample));

    occurrence_ready_ = false;
    state_ready_ = false;
    node_occurrences_.clear();
    states_.clear();
    sample_state_version_.clear();
    selected_seed_.clear();
    indexed_num_nodes_ = 0U;
    satisfied_count_ = 0;
    sample_satisfying_candidates_cache_.clear();
    cached_marginal_gain_.clear();
    cached_progress_score_.clear();
    cached_progress_signature_.clear();
    progress_cache_valid_.clear();
    last_affected_samples_.clear();
    last_affected_gates_.clear();
    candidate_version_.clear();
    clear_candidate_heap();
    stats_ = RTWCollectionStats{};
    gain_cache_ready_ = false;
}

Count RTWCollection::size() const {
    return static_cast<Count>(samples_.size());
}

void RTWCollection::build_global_occurrence_index(NodeId num_nodes) {
    node_occurrences_.assign(num_nodes, {});
    Count occurrence_index_entries = 0U;
    for (std::size_t sid = 0; sid < samples_.size(); ++sid) {
        if (sid >= static_cast<std::size_t>(kInvalidSampleId)) {
            throw std::overflow_error("sample index exceeds SampleId range");
        }
        const SampleId sample_id = static_cast<SampleId>(sid);
        const auto& sample = samples_[sid];
        for (NodeId node : sample.formula_nodes) {
            if (node >= num_nodes) {
                throw std::invalid_argument("formula node id out of range in occurrence index");
            }
            node_occurrences_[node].push_back(Occurrence{sample_id});
            ++occurrence_index_entries;
        }
    }

    indexed_num_nodes_ = num_nodes;
    selected_seed_.assign(indexed_num_nodes_, 0U);
    cached_progress_score_.assign(indexed_num_nodes_, 0U);
    cached_progress_signature_.assign(indexed_num_nodes_, 0U);
    progress_cache_valid_.assign(indexed_num_nodes_, 0U);
    candidate_version_.assign(indexed_num_nodes_, 0U);
    clear_candidate_heap();
    cached_marginal_gain_.clear();
    sample_satisfying_candidates_cache_.clear();
    sample_state_version_.clear();
    gain_cache_ready_ = false;
    last_affected_samples_.clear();
    last_affected_gates_.clear();
    stats_ = RTWCollectionStats{};
    stats_.sample_count = static_cast<Count>(samples_.size());
    stats_.occurrence_index_entries = occurrence_index_entries;
    occurrence_ready_ = true;

    if (state_ready_) {
        // Keep incremental state clean; caller can add seeds again.
        initialize_incremental_state();
    }
}

void RTWCollection::initialize_incremental_state() {
    states_.clear();
    states_.reserve(samples_.size());

    for (const auto& sample : samples_) {
        SampleState state;
        state.node_true.assign(sample.formula_nodes.size(), 0U);
        state.hyperedge_count.assign(sample.hyperedges.size(), 0U);
        state.hyperedge_triggered.assign(sample.hyperedges.size(), 0U);
        state.sample_satisfied = false;
        states_.push_back(std::move(state));
    }
    sample_state_version_.assign(samples_.size(), 0U);

    satisfied_count_ = 0;
    if (occurrence_ready_) {
        selected_seed_.assign(indexed_num_nodes_, 0U);
    }
    sample_satisfying_candidates_cache_.clear();
    cached_marginal_gain_.clear();
    if (occurrence_ready_) {
        cached_progress_score_.assign(indexed_num_nodes_, 0U);
        cached_progress_signature_.assign(indexed_num_nodes_, 0U);
        progress_cache_valid_.assign(indexed_num_nodes_, 0U);
        candidate_version_.assign(indexed_num_nodes_, 0U);
    } else {
        cached_progress_score_.clear();
        cached_progress_signature_.clear();
        progress_cache_valid_.clear();
        candidate_version_.clear();
    }
    clear_candidate_heap();
    gain_cache_ready_ = false;
    last_affected_samples_.clear();
    last_affected_gates_.clear();
    reset_runtime_stats();
    state_ready_ = true;
}

void RTWCollection::initialize_gain_cache() {
    require_occurrence_index();
    require_state_initialized();

    cached_marginal_gain_.assign(indexed_num_nodes_, 0U);
    cached_progress_score_.assign(indexed_num_nodes_, 0U);
    cached_progress_signature_.assign(indexed_num_nodes_, 0U);
    progress_cache_valid_.assign(indexed_num_nodes_, 0U);
    candidate_version_.assign(indexed_num_nodes_, 0U);
    sample_satisfying_candidates_cache_.assign(samples_.size(), {});
    clear_candidate_heap();
    Count recomputations = 0U;
    const auto t0 = SteadyClock::now();
    const Count total_samples = static_cast<Count>(samples_.size());
    const std::vector<Count> progress_checkpoints =
        build_progress_checkpoints(total_samples, kGainInitProgressPercentStep);
    std::size_t next_progress_checkpoint = 0U;
    if (!progress_checkpoints.empty()) {
        std::cout << "[rtw-greedy] gain init started: samples=" << total_samples
                  << ", checkpoint_step=" << kGainInitProgressPercentStep << "%"
                  << ", total_elapsed_s=" << runtime_elapsed_seconds()
                  << std::endl;
    }

    for (std::size_t sid = 0; sid < samples_.size(); ++sid) {
        const auto& sample = samples_[sid];
        const auto& state = states_[sid];
        if (!state.sample_satisfied) {
            const bool log_large_sample =
                static_cast<Count>(sample.formula_nodes.size()) >=
                    kGainInitLargeSampleFormulaNodesThreshold ||
                static_cast<Count>(sample.hyperedges.size()) >=
                    kGainInitLargeSampleHyperedgesThreshold;
            const auto t_sample_begin = SteadyClock::now();
            const Count checks_before = recomputations;
            if (log_large_sample) {
                std::cout << "[rtw-greedy] gain init sample started: sid="
                          << (sid + 1U) << "/" << total_samples
                          << ", root=" << sample.root
                          << ", formula_nodes=" << sample.formula_nodes.size()
                          << ", deps=" << sample.deps.size()
                          << ", hyperedges=" << sample.hyperedges.size()
                          << ", total_elapsed_s=" << runtime_elapsed_seconds()
                          << std::endl;
            }
            std::vector<NodeId> satisfying_candidates;
            recomputations +=
                compute_singleton_satisfying_candidates(
                    sample, state, &satisfying_candidates, nullptr);
            sample_satisfying_candidates_cache_[sid] = std::move(satisfying_candidates);
            for (NodeId candidate : sample_satisfying_candidates_cache_[sid]) {
                ++cached_marginal_gain_[candidate];
            }
            if (log_large_sample) {
                std::cout << "[rtw-greedy] gain init sample done: sid="
                          << (sid + 1U) << "/" << total_samples
                          << ", checks_added=" << (recomputations - checks_before)
                          << ", satisfying_candidates="
                          << sample_satisfying_candidates_cache_[sid].size()
                          << ", elapsed_s=" << elapsed_seconds(t_sample_begin, SteadyClock::now())
                          << ", total_elapsed_s=" << runtime_elapsed_seconds()
                          << std::endl;
            }
        }

        const Count completed = static_cast<Count>(sid + 1U);
        while (next_progress_checkpoint < progress_checkpoints.size() &&
               completed >= progress_checkpoints[next_progress_checkpoint]) {
            const Count checkpoint = progress_checkpoints[next_progress_checkpoint];
            const double elapsed = elapsed_seconds(t0, SteadyClock::now());
            const double rate = (elapsed > 1e-12) ? (static_cast<double>(checkpoint) / elapsed) : 0.0;
            const double eta = (rate > 1e-12)
                                   ? (static_cast<double>(total_samples - checkpoint) / rate)
                                   : 0.0;
            const double pct = static_cast<double>(checkpoint) * 100.0 /
                               static_cast<double>(total_samples);
            std::cout << "[rtw-greedy] gain init progress: " << checkpoint
                      << "/" << total_samples
                      << " (" << pct << "%)"
                      << ", formula_node_checks=" << recomputations
                      << ", elapsed_s=" << elapsed
                      << ", eta_s=" << eta
                      << ", rate=" << rate << " samples/s"
                      << ", total_elapsed_s=" << runtime_elapsed_seconds()
                      << std::endl;
            ++next_progress_checkpoint;
        }
    }

    std::vector<CandidateHeapEntry> heap_entries;
    heap_entries.reserve(indexed_num_nodes_);
    for (NodeId v = 0; v < indexed_num_nodes_; ++v) {
        if (selected_seed_[v] == 0U) {
            heap_entries.push_back(CandidateHeapEntry{
                cached_marginal_gain_[v],
                v,
                candidate_version_[v],
            });
        }
    }
    candidate_heap_ = std::priority_queue<CandidateHeapEntry,
                                          std::vector<CandidateHeapEntry>,
                                          CandidateHeapCompare>(
        CandidateHeapCompare{}, std::move(heap_entries));
    const auto t1 = SteadyClock::now();
    stats_.gain_cache_initial_recomputations += recomputations;
    stats_.gain_cache_initialization_seconds += std::chrono::duration<double>(t1 - t0).count();
    if (!progress_checkpoints.empty()) {
        std::cout << "[rtw-greedy] gain init done: samples=" << total_samples
                  << ", formula_node_checks=" << recomputations
                  << ", seconds=" << elapsed_seconds(t0, t1)
                  << ", total_elapsed_s=" << runtime_elapsed_seconds()
                  << std::endl;
    }
    gain_cache_ready_ = true;
}

Count RTWCollection::compute_singleton_satisfying_candidates(
    const RTWSample& sample,
    const SampleState& state,
    std::vector<NodeId>* satisfying_candidates,
    const std::vector<NodeId>* prioritized_candidates) const {
    if (satisfying_candidates == nullptr) {
        throw std::invalid_argument("satisfying_candidates pointer cannot be null");
    }
    satisfying_candidates->clear();

    if (state.sample_satisfied || sample.formula_nodes.empty()) {
        return 0U;
    }

    std::vector<std::size_t> candidate_local_indices;
    auto add_full_candidate_local_indices = [&candidate_local_indices, this, &sample]() {
        candidate_local_indices.clear();
        candidate_local_indices.reserve(sample.formula_nodes.size());
        for (std::size_t local_idx = 0; local_idx < sample.formula_nodes.size(); ++local_idx) {
            const NodeId candidate = sample.formula_nodes[local_idx];
            if (candidate >= indexed_num_nodes_) {
                throw std::out_of_range("candidate out of occurrence index range");
            }
            if (selected_seed_[candidate] == 0U) {
                candidate_local_indices.push_back(local_idx);
            }
        }
    };

    if (prioritized_candidates == nullptr || prioritized_candidates->empty()) {
        add_full_candidate_local_indices();
    } else {
        std::vector<std::uint8_t> used(sample.formula_nodes.size(), 0U);
        candidate_local_indices.reserve(
            std::min<std::size_t>(prioritized_candidates->size(), sample.formula_nodes.size()));
        for (NodeId candidate : *prioritized_candidates) {
            std::size_t local_idx = 0U;
            if (!sample.try_get_formula_index(candidate, &local_idx)) {
                continue;
            }
            if (used[local_idx] != 0U) {
                continue;
            }
            if (candidate >= indexed_num_nodes_) {
                throw std::out_of_range("candidate out of occurrence index range");
            }
            if (selected_seed_[candidate] != 0U) {
                continue;
            }
            used[local_idx] = 1U;
            candidate_local_indices.push_back(local_idx);
        }

        Count full_candidate_count = 0U;
        for (std::size_t local_idx = 0; local_idx < sample.formula_nodes.size(); ++local_idx) {
            const NodeId candidate = sample.formula_nodes[local_idx];
            if (candidate >= indexed_num_nodes_) {
                throw std::out_of_range("candidate out of occurrence index range");
            }
            if (selected_seed_[candidate] == 0U) {
                ++full_candidate_count;
            }
        }

        bool fallback_to_full = candidate_local_indices.empty();
        if (!fallback_to_full) {
            const Count prioritized_count = static_cast<Count>(candidate_local_indices.size());
            if (prioritized_count < full_candidate_count) {
                const Count savings = full_candidate_count - prioritized_count;
                const bool too_close =
                    static_cast<std::uint64_t>(prioritized_count) * 100ULL >=
                    static_cast<std::uint64_t>(full_candidate_count) *
                        static_cast<std::uint64_t>(kLocalCandidateFallbackPercent);
                const bool low_savings = savings < kLocalCandidateFallbackMinSavings;
                fallback_to_full = too_close || low_savings;
            }
        }
        if (fallback_to_full) {
            add_full_candidate_local_indices();
        }
    }
    if (candidate_local_indices.empty()) {
        return 0U;
    }

    const bool in_gain_init = !gain_cache_ready_;
    const bool log_chunk_diagnostics =
        in_gain_init &&
        (static_cast<Count>(sample.formula_nodes.size()) >=
             kGainInitLargeSampleFormulaNodesThreshold ||
         static_cast<Count>(sample.hyperedges.size()) >=
             kGainInitLargeSampleHyperedgesThreshold ||
         static_cast<Count>(candidate_local_indices.size()) >=
             kGainInitLargeSampleCandidatesThreshold);

    bool has_non_empty_base_state = false;
    for (std::uint8_t mark : state.node_true) {
        if (mark != 0U) {
            has_non_empty_base_state = true;
            break;
        }
    }
    if (!has_non_empty_base_state) {
        for (std::uint32_t count : state.hyperedge_count) {
            if (count != 0U) {
                has_non_empty_base_state = true;
                break;
            }
        }
    }
    if (!has_non_empty_base_state) {
        for (std::uint8_t triggered : state.hyperedge_triggered) {
            if (triggered != 0U) {
                has_non_empty_base_state = true;
                break;
            }
        }
    }
    if (has_non_empty_base_state) {
        std::size_t root_idx = 0U;
        if (!sample.try_get_formula_index(sample.root, &root_idx)) {
            return 0U;
        }
        std::vector<std::size_t> inactive_local_indices;
        inactive_local_indices.reserve(candidate_local_indices.size());
        for (std::size_t local_idx : candidate_local_indices) {
            if (state.node_true[local_idx] == 0U) {
                inactive_local_indices.push_back(local_idx);
            }
        }
        if (inactive_local_indices.empty()) {
            return static_cast<Count>(candidate_local_indices.size());
        }

        if (sample.formula_nodes.size() > std::numeric_limits<std::uint32_t>::max() ||
            sample.hyperedges.size() > std::numeric_limits<std::uint32_t>::max() ||
            inactive_local_indices.size() > std::numeric_limits<std::uint32_t>::max()) {
            // Conservative fallback for index packing limits.
            for (std::size_t local_idx : inactive_local_indices) {
                const NodeId candidate = sample.formula_nodes[local_idx];
                bool would_be_satisfied = false;
                if (simulate_add_seed(sample, state, candidate, &would_be_satisfied, nullptr) &&
                    would_be_satisfied) {
                    satisfying_candidates->push_back(candidate);
                }
            }
            return static_cast<Count>(candidate_local_indices.size());
        }

        Count inactive_nodes = 0U;
        for (std::uint8_t mark : state.node_true) {
            if (mark == 0U) {
                ++inactive_nodes;
            }
        }
        Count inactive_gates = 0U;
        for (HyperedgeId hid = 0; hid < sample.hyperedges.size(); ++hid) {
            if (state.hyperedge_triggered[hid] != 0U) {
                continue;
            }
            bool touches_inactive = false;
            for (NodeId member : sample.hyperedge_nodes(hid)) {
                std::size_t member_idx = 0U;
                if (!sample.try_get_formula_index(member, &member_idx)) {
                    throw std::logic_error("RTW hyperedge node missing during inactive scan");
                }
                if (state.node_true[member_idx] == 0U) {
                    touches_inactive = true;
                    break;
                }
            }
            if (touches_inactive) {
                ++inactive_gates;
            }
        }

        const Count work_per_label = std::max<Count>(1U, inactive_nodes + inactive_gates);
        const long double usable_bytes =
            static_cast<long double>(kBatchScratchBudgetBytes) * kBatchBudgetUtilization;
        const Count state_budget = std::max<Count>(
            1U,
            static_cast<Count>(usable_bytes / static_cast<long double>(kBatchStateBytesEstimate)));
        Count chunk_size = state_budget / work_per_label;
        if (chunk_size == 0U) {
            chunk_size = 1U;
        }
        if (chunk_size > static_cast<Count>(inactive_local_indices.size())) {
            chunk_size = static_cast<Count>(inactive_local_indices.size());
        }

        if (chunk_size <= 1U && inactive_local_indices.size() > 1U) {
            for (std::size_t local_idx : inactive_local_indices) {
                const NodeId candidate = sample.formula_nodes[local_idx];
                bool would_be_satisfied = false;
                if (simulate_add_seed(sample, state, candidate, &would_be_satisfied, nullptr) &&
                    would_be_satisfied) {
                    satisfying_candidates->push_back(candidate);
                }
            }
            return static_cast<Count>(candidate_local_indices.size());
        }

        auto pack_key = [](std::uint32_t high, std::uint32_t low) -> std::uint64_t {
            return (static_cast<std::uint64_t>(high) << 32U) | static_cast<std::uint64_t>(low);
        };

        const std::size_t total_inactive = inactive_local_indices.size();
        const std::size_t total_chunks =
            (total_inactive + static_cast<std::size_t>(chunk_size) - 1U) /
            static_cast<std::size_t>(chunk_size);
        Count completed_chunks = 0U;
        std::vector<Count> chunk_checkpoints;
        std::size_t next_chunk_checkpoint = 0U;
        if (log_chunk_diagnostics && total_chunks > 1U) {
            chunk_checkpoints = build_progress_checkpoints(
                static_cast<Count>(total_chunks), kGainInitChunkProgressPercentStep);
            std::cout << "[rtw-greedy] gain init sample chunk started: branch=non_empty"
                      << ", root=" << sample.root
                      << ", candidates=" << candidate_local_indices.size()
                      << ", chunk_size=" << chunk_size
                      << ", total_chunks=" << total_chunks
                      << ", total_elapsed_s=" << runtime_elapsed_seconds()
                      << std::endl;
        }
        for (std::size_t chunk_begin = 0; chunk_begin < total_inactive;
             chunk_begin += static_cast<std::size_t>(chunk_size)) {
            const std::size_t chunk_end =
                std::min<std::size_t>(chunk_begin + static_cast<std::size_t>(chunk_size),
                                      total_inactive);
            const std::size_t label_count = chunk_end - chunk_begin;
            std::vector<std::uint8_t> label_satisfied(label_count, 0U);
            std::unordered_set<std::uint64_t> seen_label_node;
            std::unordered_map<std::uint64_t, std::uint32_t> gate_label_count;
            std::queue<std::uint64_t> queue;

            const std::size_t estimated_states = static_cast<std::size_t>(work_per_label) * label_count;
            const std::size_t reserve_hint = std::min<std::size_t>(estimated_states, 1000000U);
            seen_label_node.reserve(reserve_hint);
            gate_label_count.reserve(std::min<std::size_t>(reserve_hint, 500000U));

            for (std::size_t label_local = 0; label_local < label_count; ++label_local) {
                const std::size_t sample_local_idx =
                    inactive_local_indices[chunk_begin + label_local];
                const std::uint64_t state_key = pack_key(
                    static_cast<std::uint32_t>(label_local),
                    static_cast<std::uint32_t>(sample_local_idx));
                seen_label_node.insert(state_key);
                queue.push(state_key);
            }

            while (!queue.empty()) {
                const std::uint64_t state_key = queue.front();
                queue.pop();
                const std::uint32_t label_local = static_cast<std::uint32_t>(state_key >> 32U);
                const std::uint32_t node_local = static_cast<std::uint32_t>(state_key);
                if (label_local >= label_count || node_local >= sample.formula_nodes.size()) {
                    continue;
                }
                if (label_satisfied[label_local] != 0U) {
                    continue;
                }
                if (node_local == root_idx) {
                    label_satisfied[label_local] = 1U;
                    continue;
                }
                if (state.node_true[node_local] != 0U) {
                    continue;
                }

                const NodeId node = sample.formula_nodes[node_local];
                for (NodeId dst : sample.out_dep_targets(node)) {
                    std::size_t dst_idx = 0U;
                    if (!sample.try_get_formula_index(dst, &dst_idx)) {
                        continue;
                    }
                    if (state.node_true[dst_idx] != 0U) {
                        continue;
                    }
                    if (dst_idx == root_idx) {
                        label_satisfied[label_local] = 1U;
                        break;
                    }
                    const std::uint64_t dst_key =
                        pack_key(label_local, static_cast<std::uint32_t>(dst_idx));
                    if (seen_label_node.insert(dst_key).second) {
                        queue.push(dst_key);
                    }
                }
                if (label_satisfied[label_local] != 0U) {
                    continue;
                }

                for (HyperedgeId hid : sample.incident_hyperedges(node)) {
                    if (hid >= sample.hyperedges.size()) {
                        throw std::logic_error("RTW hyperedge id out of range during batch propagation");
                    }
                    if (state.hyperedge_triggered[hid] != 0U) {
                        continue;
                    }

                    const auto& hyperedge = sample.hyperedges[hid];
                    const std::uint32_t triggered_mark = hyperedge.node_count + 1U;
                    const std::uint64_t gate_key =
                        pack_key(label_local, static_cast<std::uint32_t>(hid));

                    std::uint32_t count_delta = 0U;
                    auto it = gate_label_count.find(gate_key);
                    if (it != gate_label_count.end()) {
                        count_delta = it->second;
                        if (count_delta >= triggered_mark) {
                            continue;
                        }
                    }

                    if (count_delta < hyperedge.node_count) {
                        ++count_delta;
                    }
                    const std::uint64_t effective_count =
                        static_cast<std::uint64_t>(state.hyperedge_count[hid]) +
                        static_cast<std::uint64_t>(count_delta);
                    if (effective_count < static_cast<std::uint64_t>(hyperedge.threshold)) {
                        gate_label_count[gate_key] = count_delta;
                        continue;
                    }

                    gate_label_count[gate_key] = triggered_mark;
                    for (NodeId member : sample.hyperedge_nodes(hid)) {
                        std::size_t member_idx = 0U;
                        if (!sample.try_get_formula_index(member, &member_idx)) {
                            throw std::logic_error("RTW hyperedge node missing during batch propagation");
                        }
                        if (state.node_true[member_idx] != 0U) {
                            continue;
                        }
                        if (member_idx == root_idx) {
                            label_satisfied[label_local] = 1U;
                            break;
                        }
                        const std::uint64_t member_key =
                            pack_key(label_local, static_cast<std::uint32_t>(member_idx));
                        if (seen_label_node.insert(member_key).second) {
                            queue.push(member_key);
                        }
                    }
                    if (label_satisfied[label_local] != 0U) {
                        break;
                    }
                }
            }

            for (std::size_t label_local = 0; label_local < label_count; ++label_local) {
                if (label_satisfied[label_local] == 0U) {
                    continue;
                }
                const std::size_t sample_local_idx =
                    inactive_local_indices[chunk_begin + label_local];
                satisfying_candidates->push_back(sample.formula_nodes[sample_local_idx]);
            }

            if (!chunk_checkpoints.empty()) {
                ++completed_chunks;
                while (next_chunk_checkpoint < chunk_checkpoints.size() &&
                       completed_chunks >= chunk_checkpoints[next_chunk_checkpoint]) {
                    const Count checkpoint = chunk_checkpoints[next_chunk_checkpoint];
                    const double pct =
                        static_cast<double>(checkpoint) * 100.0 /
                        static_cast<double>(total_chunks);
                    std::cout << "[rtw-greedy] gain init sample chunk progress: branch=non_empty"
                              << ", root=" << sample.root
                              << ", progress=" << checkpoint
                              << "/" << total_chunks
                              << " (" << pct << "%)"
                              << ", total_elapsed_s=" << runtime_elapsed_seconds()
                              << std::endl;
                    ++next_chunk_checkpoint;
                }
            }
        }
        if (!chunk_checkpoints.empty()) {
            std::cout << "[rtw-greedy] gain init sample chunk done: branch=non_empty"
                      << ", root=" << sample.root
                      << ", total_chunks=" << total_chunks
                      << ", total_elapsed_s=" << runtime_elapsed_seconds()
                      << std::endl;
        }
        return static_cast<Count>(candidate_local_indices.size());
    }

    // Definite satisfying region: reverse closure from root via ordinary reverse dependencies
    // and threshold-1 hyperedges.
    std::vector<std::uint8_t> definite(sample.formula_nodes.size(), 0U);
    std::queue<std::size_t> reverse_queue;
    std::size_t root_idx = 0U;
    if (!sample.try_get_formula_index(sample.root, &root_idx)) {
        return 0U;
    }
    definite[root_idx] = 1U;
    reverse_queue.push(root_idx);
    while (!reverse_queue.empty()) {
        const std::size_t idx = reverse_queue.front();
        reverse_queue.pop();
        const NodeId node = sample.formula_nodes[idx];

        for (NodeId src : sample.in_dep_sources(node)) {
            std::size_t src_idx = 0U;
            if (!sample.try_get_formula_index(src, &src_idx)) {
                continue;
            }
            if (definite[src_idx] == 0U) {
                definite[src_idx] = 1U;
                reverse_queue.push(src_idx);
            }
        }

        for (HyperedgeId hid : sample.incident_hyperedges(node)) {
            if (hid >= sample.hyperedges.size()) {
                throw std::logic_error("RTW hyperedge id out of range during reverse closure");
            }
            const auto& hyperedge = sample.hyperedges[hid];
            if (hyperedge.threshold != 1U) {
                continue;
            }
            for (NodeId member : sample.hyperedge_nodes(hid)) {
                std::size_t member_idx = 0U;
                if (!sample.try_get_formula_index(member, &member_idx)) {
                    throw std::logic_error("RTW hyperedge node missing during reverse closure");
                }
                if (definite[member_idx] == 0U) {
                    definite[member_idx] = 1U;
                    reverse_queue.push(member_idx);
                }
            }
        }
    }

    std::vector<std::size_t> uncertain_local_indices;
    uncertain_local_indices.reserve(candidate_local_indices.size());
    for (std::size_t local_idx : candidate_local_indices) {
        const NodeId candidate = sample.formula_nodes[local_idx];
        if (definite[local_idx] != 0U) {
            satisfying_candidates->push_back(candidate);
        } else {
            uncertain_local_indices.push_back(local_idx);
        }
    }
    if (uncertain_local_indices.empty()) {
        return static_cast<Count>(candidate_local_indices.size());
    }

    if (sample.formula_nodes.size() > std::numeric_limits<std::uint32_t>::max() ||
        sample.hyperedges.size() > std::numeric_limits<std::uint32_t>::max() ||
        uncertain_local_indices.size() > std::numeric_limits<std::uint32_t>::max()) {
        // Conservative fallback for index packing limits.
        for (std::size_t local_idx : uncertain_local_indices) {
            const NodeId candidate = sample.formula_nodes[local_idx];
            bool would_be_satisfied = false;
            if (simulate_add_seed(sample, state, candidate, &would_be_satisfied, nullptr) &&
                would_be_satisfied) {
                satisfying_candidates->push_back(candidate);
            }
        }
        return static_cast<Count>(candidate_local_indices.size());
    }

    Count uncertain_nodes = 0U;
    for (std::uint8_t mark : definite) {
        if (mark == 0U) {
            ++uncertain_nodes;
        }
    }
    Count uncertain_gates = 0U;
    for (HyperedgeId hid = 0; hid < sample.hyperedges.size(); ++hid) {
        bool touches_uncertain = false;
        for (NodeId member : sample.hyperedge_nodes(hid)) {
            std::size_t member_idx = 0U;
            if (!sample.try_get_formula_index(member, &member_idx)) {
                throw std::logic_error("RTW hyperedge node missing during uncertain scan");
            }
            if (definite[member_idx] == 0U) {
                touches_uncertain = true;
                break;
            }
        }
        if (touches_uncertain) {
            ++uncertain_gates;
        }
    }

    const Count work_per_label = std::max<Count>(1U, uncertain_nodes + uncertain_gates);
    const long double usable_bytes =
        static_cast<long double>(kBatchScratchBudgetBytes) * kBatchBudgetUtilization;
    const Count state_budget = std::max<Count>(
        1U,
        static_cast<Count>(usable_bytes / static_cast<long double>(kBatchStateBytesEstimate)));
    Count chunk_size = state_budget / work_per_label;
    if (chunk_size == 0U) {
        chunk_size = 1U;
    }
    if (chunk_size > static_cast<Count>(uncertain_local_indices.size())) {
        chunk_size = static_cast<Count>(uncertain_local_indices.size());
    }

    if (chunk_size <= 1U && uncertain_local_indices.size() > 1U) {
        // For very large uncertain regions, avoid high-overhead batch structures.
        for (std::size_t local_idx : uncertain_local_indices) {
            const NodeId candidate = sample.formula_nodes[local_idx];
            bool would_be_satisfied = false;
            if (simulate_add_seed(sample, state, candidate, &would_be_satisfied, nullptr) &&
                would_be_satisfied) {
                satisfying_candidates->push_back(candidate);
            }
        }
        return static_cast<Count>(candidate_local_indices.size());
    }

    auto pack_key = [](std::uint32_t high, std::uint32_t low) -> std::uint64_t {
        return (static_cast<std::uint64_t>(high) << 32U) | static_cast<std::uint64_t>(low);
    };

    const std::size_t total_uncertain = uncertain_local_indices.size();
    const std::size_t total_chunks =
        (total_uncertain + static_cast<std::size_t>(chunk_size) - 1U) /
        static_cast<std::size_t>(chunk_size);
    Count completed_chunks = 0U;
    std::vector<Count> chunk_checkpoints;
    std::size_t next_chunk_checkpoint = 0U;
    if (log_chunk_diagnostics && total_chunks > 1U) {
        chunk_checkpoints = build_progress_checkpoints(
            static_cast<Count>(total_chunks), kGainInitChunkProgressPercentStep);
        std::cout << "[rtw-greedy] gain init sample chunk started: branch=empty"
                  << ", root=" << sample.root
                  << ", candidates=" << candidate_local_indices.size()
                  << ", chunk_size=" << chunk_size
                  << ", total_chunks=" << total_chunks
                  << ", total_elapsed_s=" << runtime_elapsed_seconds()
                  << std::endl;
    }
    for (std::size_t chunk_begin = 0; chunk_begin < total_uncertain;
         chunk_begin += static_cast<std::size_t>(chunk_size)) {
        const std::size_t chunk_end =
            std::min<std::size_t>(chunk_begin + static_cast<std::size_t>(chunk_size),
                                  total_uncertain);
        const std::size_t label_count = chunk_end - chunk_begin;
        std::vector<std::uint8_t> label_satisfied(label_count, 0U);
        std::unordered_set<std::uint64_t> seen_label_node;
        std::unordered_map<std::uint64_t, std::uint32_t> gate_label_count;
        std::queue<std::uint64_t> queue;

        const std::size_t estimated_states = static_cast<std::size_t>(work_per_label) * label_count;
        const std::size_t reserve_hint = std::min<std::size_t>(estimated_states, 1000000U);
        seen_label_node.reserve(reserve_hint);
        gate_label_count.reserve(std::min<std::size_t>(reserve_hint, 500000U));

        for (std::size_t label_local = 0; label_local < label_count; ++label_local) {
            const std::size_t sample_local_idx = uncertain_local_indices[chunk_begin + label_local];
            const std::uint64_t state_key = pack_key(
                static_cast<std::uint32_t>(label_local),
                static_cast<std::uint32_t>(sample_local_idx));
            seen_label_node.insert(state_key);
            queue.push(state_key);
        }

        while (!queue.empty()) {
            const std::uint64_t state_key = queue.front();
            queue.pop();
            const std::uint32_t label_local = static_cast<std::uint32_t>(state_key >> 32U);
            const std::uint32_t node_local = static_cast<std::uint32_t>(state_key);
            if (label_local >= label_count || node_local >= sample.formula_nodes.size()) {
                continue;
            }
            if (label_satisfied[label_local] != 0U) {
                continue;
            }

            if (definite[node_local] != 0U) {
                label_satisfied[label_local] = 1U;
                continue;
            }

            const NodeId node = sample.formula_nodes[node_local];
            for (NodeId dst : sample.out_dep_targets(node)) {
                std::size_t dst_idx = 0U;
                if (!sample.try_get_formula_index(dst, &dst_idx)) {
                    continue;
                }
                if (definite[dst_idx] != 0U) {
                    label_satisfied[label_local] = 1U;
                    break;
                }
                const std::uint64_t dst_key =
                    pack_key(label_local, static_cast<std::uint32_t>(dst_idx));
                if (seen_label_node.insert(dst_key).second) {
                    queue.push(dst_key);
                }
            }
            if (label_satisfied[label_local] != 0U) {
                continue;
            }

            for (HyperedgeId hid : sample.incident_hyperedges(node)) {
                if (hid >= sample.hyperedges.size()) {
                    throw std::logic_error("RTW hyperedge id out of range during batch propagation");
                }
                const auto& hyperedge = sample.hyperedges[hid];
                const std::uint32_t triggered_mark = hyperedge.node_count + 1U;
                const std::uint64_t gate_key =
                    pack_key(label_local, static_cast<std::uint32_t>(hid));

                std::uint32_t count = 0U;
                auto it = gate_label_count.find(gate_key);
                if (it != gate_label_count.end()) {
                    count = it->second;
                    if (count >= triggered_mark) {
                        continue;
                    }
                }

                if (count < hyperedge.node_count) {
                    ++count;
                }
                if (count < hyperedge.threshold) {
                    gate_label_count[gate_key] = count;
                    continue;
                }

                gate_label_count[gate_key] = triggered_mark;
                for (NodeId member : sample.hyperedge_nodes(hid)) {
                    std::size_t member_idx = 0U;
                    if (!sample.try_get_formula_index(member, &member_idx)) {
                        throw std::logic_error("RTW hyperedge node missing during batch propagation");
                    }
                    if (definite[member_idx] != 0U) {
                        label_satisfied[label_local] = 1U;
                        break;
                    }
                    const std::uint64_t member_key =
                        pack_key(label_local, static_cast<std::uint32_t>(member_idx));
                    if (seen_label_node.insert(member_key).second) {
                        queue.push(member_key);
                    }
                }
                if (label_satisfied[label_local] != 0U) {
                    break;
                }
            }
        }

        for (std::size_t label_local = 0; label_local < label_count; ++label_local) {
            if (label_satisfied[label_local] == 0U) {
                continue;
            }
            const std::size_t sample_local_idx = uncertain_local_indices[chunk_begin + label_local];
            satisfying_candidates->push_back(sample.formula_nodes[sample_local_idx]);
        }

        if (!chunk_checkpoints.empty()) {
            ++completed_chunks;
            while (next_chunk_checkpoint < chunk_checkpoints.size() &&
                   completed_chunks >= chunk_checkpoints[next_chunk_checkpoint]) {
                const Count checkpoint = chunk_checkpoints[next_chunk_checkpoint];
                const double pct =
                    static_cast<double>(checkpoint) * 100.0 /
                    static_cast<double>(total_chunks);
                std::cout << "[rtw-greedy] gain init sample chunk progress: branch=empty"
                          << ", root=" << sample.root
                          << ", progress=" << checkpoint
                          << "/" << total_chunks
                          << " (" << pct << "%)"
                          << ", total_elapsed_s=" << runtime_elapsed_seconds()
                          << std::endl;
                ++next_chunk_checkpoint;
            }
        }
    }

    if (!chunk_checkpoints.empty()) {
        std::cout << "[rtw-greedy] gain init sample chunk done: branch=empty"
                  << ", root=" << sample.root
                  << ", total_chunks=" << total_chunks
                  << ", total_elapsed_s=" << runtime_elapsed_seconds()
                  << std::endl;
    }

    return static_cast<Count>(candidate_local_indices.size());
}

Count RTWCollection::current_satisfied_count() const {
    require_state_initialized();
    return satisfied_count_;
}

double RTWCollection::current_empirical_influence(NodeId num_nodes) const {
    require_state_initialized();
    if (samples_.empty()) {
        return 0.0;
    }
    return static_cast<double>(num_nodes) *
           (static_cast<double>(satisfied_count_) / static_cast<double>(samples_.size()));
}

Count RTWCollection::marginal_gain(NodeId candidate) const {
    return compute_marginal_gain_for_candidate(candidate);
}

Count RTWCollection::cached_marginal_gain(NodeId candidate) const {
    require_gain_cache_initialized();
    if (candidate >= indexed_num_nodes_) {
        throw std::out_of_range("candidate out of occurrence index range");
    }
    return cached_marginal_gain_[candidate];
}

NodeId RTWCollection::best_candidate(bool use_progress_tiebreak) {
    require_gain_cache_initialized();
    discard_stale_candidate_heap_entries();
    if (candidate_heap_.empty()) {
        return kInvalidNodeId;
    }
    if (!use_progress_tiebreak) {
        return candidate_heap_.top().node;
    }

    const Count best_gain = candidate_heap_.top().gain;
    std::vector<CandidateHeapEntry> tied_entries;
    while (true) {
        discard_stale_candidate_heap_entries();
        if (candidate_heap_.empty() || candidate_heap_.top().gain != best_gain) {
            break;
        }
        tied_entries.push_back(candidate_heap_.top());
        candidate_heap_.pop();
    }

    NodeId best_node = kInvalidNodeId;
    Count best_progress = 0U;
    for (const auto& entry : tied_entries) {
        const Count progress = progress_score(entry.node);
        if (best_node == kInvalidNodeId || progress > best_progress ||
            (progress == best_progress && entry.node < best_node)) {
            best_node = entry.node;
            best_progress = progress;
        }
    }

    for (const auto& entry : tied_entries) {
        candidate_heap_.push(entry);
    }
    return best_node;
}

Count RTWCollection::compute_marginal_gain_for_candidate(NodeId candidate) const {
    require_occurrence_index();
    require_state_initialized();

    if (candidate >= indexed_num_nodes_) {
        throw std::out_of_range("candidate out of occurrence index range");
    }
    if (selected_seed_[candidate] != 0U) {
        return 0;
    }

    Count gain = 0U;
    for (const auto& occ : node_occurrences_[candidate]) {
        const std::size_t sid = static_cast<std::size_t>(occ.sample_id);
        const auto& state = states_[sid];
        if (state.sample_satisfied) {
            continue;
        }

        bool would_be_satisfied = false;
        if (simulate_add_seed(samples_[sid], state, candidate, &would_be_satisfied, nullptr) &&
            would_be_satisfied) {
            ++gain;
        }
    }

    return gain;
}

Count RTWCollection::progress_score(NodeId candidate) const {
    require_occurrence_index();
    require_state_initialized();

    if (candidate >= indexed_num_nodes_) {
        throw std::out_of_range("candidate out of occurrence index range");
    }
    if (selected_seed_[candidate] != 0U) {
        return 0;
    }
    if (cached_progress_score_.size() != indexed_num_nodes_ ||
        cached_progress_signature_.size() != indexed_num_nodes_ ||
        progress_cache_valid_.size() != indexed_num_nodes_) {
        cached_progress_score_.assign(indexed_num_nodes_, 0U);
        cached_progress_signature_.assign(indexed_num_nodes_, 0U);
        progress_cache_valid_.assign(indexed_num_nodes_, 0U);
    }
    const std::uint64_t current_signature =
        compute_progress_signature_for_candidate(candidate);
    if (progress_cache_valid_[candidate] == 0U ||
        cached_progress_signature_[candidate] != current_signature) {
        cached_progress_score_[candidate] = compute_progress_score_for_candidate(candidate);
        cached_progress_signature_[candidate] = current_signature;
        progress_cache_valid_[candidate] = 1U;
    }
    return cached_progress_score_[candidate];
}

std::uint64_t RTWCollection::compute_progress_signature_for_candidate(NodeId candidate) const {
    require_occurrence_index();
    require_state_initialized();

    if (candidate >= indexed_num_nodes_) {
        throw std::out_of_range("candidate out of occurrence index range");
    }
    if (sample_state_version_.size() != samples_.size()) {
        throw std::logic_error("sample_state_version_ is not initialized");
    }

    std::uint64_t signature = 0U;
    for (const auto& occ : node_occurrences_[candidate]) {
        const std::size_t sid = static_cast<std::size_t>(occ.sample_id);
        signature += static_cast<std::uint64_t>(sample_state_version_[sid]);
    }
    return signature;
}

Count RTWCollection::compute_progress_score_for_candidate(NodeId candidate) const {
    require_occurrence_index();
    require_state_initialized();

    if (candidate >= indexed_num_nodes_) {
        throw std::out_of_range("candidate out of occurrence index range");
    }
    if (selected_seed_[candidate] != 0U) {
        return 0;
    }

    Count score = 0;
    for (const auto& occ : node_occurrences_[candidate]) {
        const std::size_t sid = static_cast<std::size_t>(occ.sample_id);
        auto& state = states_[sid];
        if (state.sample_satisfied) {
            continue;
        }

        const Count before = current_deficit(samples_[sid], state);
        Count after = before;
        simulate_add_seed(samples_[sid], state, candidate, nullptr, &after);
        if (after < before) {
            score += (before - after);
        }
    }

    return score;
}

void RTWCollection::add_seed(NodeId seed) {
    require_occurrence_index();
    require_state_initialized();

    if (seed >= indexed_num_nodes_) {
        throw std::out_of_range("seed out of occurrence index range");
    }
    if (selected_seed_[seed] != 0U) {
        return;
    }
    selected_seed_[seed] = 1U;
    if (seed < candidate_version_.size()) {
        ++candidate_version_[seed];
    }
    invalidate_progress_cache(seed);
    if (!gain_cache_ready_ && !progress_cache_valid_.empty()) {
        std::fill(progress_cache_valid_.begin(), progress_cache_valid_.end(), 0U);
    }
    ++stats_.seed_additions;
    last_affected_samples_.clear();
    last_affected_gates_.clear();

    Count batch_recomputations = 0U;
    std::vector<NodeId> gain_delta_nodes;
    gain_delta_nodes.reserve(64U);
    std::unordered_map<NodeId, long long> gain_delta;
    gain_delta.reserve(128U);
    auto accumulate_gain_delta = [&](NodeId candidate, int delta) {
        if (selected_seed_[candidate] != 0U || delta == 0) {
            return;
        }
        auto [it, inserted] = gain_delta.emplace(candidate, delta);
        if (!inserted) {
            it->second += delta;
        } else {
            gain_delta_nodes.push_back(candidate);
        }
    };
    if (gain_cache_ready_ &&
        sample_satisfying_candidates_cache_.size() != samples_.size()) {
        throw std::logic_error("sample satisfying cache is not initialized");
    }
    if (sample_state_version_.size() != samples_.size()) {
        throw std::logic_error("sample state version is not initialized");
    }
    for (const auto& occ : node_occurrences_[seed]) {
        const std::size_t sid = static_cast<std::size_t>(occ.sample_id);
        const auto& sample = samples_[sid];
        auto& state = states_[sid];
        if (state.sample_satisfied) {
            continue;
        }

        std::size_t seed_idx = 0U;
        if (!sample.try_get_formula_index(seed, &seed_idx) || state.node_true[seed_idx] != 0U) {
            continue;
        }
        const std::vector<NodeId>* old_satisfying_candidates = nullptr;
        if (gain_cache_ready_) {
            old_satisfying_candidates = &sample_satisfying_candidates_cache_[sid];
        }

        bool state_changed = false;
        std::vector<NodeId> activated_nodes;
        std::vector<GateId> affected_gates;
        if (apply_add_seed(sample,
                           &state,
                           seed,
                           &activated_nodes,
                           &affected_gates,
                           &state_changed)) {
            ++satisfied_count_;
        }
        if (!state_changed) {
            continue;
        }
        ++sample_state_version_[sid];

        ++stats_.affected_samples;
        stats_.affected_gates += static_cast<Count>(affected_gates.size());
        stats_.activated_nodes += static_cast<Count>(activated_nodes.size());
        last_affected_samples_.push_back(static_cast<SampleId>(sid));
        for (GateId gid : affected_gates) {
            last_affected_gates_.push_back(AffectedGate{static_cast<SampleId>(sid), gid});
        }

        if (gain_cache_ready_) {
            if (state.sample_satisfied) {
                ++stats_.satisfied_sample_dirty_expansions;
            } else {
                ++stats_.fallback_sample_dirty_expansions;
            }

            for (NodeId candidate : *old_satisfying_candidates) {
                accumulate_gain_delta(candidate, -1);
            }
            if (!state.sample_satisfied) {
                const std::vector<NodeId> prioritized_candidates =
                    build_local_candidate_priority_set(sample, activated_nodes, affected_gates);
                std::vector<NodeId> new_satisfying_candidates;
                batch_recomputations += compute_singleton_satisfying_candidates(
                    sample, state, &new_satisfying_candidates, &prioritized_candidates);
                for (NodeId candidate : new_satisfying_candidates) {
                    accumulate_gain_delta(candidate, 1);
                }
                sample_satisfying_candidates_cache_[sid] = std::move(new_satisfying_candidates);
            } else {
                sample_satisfying_candidates_cache_[sid].clear();
            }
        }
    }

    if (gain_cache_ready_) {
        cached_marginal_gain_[seed] = 0U;
        stats_.dirty_candidates += static_cast<Count>(gain_delta_nodes.size());
        stats_.max_dirty_candidates_per_seed =
            std::max(stats_.max_dirty_candidates_per_seed, static_cast<Count>(gain_delta_nodes.size()));
        stats_.gain_cache_update_recomputations += batch_recomputations;
        const auto t0 = std::chrono::steady_clock::now();
        for (NodeId candidate : gain_delta_nodes) {
            auto it = gain_delta.find(candidate);
            if (it == gain_delta.end() || it->second == 0) {
                continue;
            }
            const long long old_gain = static_cast<long long>(cached_marginal_gain_[candidate]);
            const long long new_gain = old_gain + it->second;
            if (new_gain < 0) {
                throw std::logic_error("negative cached marginal gain during batch dirty update");
            }
            if (candidate < candidate_version_.size()) {
                ++candidate_version_[candidate];
            }
            cached_marginal_gain_[candidate] = static_cast<Count>(new_gain);
            push_candidate_heap_entry(candidate);
        }
        const auto t1 = std::chrono::steady_clock::now();
        stats_.gain_cache_update_seconds += std::chrono::duration<double>(t1 - t0).count();
    }
}

const std::vector<RTWSample>& RTWCollection::samples() const {
    return samples_;
}

const std::vector<SampleId>& RTWCollection::last_affected_samples() const {
    return last_affected_samples_;
}

const std::vector<RTWCollection::AffectedGate>& RTWCollection::last_affected_gates() const {
    return last_affected_gates_;
}

const RTWCollectionStats& RTWCollection::stats() const {
    return stats_;
}

Count RTWCollection::evaluate_seed_set_with_index(const std::vector<NodeId>& seeds) const {
    require_occurrence_index();

    std::vector<SampleState> local_states;
    local_states.reserve(samples_.size());
    for (const auto& sample : samples_) {
        SampleState state;
        state.node_true.assign(sample.formula_nodes.size(), 0U);
        state.hyperedge_count.assign(sample.hyperedges.size(), 0U);
        state.hyperedge_triggered.assign(sample.hyperedges.size(), 0U);
        state.sample_satisfied = false;
        local_states.push_back(std::move(state));
    }

    std::vector<std::uint8_t> local_selected(indexed_num_nodes_, 0U);
    Count satisfied = 0U;

    for (NodeId seed : seeds) {
        if (seed >= indexed_num_nodes_) {
            throw std::out_of_range("seed out of occurrence index range for evaluation");
        }
        if (local_selected[seed] != 0U) {
            continue;
        }
        local_selected[seed] = 1U;

        for (const auto& occ : node_occurrences_[seed]) {
            const std::size_t sid = static_cast<std::size_t>(occ.sample_id);
            auto& state = local_states[sid];
            if (state.sample_satisfied) {
                continue;
            }
            bool state_changed = false;
            if (apply_add_seed(samples_[sid], &state, seed, nullptr, nullptr, &state_changed)) {
                ++satisfied;
            }
        }
    }

    return satisfied;
}

Count RTWCollection::current_deficit(const RTWSample& sample, const SampleState& state) const {
    Count deficit = 0;
    for (std::size_t i = 0; i < sample.hyperedges.size(); ++i) {
        const auto& hyperedge = sample.hyperedges[i];
        const Count s = static_cast<Count>(hyperedge.node_count);
        if (s == 0U) {
            continue;
        }
        const Count c = static_cast<Count>(state.hyperedge_count[i]);
        if (c >= s || c >= static_cast<Count>(hyperedge.threshold)) {
            continue;
        }
        deficit += (s - c) * (static_cast<Count>(hyperedge.threshold) - c);
    }
    return deficit;
}

bool RTWCollection::simulate_add_seed(const RTWSample& sample,
                                      const SampleState& base,
                                      NodeId seed,
                                      bool* would_be_satisfied,
                                      Count* deficit_after) const {
    if (would_be_satisfied != nullptr) {
        *would_be_satisfied = false;
    }

    std::size_t seed_idx = 0;
    if (!sample.try_get_formula_index(seed, &seed_idx)) {
        if (deficit_after != nullptr) {
            *deficit_after = current_deficit(sample, base);
        }
        return false;
    }

    if (base.node_true[seed_idx] != 0U) {
        if (deficit_after != nullptr) {
            *deficit_after = current_deficit(sample, base);
        }
        return false;
    }

    std::vector<std::uint8_t> node_true = base.node_true;
    std::vector<std::uint32_t> hyperedge_count = base.hyperedge_count;
    std::vector<std::uint8_t> hyperedge_triggered = base.hyperedge_triggered;

    std::queue<std::size_t> queue;
    node_true[seed_idx] = 1U;
    queue.push(seed_idx);

    while (!queue.empty()) {
        const std::size_t idx = queue.front();
        queue.pop();
        const NodeId node = sample.formula_nodes[idx];

        for (NodeId dst : sample.out_dep_targets(node)) {
            std::size_t dst_idx = 0;
            if (!sample.try_get_formula_index(dst, &dst_idx)) {
                continue;
            }
            if (node_true[dst_idx] == 0U) {
                node_true[dst_idx] = 1U;
                queue.push(dst_idx);
            }
        }

        for (HyperedgeId hid : sample.incident_hyperedges(node)) {
            if (hid >= sample.hyperedges.size()) {
                throw std::logic_error("RTW hyperedge id out of range during simulation");
            }
            if (hyperedge_triggered[hid] != 0U) {
                continue;
            }

            const auto& hyperedge = sample.hyperedges[hid];
            if (hyperedge_count[hid] < hyperedge.node_count) {
                ++hyperedge_count[hid];
            }
            if (hyperedge_count[hid] < hyperedge.threshold) {
                continue;
            }

            hyperedge_triggered[hid] = 1U;
            for (NodeId target : sample.hyperedge_nodes(hid)) {
                std::size_t target_idx = 0;
                if (!sample.try_get_formula_index(target, &target_idx)) {
                    throw std::logic_error("RTW hyperedge node missing during simulation");
                }
                if (node_true[target_idx] == 0U) {
                    node_true[target_idx] = 1U;
                    queue.push(target_idx);
                }
            }
        }
    }

    std::size_t root_idx = 0;
    bool root_true = false;
    if (sample.try_get_formula_index(sample.root, &root_idx)) {
        root_true = (node_true[root_idx] != 0U);
    }

    if (would_be_satisfied != nullptr) {
        *would_be_satisfied = (!base.sample_satisfied && root_true);
    }

    if (deficit_after != nullptr) {
        Count deficit = 0;
        for (std::size_t i = 0; i < sample.hyperedges.size(); ++i) {
            const auto& hyperedge = sample.hyperedges[i];
            const Count s = static_cast<Count>(hyperedge.node_count);
            if (s == 0U) {
                continue;
            }
            const Count c = static_cast<Count>(hyperedge_count[i]);
            if (c >= s || c >= static_cast<Count>(hyperedge.threshold)) {
                continue;
            }
            deficit += (s - c) * (static_cast<Count>(hyperedge.threshold) - c);
        }
        *deficit_after = deficit;
    }

    return true;
}

bool RTWCollection::apply_add_seed(const RTWSample& sample,
                                   SampleState* state,
                                   NodeId seed,
                                   std::vector<NodeId>* activated_nodes,
                                   std::vector<GateId>* affected_gates,
                                   bool* state_changed) const {
    if (state == nullptr) {
        throw std::invalid_argument("state pointer cannot be null");
    }
    if (state_changed != nullptr) {
        *state_changed = false;
    }
    if (affected_gates != nullptr) {
        affected_gates->clear();
    }
    if (activated_nodes != nullptr) {
        activated_nodes->clear();
    }

    std::size_t seed_idx = 0;
    if (!sample.try_get_formula_index(seed, &seed_idx)) {
        return false;
    }
    if (state->node_true[seed_idx] != 0U) {
        return false;
    }

    std::vector<std::uint8_t> hyperedge_touched;
    if (affected_gates != nullptr) {
        hyperedge_touched.assign(sample.hyperedges.size(), 0U);
    }

    std::queue<std::size_t> queue;
    state->node_true[seed_idx] = 1U;
    queue.push(seed_idx);
    if (activated_nodes != nullptr) {
        activated_nodes->push_back(seed);
    }
    if (state_changed != nullptr) {
        *state_changed = true;
    }

    while (!queue.empty()) {
        const std::size_t idx = queue.front();
        queue.pop();
        const NodeId node = sample.formula_nodes[idx];

        for (NodeId dst : sample.out_dep_targets(node)) {
            std::size_t dst_idx = 0;
            if (!sample.try_get_formula_index(dst, &dst_idx)) {
                continue;
            }
            if (state->node_true[dst_idx] == 0U) {
                state->node_true[dst_idx] = 1U;
                queue.push(dst_idx);
                if (activated_nodes != nullptr) {
                    activated_nodes->push_back(dst);
                }
            }
        }

        for (HyperedgeId hid : sample.incident_hyperedges(node)) {
            if (hid >= sample.hyperedges.size()) {
                throw std::logic_error("RTW hyperedge id out of range during apply");
            }
            if (state->hyperedge_triggered[hid] != 0U) {
                continue;
            }

            const auto& hyperedge = sample.hyperedges[hid];
            if (state->hyperedge_count[hid] < hyperedge.node_count) {
                ++state->hyperedge_count[hid];
                if (affected_gates != nullptr && hyperedge_touched[hid] == 0U) {
                    hyperedge_touched[hid] = 1U;
                    affected_gates->push_back(hid);
                }
            }
            if (state->hyperedge_count[hid] < hyperedge.threshold) {
                continue;
            }

            state->hyperedge_triggered[hid] = 1U;
            for (NodeId target : sample.hyperedge_nodes(hid)) {
                std::size_t target_idx = 0;
                if (!sample.try_get_formula_index(target, &target_idx)) {
                    throw std::logic_error("RTW hyperedge node missing during apply");
                }
                if (state->node_true[target_idx] == 0U) {
                    state->node_true[target_idx] = 1U;
                    queue.push(target_idx);
                    if (activated_nodes != nullptr) {
                        activated_nodes->push_back(target);
                    }
                    if (state_changed != nullptr) {
                        *state_changed = true;
                    }
                }
            }
        }
    }

    if (state->sample_satisfied) {
        return false;
    }

    std::size_t root_idx = 0;
    if (!sample.try_get_formula_index(sample.root, &root_idx)) {
        return false;
    }

    if (state->node_true[root_idx] != 0U) {
        state->sample_satisfied = true;
        return true;
    }

    return false;
}

void RTWCollection::require_occurrence_index() const {
    if (!occurrence_ready_) {
        throw std::logic_error("build_global_occurrence_index() must be called first");
    }
}

void RTWCollection::require_state_initialized() const {
    if (!state_ready_) {
        throw std::logic_error("initialize_incremental_state() must be called first");
    }
}

void RTWCollection::require_gain_cache_initialized() const {
    require_occurrence_index();
    require_state_initialized();
    if (!gain_cache_ready_) {
        throw std::logic_error("initialize_gain_cache() must be called first");
    }
}

void RTWCollection::clear_candidate_heap() {
    candidate_heap_ = std::priority_queue<CandidateHeapEntry,
                                          std::vector<CandidateHeapEntry>,
                                          CandidateHeapCompare>{};
}

bool RTWCollection::is_candidate_heap_entry_current(const CandidateHeapEntry& entry) const {
    if (entry.node == kInvalidNodeId || entry.node >= indexed_num_nodes_) {
        return false;
    }
    if (selected_seed_[entry.node] != 0U) {
        return false;
    }
    if (entry.node >= candidate_version_.size() ||
        entry.version != candidate_version_[entry.node]) {
        return false;
    }
    if (entry.node >= cached_marginal_gain_.size() ||
        entry.gain != cached_marginal_gain_[entry.node]) {
        return false;
    }
    return true;
}

void RTWCollection::discard_stale_candidate_heap_entries() {
    while (!candidate_heap_.empty() && !is_candidate_heap_entry_current(candidate_heap_.top())) {
        candidate_heap_.pop();
    }
}

void RTWCollection::push_candidate_heap_entry(NodeId candidate) {
    if (candidate >= indexed_num_nodes_ || selected_seed_[candidate] != 0U) {
        return;
    }
    if (candidate >= candidate_version_.size() || candidate >= cached_marginal_gain_.size()) {
        return;
    }
    candidate_heap_.push(CandidateHeapEntry{
        cached_marginal_gain_[candidate],
        candidate,
        candidate_version_[candidate],
    });
}

void RTWCollection::invalidate_progress_cache(NodeId candidate) const {
    if (candidate < progress_cache_valid_.size()) {
        progress_cache_valid_[candidate] = 0U;
    }
}

void RTWCollection::reset_runtime_stats() {
    const Count sample_count = stats_.sample_count;
    const Count occurrence_index_entries = stats_.occurrence_index_entries;
    stats_ = RTWCollectionStats{};
    stats_.sample_count = occurrence_ready_ ? sample_count : static_cast<Count>(samples_.size());
    stats_.occurrence_index_entries = occurrence_index_entries;
}

}  // namespace htc
