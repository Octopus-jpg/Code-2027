#pragma once

#include <cstddef>
#include <cstdint>
#include <queue>
#include <unordered_map>
#include <vector>

#include "htc/graph.hpp"
#include "htc/random.hpp"

namespace htc {

struct RTWDependency {
    NodeId src = kInvalidNodeId;
    NodeId dst = kInvalidNodeId;
};

template <typename T>
struct RTWRange {
    const T* ptr = nullptr;
    std::size_t len = 0;

    const T* begin() const {
        return ptr;
    }

    const T* end() const {
        return ptr == nullptr ? nullptr : (ptr + len);
    }

    bool empty() const {
        return len == 0U;
    }

    std::size_t size() const {
        return len;
    }
};

struct RTWGate {
    GateId id = kInvalidGateId;
    HyperedgeId hyperedge_id = kInvalidHyperedgeId;
    NodeId target = kInvalidNodeId;
    std::uint32_t threshold = 1;
    std::uint32_t input_offset = 0;
    std::uint32_t input_count = 0;
};

struct RTWHyperedge {
    HyperedgeId original_id = kInvalidHyperedgeId;
    std::uint32_t threshold = 1;
    std::uint32_t node_offset = 0;
    std::uint32_t node_count = 0;
};

class RTWSample {
public:
    NodeId root = kInvalidNodeId;
    std::vector<NodeId> formula_nodes;
    std::vector<RTWDependency> deps;
    std::vector<RTWHyperedge> hyperedges;

    void add_formula_node(NodeId v);
    void add_dependency(NodeId src, NodeId dst);
    void add_hyperedge(HyperedgeId h, std::uint32_t threshold, const std::vector<NodeId>& nodes);
    bool contains_formula(NodeId v) const;
    bool try_get_formula_index(NodeId v, std::size_t* index) const;
    Count gate_count() const;
    Count gate_input_incidence_count() const;
    Count witness_size() const;
    void build_indices();

    RTWRange<NodeId> out_dep_targets(NodeId v) const;
    RTWRange<NodeId> in_dep_sources(NodeId v) const;
    RTWRange<HyperedgeId> incident_hyperedges(NodeId v) const;
    RTWRange<NodeId> hyperedge_nodes(HyperedgeId hid) const;

private:
    static RTWRange<NodeId> node_range_from_csr(const std::vector<std::uint32_t>& offsets,
                                                const std::vector<NodeId>& values,
                                                std::size_t idx);
    static RTWRange<HyperedgeId> hyperedge_range_from_csr(const std::vector<std::uint32_t>& offsets,
                                                          const std::vector<HyperedgeId>& values,
                                                          std::size_t idx);

    std::unordered_map<NodeId, std::size_t> formula_pos_;
    std::vector<NodeId> hyperedge_nodes_flat_;
    std::vector<std::uint32_t> out_dep_offsets_;
    std::vector<NodeId> out_dep_values_;
    std::vector<std::uint32_t> in_dep_offsets_;
    std::vector<NodeId> in_dep_values_;
    std::vector<std::uint32_t> incident_hyperedge_offsets_;
    std::vector<HyperedgeId> incident_hyperedge_values_;
};

struct RTWStats {
    Count witness_count = 0;
    Count total_formula_nodes = 0;
    Count total_dependencies = 0;
    Count total_gates = 0;
    Count total_gate_input_incidences = 0;
    Count total_witness_size = 0;

    void add_sample(const RTWSample& sample);
    double avg_formula_nodes() const;
    double avg_dependencies() const;
    double avg_gates() const;
    double avg_gate_input_incidences() const;
    double avg_witness_size() const;
};

struct RTWCollectionStats {
    Count sample_count = 0;
    Count occurrence_index_entries = 0;
    Count gain_cache_initial_recomputations = 0;
    Count gain_cache_update_recomputations = 0;
    Count seed_additions = 0;
    Count affected_samples = 0;
    Count affected_gates = 0;
    Count activated_nodes = 0;
    Count dirty_candidates = 0;
    Count max_dirty_candidates_per_seed = 0;
    Count satisfied_sample_dirty_expansions = 0;
    Count fallback_sample_dirty_expansions = 0;
    double gain_cache_initialization_seconds = 0.0;
    double gain_cache_update_seconds = 0.0;

    double avg_dirty_candidates_per_seed() const;
    double avg_affected_samples_per_seed() const;
};

class RTWSampler {
public:
    explicit RTWSampler(const HybridHypergraph& graph);

    RTWSample sample(Random& random) const;
    RTWSample sample_with_root(NodeId root, Random& random) const;

private:
    const HybridHypergraph& graph_;
};

class RTWEvaluator {
public:
    bool is_satisfied(const RTWSample& sample, const std::vector<NodeId>& seeds) const;
    Count total_gate_deficit(const RTWSample& sample, const std::vector<NodeId>& seeds) const;

private:
    struct EvalState {
        std::vector<std::uint8_t> node_true;
        std::vector<std::uint32_t> hyperedge_count;
        std::vector<std::uint8_t> hyperedge_triggered;
    };

    EvalState run_closure(const RTWSample& sample, const std::vector<NodeId>& seeds) const;
};

class RTWCollection {
public:
    struct AffectedGate {
        SampleId sample_id = kInvalidSampleId;
        GateId gate_id = kInvalidGateId;
    };

    void add_sample(RTWSample sample);
    Count size() const;

    void build_global_occurrence_index(NodeId num_nodes);
    void initialize_incremental_state();
    void initialize_gain_cache();

    Count current_satisfied_count() const;
    double current_empirical_influence(NodeId num_nodes) const;

    Count cached_marginal_gain(NodeId candidate) const;
    Count marginal_gain(NodeId candidate) const;
    Count progress_score(NodeId candidate) const;
    NodeId best_candidate(bool use_progress_tiebreak);
    void add_seed(NodeId seed);
    Count evaluate_seed_set_with_index(const std::vector<NodeId>& seeds) const;

    const std::vector<RTWSample>& samples() const;
    const std::vector<SampleId>& last_affected_samples() const;
    const std::vector<AffectedGate>& last_affected_gates() const;
    const RTWCollectionStats& stats() const;

private:
    struct Occurrence {
        SampleId sample_id = kInvalidSampleId;
    };

    struct SampleState {
        std::vector<std::uint8_t> node_true;
        std::vector<std::uint32_t> hyperedge_count;
        std::vector<std::uint8_t> hyperedge_triggered;
        bool sample_satisfied = false;
    };

    struct CandidateHeapEntry {
        Count gain = 0;
        NodeId node = kInvalidNodeId;
        Count version = 0;
    };

    struct CandidateHeapCompare {
        bool operator()(const CandidateHeapEntry& a, const CandidateHeapEntry& b) const {
            if (a.gain != b.gain) {
                return a.gain < b.gain;
            }
            return a.node > b.node;
        }
    };

    Count compute_marginal_gain_for_candidate(NodeId candidate) const;
    Count compute_progress_score_for_candidate(NodeId candidate) const;
    std::uint64_t compute_progress_signature_for_candidate(NodeId candidate) const;
    Count compute_singleton_satisfying_candidates(const RTWSample& sample,
                                                  const SampleState& state,
                                                  std::vector<NodeId>* satisfying_candidates,
                                                  const std::vector<NodeId>* prioritized_candidates) const;
    Count current_deficit(const RTWSample& sample, const SampleState& state) const;
    bool simulate_add_seed(const RTWSample& sample,
                           const SampleState& base,
                           NodeId seed,
                           bool* would_be_satisfied,
                           Count* deficit_after) const;
    bool apply_add_seed(const RTWSample& sample,
                        SampleState* state,
                        NodeId seed,
                        std::vector<NodeId>* activated_nodes,
                        std::vector<GateId>* affected_gates,
                        bool* state_changed) const;

    void require_occurrence_index() const;
    void require_state_initialized() const;
    void require_gain_cache_initialized() const;
    void reset_runtime_stats();
    void clear_candidate_heap();
    void discard_stale_candidate_heap_entries();
    bool is_candidate_heap_entry_current(const CandidateHeapEntry& entry) const;
    void push_candidate_heap_entry(NodeId candidate);
    void invalidate_progress_cache(NodeId candidate) const;

    std::vector<std::vector<Occurrence>> node_occurrences_;
    std::vector<RTWSample> samples_;
    std::vector<SampleState> states_;
    std::vector<Count> sample_state_version_;
    std::vector<std::vector<NodeId>> sample_satisfying_candidates_cache_;
    std::vector<Count> cached_marginal_gain_;
    mutable std::vector<Count> cached_progress_score_;
    mutable std::vector<std::uint64_t> cached_progress_signature_;
    mutable std::vector<std::uint8_t> progress_cache_valid_;
    std::vector<SampleId> last_affected_samples_;
    std::vector<AffectedGate> last_affected_gates_;

    std::vector<std::uint8_t> selected_seed_;
    std::vector<Count> candidate_version_;
    std::priority_queue<CandidateHeapEntry,
                        std::vector<CandidateHeapEntry>,
                        CandidateHeapCompare>
        candidate_heap_;
    RTWCollectionStats stats_;
    NodeId indexed_num_nodes_ = 0;
    Count satisfied_count_ = 0;
    bool occurrence_ready_ = false;
    bool state_ready_ = false;
    bool gain_cache_ready_ = false;
};

}  // namespace htc
