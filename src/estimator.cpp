#include "htc/estimator.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "htc/cascade.hpp"
#include "htc/ic.hpp"
#include "htc/random.hpp"
#include "htc/runtime_clock.hpp"
#include "htc/rtw.hpp"

namespace htc {
namespace {

inline constexpr Count kSamplingEarlyStopCheckInterval = 2048U;
inline constexpr Count kIEProgressStepPercent = 5U;
inline constexpr Count kIEMCProgressStepPercent = 5U;

Count progress_satisfied_threshold(Count target_satisfied, Count percent) {
    if (target_satisfied == 0U) {
        return 0U;
    }
    const long double raw = static_cast<long double>(target_satisfied) *
                            static_cast<long double>(percent) / 100.0L;
    const Count threshold = static_cast<Count>(std::ceil(raw));
    return std::max<Count>(1U, threshold);
}

void log_rtw_est_progress(Count satisfied,
                          Count target_satisfied,
                          Count total_samples,
                          Count percent_checkpoint,
                          const std::chrono::steady_clock::time_point& t0) {
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    double eta = 0.0;
    if (satisfied > 0U && satisfied < target_satisfied) {
        eta = elapsed * static_cast<double>(target_satisfied - satisfied) /
              static_cast<double>(satisfied);
    }
    const double hit_rate = (total_samples == 0U)
                                ? 0.0
                                : static_cast<double>(satisfied) / static_cast<double>(total_samples);
    const double sample_rate =
        (elapsed <= 0.0) ? 0.0 : static_cast<double>(total_samples) / elapsed;

    std::cout << "[ie:rtw-est] phase=sampling progress=" << satisfied << "/" << target_satisfied
              << " (" << percent_checkpoint << "%)"
              << ", total_samples=" << total_samples
              << ", hit_rate=" << hit_rate
              << ", elapsed_s=" << elapsed
              << ", eta_s=" << eta
              << ", sample_rate=" << sample_rate << "/s"
              << ", total_elapsed_s=" << runtime_elapsed_seconds() << std::endl;
}

void log_mc_htc_progress(Count completed,
                         Count total_simulations,
                         double spread_sum,
                         Count percent_checkpoint,
                         const std::chrono::steady_clock::time_point& t0) {
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    double eta = 0.0;
    if (completed > 0U && completed < total_simulations) {
        eta = elapsed * static_cast<double>(total_simulations - completed) /
              static_cast<double>(completed);
    }
    const double sim_rate =
        (elapsed <= 0.0) ? 0.0 : static_cast<double>(completed) / elapsed;
    const double mean_spread =
        (completed == 0U) ? 0.0 : (spread_sum / static_cast<double>(completed));
    std::cout << "[ie:mc-htc] phase=simulation progress=" << completed << "/"
              << total_simulations
              << " (" << percent_checkpoint << "%)"
              << ", avg_spread_so_far=" << mean_spread
              << ", elapsed_s=" << elapsed
              << ", eta_s=" << eta
              << ", sim_rate=" << sim_rate << "/s"
              << ", total_elapsed_s=" << runtime_elapsed_seconds() << std::endl;
}

void log_ur_ie_progress(Count covered,
                        Count target_covered,
                        Count total_samples,
                        Count percent_checkpoint,
                        const std::chrono::steady_clock::time_point& t0) {
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    double eta = 0.0;
    if (covered > 0U && covered < target_covered) {
        eta = elapsed * static_cast<double>(target_covered - covered) /
              static_cast<double>(covered);
    }
    const double hit_rate = (total_samples == 0U)
                                ? 0.0
                                : static_cast<double>(covered) / static_cast<double>(total_samples);
    const double sample_rate =
        (elapsed <= 0.0) ? 0.0 : static_cast<double>(total_samples) / elapsed;

    std::cout << "[ie:ur-ie] phase=sampling progress=" << covered << "/" << target_covered
              << " (" << percent_checkpoint << "%)"
              << ", total_samples=" << total_samples
              << ", hit_rate=" << hit_rate
              << ", elapsed_s=" << elapsed
              << ", eta_s=" << eta
              << ", sample_rate=" << sample_rate << "/s"
              << ", total_elapsed_s=" << runtime_elapsed_seconds() << std::endl;
}

void validate_seed_set(const HybridHypergraph& graph, const std::vector<NodeId>& seeds) {
    std::unordered_set<NodeId> seen;
    for (NodeId v : seeds) {
        if (v >= graph.num_nodes()) {
            throw std::out_of_range("seed out of graph range");
        }
        if (!seen.insert(v).second) {
            throw std::invalid_argument("duplicate seed detected");
        }
    }
}

class RTWLightweightIEEngine {
public:
    RTWLightweightIEEngine(const HybridHypergraph& graph, const std::vector<std::uint8_t>& in_seed)
        : graph_(graph),
          in_seed_(in_seed),
          local_index_(graph.num_nodes(), kInvalidNodeId),
          expanded_(graph.num_nodes(), 0U),
          enqueued_(graph.num_nodes(), 0U),
          hyperedge_included_(graph.num_hyperedges(), 0U),
          hyperedge_true_count_(graph.num_hyperedges(), 0U),
          hyperedge_triggered_(graph.num_hyperedges(), 0U) {}

    bool sample_and_check(Random& random, NodeId root) {
        reset();
        if (sample_witness(random, root)) {
            return true;
        }
        return evaluate_root_satisfaction(root);
    }

private:
    void reset() {
        for (NodeId v : sample_nodes_) {
            local_index_[v] = kInvalidNodeId;
            expanded_[v] = 0U;
            enqueued_[v] = 0U;
        }
        sample_nodes_.clear();
        out_dep_local_.clear();
        node_true_.clear();
        sample_contains_seed_ = false;

        for (HyperedgeId hid : included_hyperedges_) {
            hyperedge_included_[hid] = 0U;
            hyperedge_true_count_[hid] = 0U;
            hyperedge_triggered_[hid] = 0U;
        }
        included_hyperedges_.clear();
    }

    NodeId ensure_sample_node(NodeId v) {
        if (v >= graph_.num_nodes()) {
            throw std::out_of_range("sample node out of graph range");
        }
        if (local_index_[v] != kInvalidNodeId) {
            return local_index_[v];
        }
        if (sample_nodes_.size() >= static_cast<std::size_t>(kInvalidNodeId)) {
            throw std::overflow_error("too many sample nodes in RTW lightweight IE");
        }
        const NodeId local = static_cast<NodeId>(sample_nodes_.size());
        sample_nodes_.push_back(v);
        local_index_[v] = local;
        out_dep_local_.emplace_back();
        if (in_seed_[v] != 0U) {
            sample_contains_seed_ = true;
        }
        return local;
    }

    bool include_hyperedge(HyperedgeId hid) {
        if (hid >= graph_.num_hyperedges()) {
            throw std::out_of_range("hyperedge id out of range");
        }
        if (hyperedge_included_[hid] != 0U) {
            return false;
        }
        hyperedge_included_[hid] = 1U;
        included_hyperedges_.push_back(hid);
        return true;
    }

    void enqueue_if_needed(NodeId v, std::queue<NodeId>* queue) {
        if (queue == nullptr) {
            throw std::invalid_argument("queue cannot be null");
        }
        if (expanded_[v] != 0U || enqueued_[v] != 0U) {
            return;
        }
        queue->push(v);
        enqueued_[v] = 1U;
    }

    bool sample_witness(Random& random, NodeId root) {
        ensure_sample_node(root);
        if (in_seed_[root] != 0U) {
            return true;
        }

        std::queue<NodeId> queue;
        enqueue_if_needed(root, &queue);
        Count expanded_count = 0U;

        while (!queue.empty()) {
            const NodeId v = queue.front();
            queue.pop();
            enqueued_[v] = 0U;
            if (expanded_[v] != 0U) {
                continue;
            }
            expanded_[v] = 1U;
            const NodeId v_local = local_index_[v];
            if (v_local == kInvalidNodeId) {
                throw std::logic_error("expanded node is missing local index");
            }

            for (EdgeId eid : graph_.in_edges(v)) {
                const auto& e = graph_.edge(eid);
                if (!random.bernoulli(e.prob)) {
                    continue;
                }
                const NodeId src_local = ensure_sample_node(e.src);
                out_dep_local_[src_local].push_back(v_local);
                enqueue_if_needed(e.src, &queue);
            }

            for (HyperedgeId hid : graph_.incident_hyperedges(v)) {
                if (!include_hyperedge(hid)) {
                    continue;
                }
                const auto& h = graph_.hyperedge(hid);
                for (NodeId u : h.nodes) {
                    if (u == v) {
                        continue;
                    }
                    ensure_sample_node(u);
                    enqueue_if_needed(u, &queue);
                }
            }

            ++expanded_count;
            if (sample_contains_seed_ && kSamplingEarlyStopCheckInterval != 0U &&
                (expanded_count % kSamplingEarlyStopCheckInterval) == 0U) {
                if (evaluate_root_satisfaction(root)) {
                    return true;
                }
            }
        }

        return false;
    }

    bool evaluate_root_satisfaction(NodeId root) {
        const NodeId root_local = local_index_[root];
        if (root_local == kInvalidNodeId) {
            throw std::logic_error("RTW sample root missing after sampling");
        }

        for (HyperedgeId hid : included_hyperedges_) {
            hyperedge_true_count_[hid] = 0U;
            hyperedge_triggered_[hid] = 0U;
        }

        node_true_.assign(sample_nodes_.size(), 0U);
        std::queue<NodeId> queue;
        for (NodeId local = 0; local < static_cast<NodeId>(sample_nodes_.size()); ++local) {
            const NodeId v = sample_nodes_[local];
            if (in_seed_[v] == 0U) {
                continue;
            }
            node_true_[local] = 1U;
            if (local == root_local) {
                return true;
            }
            queue.push(local);
        }

        while (!queue.empty()) {
            const NodeId local = queue.front();
            queue.pop();
            const NodeId u = sample_nodes_[local];

            for (NodeId dst_local : out_dep_local_[local]) {
                if (node_true_[dst_local] != 0U) {
                    continue;
                }
                node_true_[dst_local] = 1U;
                if (dst_local == root_local) {
                    return true;
                }
                queue.push(dst_local);
            }

            for (HyperedgeId hid : graph_.incident_hyperedges(u)) {
                if (hyperedge_included_[hid] == 0U || hyperedge_triggered_[hid] != 0U) {
                    continue;
                }

                const auto& h = graph_.hyperedge(hid);
                if (hyperedge_true_count_[hid] < h.nodes.size()) {
                    ++hyperedge_true_count_[hid];
                }
                if (hyperedge_true_count_[hid] < h.threshold) {
                    continue;
                }

                hyperedge_triggered_[hid] = 1U;
                for (NodeId v : h.nodes) {
                    const NodeId target_local = local_index_[v];
                    if (target_local == kInvalidNodeId || node_true_[target_local] != 0U) {
                        continue;
                    }
                    node_true_[target_local] = 1U;
                    if (target_local == root_local) {
                        return true;
                    }
                    queue.push(target_local);
                }
            }
        }

        return false;
    }

    const HybridHypergraph& graph_;
    const std::vector<std::uint8_t>& in_seed_;

    std::vector<NodeId> local_index_;
    std::vector<std::uint8_t> expanded_;
    std::vector<std::uint8_t> enqueued_;
    std::vector<NodeId> sample_nodes_;
    std::vector<std::vector<NodeId>> out_dep_local_;
    std::vector<std::uint8_t> node_true_;

    std::vector<std::uint8_t> hyperedge_included_;
    std::vector<std::uint32_t> hyperedge_true_count_;
    std::vector<std::uint8_t> hyperedge_triggered_;
    std::vector<HyperedgeId> included_hyperedges_;
    bool sample_contains_seed_ = false;
};

class URIEarlyStopHitEngine {
public:
    URIEarlyStopHitEngine(const ICGraph& graph,
                          const std::vector<std::uint8_t>& in_seed_users,
                          NodeId user_node_count)
        : graph_(graph),
          in_seed_users_(in_seed_users),
          user_node_count_(user_node_count),
          visited_stamp_(graph.num_nodes(), 0U) {
        if (in_seed_users_.size() != static_cast<std::size_t>(user_node_count_)) {
            throw std::invalid_argument("URIEarlyStopHitEngine seed mask size mismatch");
        }
        queue_.reserve(std::min<std::size_t>(graph.num_nodes(), 65536U));
    }

    bool sample_hit_with_root(NodeId root, Random& random) {
        if (root >= user_node_count_) {
            throw std::out_of_range("UR-IE root must be in user-node range");
        }
        advance_stamp();
        queue_.clear();
        mark_visited(root);
        if (in_seed_users_[root] != 0U) {
            return true;
        }
        queue_.push_back(root);

        std::size_t head = 0U;
        while (head < queue_.size()) {
            const NodeId v = queue_[head++];
            for (EdgeId eid : graph_.in_edges(v)) {
                const auto& e = graph_.edge(eid);
                const NodeId src = e.src;
                if (is_visited(src)) {
                    continue;
                }
                if (!random.bernoulli(e.prob)) {
                    continue;
                }
                mark_visited(src);
                if (src < user_node_count_ && in_seed_users_[src] != 0U) {
                    return true;
                }
                queue_.push_back(src);
            }
        }
        return false;
    }

private:
    bool is_visited(NodeId v) const { return visited_stamp_[v] == stamp_; }

    void mark_visited(NodeId v) { visited_stamp_[v] = stamp_; }

    void advance_stamp() {
        if (stamp_ == std::numeric_limits<std::uint32_t>::max()) {
            std::fill(visited_stamp_.begin(), visited_stamp_.end(), 0U);
            stamp_ = 1U;
            return;
        }
        ++stamp_;
        if (stamp_ == 0U) {
            // Defensive guard for unlikely wrap-around corner case.
            std::fill(visited_stamp_.begin(), visited_stamp_.end(), 0U);
            stamp_ = 1U;
        }
    }

    const ICGraph& graph_;
    const std::vector<std::uint8_t>& in_seed_users_;
    NodeId user_node_count_ = 0U;
    std::vector<std::uint32_t> visited_stamp_;
    std::uint32_t stamp_ = 1U;
    std::vector<NodeId> queue_;
};

}  // namespace

Count compute_forward_mc_samples(NodeId n, Count seed_set_size, double epsilon, double delta) {
    if (n == 0U) {
        throw std::invalid_argument("node count must be > 0 for MC sample sizing");
    }
    if (seed_set_size == 0U) {
        throw std::invalid_argument("seed_set_size must be > 0 for MC sample sizing");
    }
    if (epsilon <= 0.0 || epsilon >= 1.0) {
        throw std::invalid_argument("epsilon must be in (0,1)");
    }
    if (delta <= 0.0 || delta >= 1.0) {
        throw std::invalid_argument("delta must be in (0,1)");
    }

    const long double numerator = 3.0L * static_cast<long double>(n) *
                                  std::log(2.0L / static_cast<long double>(delta));
    const long double eps = static_cast<long double>(epsilon);
    const long double denominator = eps * eps * static_cast<long double>(seed_set_size);
    const long double raw = numerator / denominator;
    if (!std::isfinite(raw)) {
        throw std::overflow_error("computed MC sample count is not finite");
    }
    const long double ceiling = std::ceil(raw);
    const long double max_count = static_cast<long double>(std::numeric_limits<Count>::max());
    if (ceiling > max_count) {
        throw std::overflow_error("computed MC sample count overflows Count");
    }

    const Count samples = static_cast<Count>(ceiling);
    return (samples == 0U) ? 1U : samples;
}

RTWEstimator::RTWEstimator(const HybridHypergraph& graph) : graph_(graph) {}

IEResult RTWEstimator::estimate_adaptive(const std::vector<NodeId>& seeds,
                                         const IEConfig& config) const {
    validate_seed_set(graph_, seeds);
    if (config.epsilon <= 0.0 || config.epsilon >= 1.0) {
        throw std::invalid_argument("epsilon must be in (0,1)");
    }
    if (config.delta <= 0.0 || config.delta >= 1.0) {
        throw std::invalid_argument("delta must be in (0,1)");
    }

    const double eps = config.epsilon;
    const double delta = config.delta;
    const double raw_lambda = 2.0 * (1.0 + eps) * (1.0 + eps / 3.0) *
                              std::log(2.0 / delta) / (eps * eps);
    const Count lambda = static_cast<Count>(std::ceil(raw_lambda));

    const auto t0 = std::chrono::steady_clock::now();
    std::cout << "[ie:rtw-est] phase=sampling started"
              << ", target_satisfied=" << lambda
              << ", progress_step=" << kIEProgressStepPercent << "%"
              << ", total_elapsed_s=" << runtime_elapsed_seconds() << std::endl;

    Random random(config.seed);
    std::vector<std::uint8_t> in_seed(graph_.num_nodes(), 0U);
    for (NodeId v : seeds) {
        in_seed[v] = 1U;
    }
    RTWLightweightIEEngine engine(graph_, in_seed);

    Count satisfied = 0;
    Count total = 0;
    Count next_progress_percent = kIEProgressStepPercent;
    Count next_progress_threshold = progress_satisfied_threshold(lambda, next_progress_percent);
    while (satisfied < lambda) {
        const NodeId root = random.uniform_node(0U, graph_.num_nodes() - 1U);
        ++total;
        if (engine.sample_and_check(random, root)) {
            ++satisfied;
            while (next_progress_percent <= 100U && satisfied >= next_progress_threshold) {
                log_rtw_est_progress(satisfied, lambda, total, next_progress_percent, t0);
                next_progress_percent += kIEProgressStepPercent;
                if (next_progress_percent > 100U) {
                    break;
                }
                next_progress_threshold =
                    progress_satisfied_threshold(lambda, next_progress_percent);
            }
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed_seconds = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "[ie:rtw-est] phase=sampling done"
              << ", satisfied=" << satisfied << "/" << lambda
              << ", total_samples=" << total
              << ", elapsed_s=" << elapsed_seconds
              << ", total_elapsed_s=" << runtime_elapsed_seconds() << std::endl;

    IEResult result;
    result.method = "rtw-est";
    result.samples = total;
    result.satisfied = satisfied;
    result.estimate = static_cast<double>(graph_.num_nodes()) *
                      (static_cast<double>(satisfied) / static_cast<double>(total));
    result.seconds = elapsed_seconds;
    return result;
}

IEResult RTWEstimator::estimate_fixed(const std::vector<NodeId>& seeds,
                                      Count theta,
                                      std::uint64_t seed) const {
    validate_seed_set(graph_, seeds);
    if (theta == 0U) {
        throw std::invalid_argument("theta must be > 0 for fixed RTW estimation");
    }

    const auto t0 = std::chrono::steady_clock::now();

    Random random(seed);
    std::vector<std::uint8_t> in_seed(graph_.num_nodes(), 0U);
    for (NodeId v : seeds) {
        in_seed[v] = 1U;
    }
    RTWLightweightIEEngine engine(graph_, in_seed);

    Count satisfied = 0;
    for (Count i = 0; i < theta; ++i) {
        const NodeId root = random.uniform_node(0U, graph_.num_nodes() - 1U);
        if (engine.sample_and_check(random, root)) {
            ++satisfied;
        }
    }

    const auto t1 = std::chrono::steady_clock::now();

    IEResult result;
    result.method = "rtw-fixed";
    result.samples = theta;
    result.satisfied = satisfied;
    result.estimate = static_cast<double>(graph_.num_nodes()) *
                      (static_cast<double>(satisfied) / static_cast<double>(theta));
    result.seconds = std::chrono::duration<double>(t1 - t0).count();
    return result;
}

MCEstimator::MCEstimator(const HybridHypergraph& graph) : graph_(graph) {}

IEResult MCEstimator::estimate(const std::vector<NodeId>& seeds,
                               Count simulations,
                               std::uint64_t seed) const {
    validate_seed_set(graph_, seeds);
    if (simulations == 0U) {
        throw std::invalid_argument("simulations must be > 0 for MC estimator");
    }

    const auto t0 = std::chrono::steady_clock::now();
    std::cout << "[ie:mc-htc] phase=simulation started"
              << ", simulations=" << simulations
              << ", progress_step=" << kIEMCProgressStepPercent << "%"
              << ", total_elapsed_s=" << runtime_elapsed_seconds() << std::endl;

    Random random(seed);
    double total = 0.0;
    Count next_progress_percent = kIEMCProgressStepPercent;
    Count next_progress_threshold =
        progress_satisfied_threshold(simulations, next_progress_percent);
    for (Count i = 0; i < simulations; ++i) {
        total += static_cast<double>(
            HTCCascadeSimulator::simulate_spread(graph_, seeds, &random, "MC estimator"));
        const Count completed = i + 1U;
        while (next_progress_percent <= 100U && completed >= next_progress_threshold) {
            log_mc_htc_progress(
                completed, simulations, total, next_progress_percent, t0);
            next_progress_percent += kIEMCProgressStepPercent;
            if (next_progress_percent > 100U) {
                break;
            }
            next_progress_threshold =
                progress_satisfied_threshold(simulations, next_progress_percent);
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed_seconds = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "[ie:mc-htc] phase=simulation done"
              << ", simulations=" << simulations
              << ", avg_spread=" << (total / static_cast<double>(simulations))
              << ", elapsed_s=" << elapsed_seconds
              << ", total_elapsed_s=" << runtime_elapsed_seconds() << std::endl;

    IEResult result;
    result.method = "mc-htc";
    result.samples = simulations;
    result.satisfied = 0;
    result.estimate = total / static_cast<double>(simulations);
    result.seconds = elapsed_seconds;
    return result;
}

URIEstimator::URIEstimator(const HybridHypergraph& graph) : graph_(graph) {}

IEResult URIEstimator::estimate_adaptive(const std::vector<NodeId>& seeds,
                                         const IEConfig& config) const {
    validate_seed_set(graph_, seeds);
    if (config.epsilon <= 0.0 || config.epsilon >= 1.0) {
        throw std::invalid_argument("epsilon must be in (0,1)");
    }
    if (config.delta <= 0.0 || config.delta >= 1.0) {
        throw std::invalid_argument("delta must be in (0,1)");
    }

    const double eps = config.epsilon;
    const double delta = config.delta;
    const double raw_lambda = 2.0 * (1.0 + eps) * (1.0 + eps / 3.0) *
                              std::log(2.0 / delta) / (eps * eps);
    const Count lambda = static_cast<Count>(std::ceil(raw_lambda));

    const auto t0 = std::chrono::steady_clock::now();
    std::cout << "[ie:ur-ie] phase=sampling started"
              << ", target_covered=" << lambda
              << ", progress_step=" << kIEProgressStepPercent << "%"
              << ", total_elapsed_s=" << runtime_elapsed_seconds() << std::endl;

    const ICGraph g_plus = build_upper_relaxed_aux_ic_graph(graph_);
    Random random(config.seed);

    std::vector<std::uint8_t> in_seed(graph_.num_nodes(), 0U);
    for (NodeId v : seeds) {
        in_seed[v] = 1U;
    }
    URIEarlyStopHitEngine hit_engine(g_plus, in_seed, graph_.num_nodes());

    Count covered = 0;
    Count total_samples = 0U;
    Count next_progress_percent = kIEProgressStepPercent;
    Count next_progress_threshold = progress_satisfied_threshold(lambda, next_progress_percent);
    while (covered < lambda) {
        const NodeId root = random.uniform_node(0U, graph_.num_nodes() - 1U);
        ++total_samples;
        const bool hit = hit_engine.sample_hit_with_root(root, random);
        if (hit) {
            ++covered;
            while (next_progress_percent <= 100U && covered >= next_progress_threshold) {
                log_ur_ie_progress(
                    covered, lambda, total_samples, next_progress_percent, t0);
                next_progress_percent += kIEProgressStepPercent;
                if (next_progress_percent > 100U) {
                    break;
                }
                next_progress_threshold =
                    progress_satisfied_threshold(lambda, next_progress_percent);
            }
        }
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed_seconds = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "[ie:ur-ie] phase=sampling done"
              << ", covered=" << covered << "/" << lambda
              << ", total_samples=" << total_samples
              << ", elapsed_s=" << elapsed_seconds
              << ", total_elapsed_s=" << runtime_elapsed_seconds() << std::endl;

    IEResult result;
    result.method = "ur-ie";
    result.samples = total_samples;
    result.satisfied = covered;
    result.estimate = static_cast<double>(graph_.num_nodes()) *
                      (static_cast<double>(covered) / static_cast<double>(total_samples));
    result.seconds = elapsed_seconds;
    return result;
}

}  // namespace htc
