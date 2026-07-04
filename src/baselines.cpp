#include "htc/baselines.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

#include "htc/cascade.hpp"
#include "htc/random.hpp"
#include "htc/runtime_clock.hpp"

namespace htc {
namespace {

using SteadyClock = std::chrono::steady_clock;
constexpr Count kProgressPercentStep = 10U;
constexpr Count kFinalEvalProgressPercentStep = 1U;

double elapsed_seconds(SteadyClock::time_point begin, SteadyClock::time_point end) {
    return std::chrono::duration<double>(end - begin).count();
}

Count current_peak_rss_kb() {
#if defined(__unix__) || defined(__APPLE__)
    struct rusage usage {};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return 0U;
    }
    if (usage.ru_maxrss <= 0) {
        return 0U;
    }
#if defined(__APPLE__)
    return static_cast<Count>(usage.ru_maxrss / 1024);
#else
    return static_cast<Count>(usage.ru_maxrss);
#endif
#else
    return 0U;
#endif
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

void log_phase_start(const std::string& method,
                     const std::string& phase,
                     const std::string& details = std::string()) {
    std::cout << "[baseline:" << method << "] phase=" << phase << " start";
    if (!details.empty()) {
        std::cout << ", " << details;
    }
    std::cout << ", total_elapsed_s=" << runtime_elapsed_seconds();
    std::cout << std::endl;
}

void log_phase_end(const std::string& method,
                   const std::string& phase,
                   SteadyClock::time_point begin,
                   const std::string& details = std::string()) {
    const double elapsed = elapsed_seconds(begin, SteadyClock::now());
    std::cout << "[baseline:" << method << "] phase=" << phase
              << " done, elapsed_s=" << elapsed;
    if (!details.empty()) {
        std::cout << ", " << details;
    }
    std::cout << ", total_elapsed_s=" << runtime_elapsed_seconds();
    std::cout << std::endl;
}

void log_progress_snapshot(const std::string& method,
                           const std::string& phase,
                           Count completed,
                           Count total,
                           SteadyClock::time_point begin,
                           const std::string& details = std::string()) {
    const double elapsed = elapsed_seconds(begin, SteadyClock::now());
    const double rate = (elapsed > 1e-12) ? (static_cast<double>(completed) / elapsed) : 0.0;
    const double remaining = static_cast<double>(total - completed);
    const double eta = (rate > 1e-12) ? (remaining / rate) : 0.0;
    const double pct = (total > 0U)
                           ? (static_cast<double>(completed) * 100.0 /
                              static_cast<double>(total))
                           : 100.0;
    std::cout << "[baseline:" << method << "] phase=" << phase
              << " progress=" << completed << "/" << total
              << " (" << pct << "%)"
              << ", elapsed_s=" << elapsed
              << ", eta_s=" << eta
              << ", rate=" << rate << "/s"
              << ", total_elapsed_s=" << runtime_elapsed_seconds();
    if (!details.empty()) {
        std::cout << ", " << details;
    }
    std::cout << std::endl;
}

class ProgressTracker {
public:
    ProgressTracker(std::string method, std::string phase, Count total, Count percent_step)
        : method_(std::move(method)),
          phase_(std::move(phase)),
          total_(total),
          checkpoints_(build_progress_checkpoints(total, percent_step)),
          begin_(SteadyClock::now()) {}

    void maybe_log(Count completed, const std::string& details = std::string()) {
        while (next_checkpoint_idx_ < checkpoints_.size() &&
               completed >= checkpoints_[next_checkpoint_idx_]) {
            const Count done = checkpoints_[next_checkpoint_idx_];
            log_progress_snapshot(method_, phase_, done, total_, begin_, details);
            ++next_checkpoint_idx_;
        }
    }

private:
    std::string method_;
    std::string phase_;
    Count total_ = 0U;
    std::vector<Count> checkpoints_;
    std::size_t next_checkpoint_idx_ = 0U;
    SteadyClock::time_point begin_;
};

struct InfluenceEvalOptions {
    const char* simulation_tag = "HTCMCGreedy";
    std::string method_tag;
    std::string phase_tag;
    Count progress_percent = 0U;
};

double estimate_htc_influence(const HybridHypergraph& graph,
                              const std::vector<NodeId>& seeds,
                              Count simulations,
                              std::uint64_t seed,
                              const InfluenceEvalOptions& options = InfluenceEvalOptions()) {
    if (simulations == 0U) {
        throw std::invalid_argument("simulations must be > 0");
    }

    Random random(seed);
    ProgressTracker tracker(options.method_tag,
                            options.phase_tag,
                            simulations,
                            options.progress_percent);
    double total = 0.0;
    for (Count i = 0; i < simulations; ++i) {
        total += static_cast<double>(
            HTCCascadeSimulator::simulate_spread(graph, seeds, &random, options.simulation_tag));
        if (!options.method_tag.empty() && !options.phase_tag.empty() &&
            options.progress_percent > 0U) {
            tracker.maybe_log(i + 1U);
        }
    }
    return total / static_cast<double>(simulations);
}

IMResult evaluate_seed_set(const HybridHypergraph& graph,
                           const std::string& method_tag,
                           const std::vector<NodeId>& seeds,
                           Count simulations,
                            std::uint64_t seed) {
    IMResult result;
    result.seeds = seeds;
    result.eval_samples = simulations;
    const auto t_eval = SteadyClock::now();
    log_phase_start(method_tag,
                    "final_eval",
                    "mc_simulations=" + std::to_string(simulations));
    InfluenceEvalOptions options;
    options.simulation_tag = "Baseline final MC";
    options.method_tag = method_tag;
    options.phase_tag = "final_eval";
    options.progress_percent = kFinalEvalProgressPercentStep;
    result.empirical_influence = estimate_htc_influence(graph, seeds, simulations, seed, options);
    log_phase_end(method_tag,
                  "final_eval",
                  t_eval,
                  "empirical_influence=" + std::to_string(result.empirical_influence));
    return result;
}

std::vector<NodeId> top_k_by_score(const std::vector<double>& score, Budget k) {
    std::vector<NodeId> nodes(score.size());
    for (NodeId i = 0; i < nodes.size(); ++i) {
        nodes[i] = i;
    }

    std::sort(nodes.begin(), nodes.end(), [&](NodeId a, NodeId b) {
        if (score[a] != score[b]) {
            return score[a] > score[b];
        }
        return a < b;
    });

    if (k < nodes.size()) {
        nodes.resize(k);
    }
    return nodes;
}

Budget clipped_budget(Budget k, NodeId n) {
    return (k > n) ? static_cast<Budget>(n) : k;
}

void ensure_seed_budget(const HybridHypergraph& graph, Budget requested_k, std::vector<NodeId>* seeds) {
    if (seeds == nullptr) {
        throw std::invalid_argument("seeds cannot be null");
    }

    const NodeId n = graph.num_nodes();
    const Budget target_k = clipped_budget(requested_k, n);
    if (target_k == 0U) {
        seeds->clear();
        return;
    }

    std::vector<std::uint8_t> selected(n, 0U);
    std::vector<NodeId> normalized;
    normalized.reserve(target_k);
    for (NodeId v : *seeds) {
        if (v >= n) {
            throw std::invalid_argument("seed out of graph range");
        }
        if (selected[v] != 0U) {
            continue;
        }
        selected[v] = 1U;
        normalized.push_back(v);
        if (normalized.size() >= target_k) {
            break;
        }
    }

    if (normalized.size() < target_k) {
        std::vector<double> fallback_score(n, 0.0);
        for (NodeId v = 0; v < n; ++v) {
            fallback_score[v] = static_cast<double>(graph.out_edges(v).size()) +
                                static_cast<double>(graph.incident_hyperedges(v).size());
        }
        const std::vector<NodeId> ranked = top_k_by_score(fallback_score, static_cast<Budget>(n));
        for (NodeId v : ranked) {
            if (selected[v] != 0U) {
                continue;
            }
            selected[v] = 1U;
            normalized.push_back(v);
            if (normalized.size() >= target_k) {
                break;
            }
        }
    }

    if (normalized.size() != target_k) {
        throw std::logic_error("failed to fill seeds to target budget");
    }
    *seeds = std::move(normalized);
}

enum class HCITMOrder : std::uint8_t {
    HCI1 = 1U,
    HCI2 = 2U,
};

std::vector<NodeId> run_hci_tm_selector(const HybridHypergraph& graph,
                                        Budget k,
                                        HCITMOrder order,
                                        const std::string& method_tag) {
    const NodeId n = graph.num_nodes();
    const HyperedgeId m = graph.num_hyperedges();
    const Budget steps = (k > n) ? n : k;
    if (steps == 0U) {
        return {};
    }

    std::vector<std::uint8_t> active_node(n, 0U);
    std::vector<std::uint8_t> is_seed(n, 0U);
    std::vector<std::uint8_t> hyper_active(m, 0U);
    std::vector<std::uint32_t> hyper_active_count(m, 0U);
    std::vector<Count> score_cache(n, 0U);
    std::vector<std::uint8_t> dirty_node(n, 0U);
    std::vector<std::uint8_t> dirty_hyperedge(m, 0U);
    std::queue<NodeId> active_queue;
    std::vector<NodeId> seeds;
    std::vector<NodeId> dirty_nodes;
    std::vector<HyperedgeId> dirty_hyperedges;
    std::vector<NodeId> newly_activated_nodes;
    seeds.reserve(steps);

    const auto activate_node = [&](NodeId node) {
        if (active_node[node] != 0U) {
            return;
        }
        active_node[node] = 1U;
        newly_activated_nodes.push_back(node);
        active_queue.push(node);
        for (HyperedgeId hid : graph.incident_hyperedges(node)) {
            ++hyper_active_count[hid];
        }
    };

    const auto propagate_from_seed = [&](NodeId seed) {
        newly_activated_nodes.clear();
        activate_node(seed);
        while (!active_queue.empty()) {
            const NodeId u = active_queue.front();
            active_queue.pop();
            for (HyperedgeId hid : graph.incident_hyperedges(u)) {
                if (hyper_active[hid] != 0U) {
                    continue;
                }
                const auto& h = graph.hyperedge(hid);
                if (hyper_active_count[hid] < h.threshold) {
                    continue;
                }
                hyper_active[hid] = 1U;
                for (NodeId v : h.nodes) {
                    activate_node(v);
                }
            }
        }
    };

    const auto compute_score = [&](NodeId node) -> Count {
        Count score = static_cast<Count>(graph.incident_hyperedges(node).size());
        for (HyperedgeId hid : graph.incident_hyperedges(node)) {
            if (hyper_active[hid] != 0U) {
                continue;
            }
            const auto& h = graph.hyperedge(hid);
            if (hyper_active_count[hid] + 1U != h.threshold) {
                continue;
            }
            for (NodeId other : h.nodes) {
                if (other == node || active_node[other] != 0U) {
                    continue;
                }
                ++score;
                if (order == HCITMOrder::HCI2 && is_seed[other] == 0U) {
                    const Count other_degree = static_cast<Count>(graph.incident_hyperedges(other).size());
                    if (other_degree > 0U) {
                        score += (other_degree - 1U);
                    }
                }
            }
        }
        return score;
    };

    const auto mark_dirty = [&](NodeId node) {
        if (active_node[node] != 0U || dirty_node[node] != 0U) {
            return;
        }
        dirty_node[node] = 1U;
        dirty_nodes.push_back(node);
    };

    const auto recompute_dirty_scores = [&]() {
        for (NodeId node : dirty_nodes) {
            score_cache[node] = compute_score(node);
            dirty_node[node] = 0U;
        }
        dirty_nodes.clear();
    };

    for (NodeId node = 0; node < n; ++node) {
        mark_dirty(node);
    }
    recompute_dirty_scores();
    const auto t_selection = SteadyClock::now();

    for (Budget step = 0; step < steps; ++step) {
        NodeId best = kInvalidNodeId;
        Count best_score = 0U;
        bool found = false;
        for (NodeId node = 0; node < n; ++node) {
            if (active_node[node] != 0U) {
                continue;
            }
            const Count score = score_cache[node];
            if (!found || score > best_score || (score == best_score && node < best)) {
                found = true;
                best = node;
                best_score = score;
            }
        }
        if (!found) {
            for (NodeId node = 0; node < n; ++node) {
                if (is_seed[node] == 0U) {
                    best = node;
                    found = true;
                    break;
                }
            }
        }
        if (!found) {
            break;
        }
        seeds.push_back(best);
        is_seed[best] = 1U;
        propagate_from_seed(best);

        for (NodeId active : newly_activated_nodes) {
            for (HyperedgeId hid : graph.incident_hyperedges(active)) {
                if (dirty_hyperedge[hid] != 0U) {
                    continue;
                }
                dirty_hyperedge[hid] = 1U;
                dirty_hyperedges.push_back(hid);
            }
        }

        for (HyperedgeId hid : dirty_hyperedges) {
            const auto& h = graph.hyperedge(hid);
            for (NodeId node : h.nodes) {
                mark_dirty(node);
            }
            dirty_hyperedge[hid] = 0U;
        }
        dirty_hyperedges.clear();

        const Count dirty_recompute_count = static_cast<Count>(dirty_nodes.size());
        recompute_dirty_scores();

        const Count selected = static_cast<Count>(seeds.size());
        const double elapsed = elapsed_seconds(t_selection, SteadyClock::now());
        const double rate = (elapsed > 1e-12) ? (static_cast<double>(selected) / elapsed) : 0.0;
        const double remaining = static_cast<double>(steps - selected);
        const double eta = (rate > 1e-12) ? (remaining / rate) : 0.0;
        std::cout << "[baseline:" << method_tag << "] phase=seed_select progress="
                  << selected << "/" << steps
                  << ", latest_seed=" << best
                  << ", best_score=" << best_score
                  << ", newly_activated=" << newly_activated_nodes.size()
                  << ", dirty_recompute=" << dirty_recompute_count
                  << ", elapsed_s=" << elapsed
                  << ", eta_s=" << eta
                  << ", total_elapsed_s=" << runtime_elapsed_seconds()
                  << std::endl;
    }

    return seeds;
}

struct MOEAIndividual {
    std::vector<NodeId> seeds;
    double fitness = 0.0;
};

constexpr std::uint64_t kMOEASeedSalt = 0x9E3779B97F4A7C15ULL;

std::uint64_t mix_seed(std::uint64_t x) {
    x ^= (x >> 30);
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= (x >> 27);
    x *= 0x94D049BB133111EBULL;
    x ^= (x >> 31);
    return x;
}

bool better_individual(const MOEAIndividual& lhs, const MOEAIndividual& rhs) {
    if (std::abs(lhs.fitness - rhs.fitness) > 1e-12) {
        return lhs.fitness > rhs.fitness;
    }
    return lhs.seeds < rhs.seeds;
}

void sort_unique_seeds(std::vector<NodeId>* seeds) {
    std::sort(seeds->begin(), seeds->end());
    seeds->erase(std::unique(seeds->begin(), seeds->end()), seeds->end());
}

void fill_missing_seeds(std::vector<NodeId>* seeds,
                        NodeId n,
                        Budget target_k,
                        Random* random) {
    if (target_k == 0U || n == 0U) {
        seeds->clear();
        return;
    }

    sort_unique_seeds(seeds);
    if (seeds->size() > target_k) {
        seeds->resize(target_k);
    }
    if (seeds->size() == target_k) {
        return;
    }

    std::vector<std::uint8_t> selected(n, 0U);
    for (NodeId node : *seeds) {
        selected[node] = 1U;
    }
    while (seeds->size() < target_k) {
        NodeId candidate = random->uniform_node(0U, n - 1U);
        while (selected[candidate] != 0U) {
            candidate = random->uniform_node(0U, n - 1U);
        }
        selected[candidate] = 1U;
        seeds->push_back(candidate);
    }
    sort_unique_seeds(seeds);
}

std::vector<NodeId> random_seed_set(NodeId n, Budget k, Random* random) {
    std::vector<NodeId> seeds;
    seeds.reserve(k);
    fill_missing_seeds(&seeds, n, k, random);
    return seeds;
}

NodeId sample_weighted_index(const std::vector<double>& weights,
                             Random* random) {
    double total = 0.0;
    for (double w : weights) {
        total += (w > 0.0) ? w : 0.0;
    }
    if (total <= 0.0) {
        return random->uniform_node(0U, static_cast<NodeId>(weights.size() - 1U));
    }
    const double r = random->uniform_real() * total;
    double prefix = 0.0;
    for (std::size_t i = 0; i < weights.size(); ++i) {
        const double w = (weights[i] > 0.0) ? weights[i] : 0.0;
        prefix += w;
        if (r <= prefix) {
            return static_cast<NodeId>(i);
        }
    }
    return static_cast<NodeId>(weights.size() - 1U);
}

std::vector<NodeId> weighted_seed_set(const std::vector<NodeId>& node_pool,
                                      const std::vector<double>& node_weight,
                                      NodeId n,
                                      Budget k,
                                      Random* random) {
    if (node_pool.empty()) {
        return random_seed_set(n, k, random);
    }

    std::vector<NodeId> pool = node_pool;
    std::vector<double> weights = node_weight;
    std::vector<NodeId> seeds;
    seeds.reserve(k);

    while (!pool.empty() && seeds.size() < k) {
        const NodeId picked_idx = sample_weighted_index(weights, random);
        seeds.push_back(pool[picked_idx]);
        const std::size_t idx = static_cast<std::size_t>(picked_idx);
        pool[idx] = pool.back();
        pool.pop_back();
        weights[idx] = weights.back();
        weights.pop_back();
    }
    fill_missing_seeds(&seeds, n, k, random);
    return seeds;
}

double evaluate_moea_fitness(const HybridHypergraph& graph,
                             const std::vector<NodeId>& seeds,
                             Count fitness_simulations,
                             std::uint64_t base_seed,
                             Count eval_counter) {
    const std::uint64_t mixed_seed =
        mix_seed(base_seed ^ (eval_counter + 1U) * kMOEASeedSalt);
    return estimate_htc_influence(graph, seeds, fitness_simulations, mixed_seed);
}

std::size_t tournament_pick_index(const std::vector<MOEAIndividual>& population,
                                  Count tournament_size,
                                  Random* random) {
    if (population.empty()) {
        throw std::invalid_argument("population must not be empty");
    }
    const Count sample_size = std::min<Count>(tournament_size, population.size());
    std::size_t best = static_cast<std::size_t>(
        random->uniform_count(0U, static_cast<Count>(population.size() - 1U)));
    for (Count i = 1U; i < sample_size; ++i) {
        const std::size_t candidate = static_cast<std::size_t>(
            random->uniform_count(0U, static_cast<Count>(population.size() - 1U)));
        if (better_individual(population[candidate], population[best])) {
            best = candidate;
        }
    }
    return best;
}

std::vector<NodeId> crossover_seed_set(const std::vector<NodeId>& parent_a,
                                       const std::vector<NodeId>& parent_b,
                                       NodeId n,
                                       Budget k,
                                       Random* random) {
    std::vector<NodeId> union_pool = parent_a;
    union_pool.insert(union_pool.end(), parent_b.begin(), parent_b.end());
    sort_unique_seeds(&union_pool);

    std::vector<NodeId> child;
    child.reserve(k);
    std::vector<std::uint8_t> selected(n, 0U);

    while (!union_pool.empty() && child.size() < k) {
        const std::size_t idx = static_cast<std::size_t>(
            random->uniform_count(0U, static_cast<Count>(union_pool.size() - 1U)));
        const NodeId node = union_pool[idx];
        if (selected[node] == 0U) {
            selected[node] = 1U;
            child.push_back(node);
        }
        union_pool[idx] = union_pool.back();
        union_pool.pop_back();
    }

    while (child.size() < k) {
        NodeId node = random->uniform_node(0U, n - 1U);
        while (selected[node] != 0U) {
            node = random->uniform_node(0U, n - 1U);
        }
        selected[node] = 1U;
        child.push_back(node);
    }
    sort_unique_seeds(&child);
    return child;
}

void apply_random_mutation(std::vector<NodeId>* seeds,
                           NodeId n,
                           Random* random) {
    if (seeds->empty() || seeds->size() >= static_cast<std::size_t>(n)) {
        return;
    }
    std::vector<std::uint8_t> selected(n, 0U);
    for (NodeId node : *seeds) {
        selected[node] = 1U;
    }
    const std::size_t idx = static_cast<std::size_t>(
        random->uniform_count(0U, static_cast<Count>(seeds->size() - 1U)));
    selected[(*seeds)[idx]] = 0U;

    NodeId replacement = random->uniform_node(0U, n - 1U);
    while (selected[replacement] != 0U) {
        replacement = random->uniform_node(0U, n - 1U);
    }
    (*seeds)[idx] = replacement;
    sort_unique_seeds(seeds);
}

void apply_hybrid_aware_mutation(std::vector<NodeId>* seeds,
                                 const HybridHypergraph& graph,
                                 Random* random) {
    const NodeId n = graph.num_nodes();
    if (seeds->empty() || seeds->size() >= static_cast<std::size_t>(n)) {
        return;
    }

    std::vector<std::uint8_t> selected(n, 0U);
    for (NodeId node : *seeds) {
        selected[node] = 1U;
    }

    const std::size_t idx = static_cast<std::size_t>(
        random->uniform_count(0U, static_cast<Count>(seeds->size() - 1U)));
    const NodeId pivot = (*seeds)[idx];
    selected[pivot] = 0U;

    std::vector<NodeId> candidates;
    candidates.reserve(graph.out_edges(pivot).size() +
                       graph.in_edges(pivot).size() +
                       graph.incident_hyperedges(pivot).size() * 3U);

    for (EdgeId eid : graph.out_edges(pivot)) {
        candidates.push_back(graph.edge(eid).dst);
    }
    for (EdgeId eid : graph.in_edges(pivot)) {
        candidates.push_back(graph.edge(eid).src);
    }
    for (HyperedgeId hid : graph.incident_hyperedges(pivot)) {
        const auto& h = graph.hyperedge(hid);
        candidates.insert(candidates.end(), h.nodes.begin(), h.nodes.end());
    }

    sort_unique_seeds(&candidates);
    std::vector<NodeId> filtered;
    filtered.reserve(candidates.size());
    for (NodeId node : candidates) {
        if (selected[node] == 0U) {
            filtered.push_back(node);
        }
    }

    if (filtered.empty()) {
        selected[pivot] = 1U;
        apply_random_mutation(seeds, n, random);
        return;
    }

    const std::size_t repl_idx = static_cast<std::size_t>(
        random->uniform_count(0U, static_cast<Count>(filtered.size() - 1U)));
    (*seeds)[idx] = filtered[repl_idx];
    sort_unique_seeds(seeds);
}

std::vector<NodeId> run_hn_moea_htc_selector(const HybridHypergraph& graph,
                                             const IMConfig& config,
                                             const std::string& method_tag) {
    const NodeId n = graph.num_nodes();
    const Budget k = std::min<Budget>(config.k, n);
    if (k == 0U || n == 0U) {
        return {};
    }

    const Count population_size = std::max<Count>(1U, config.moea_population_size);
    const Count offspring_size = std::max<Count>(1U, config.moea_offspring_size);
    const Count generations = config.moea_generations;
    const Count tournament_size = std::max<Count>(1U, config.moea_tournament_size);
    const double mutation_rate = std::min(1.0, std::max(0.0, config.moea_mutation_rate));
    const double crossover_rate = std::min(1.0, std::max(0.0, config.moea_crossover_rate));
    const Count hard_fitness_cap = 10000U;
    const Count configured_fitness_cap =
        std::max<Count>(1U, std::min(config.moea_fitness_simulations_cap, hard_fitness_cap));
    const Count capped_fitness_samples =
        std::min(config.moea_fitness_simulations, configured_fitness_cap);
    const Count fitness_simulations = std::max<Count>(1U, capped_fitness_samples);

    Random random(config.seed ^ 0xA35EC11D0D7A2B5FULL);
    Count eval_counter = 0U;

    std::vector<NodeId> ranked_nodes(n);
    for (NodeId node = 0; node < n; ++node) {
        ranked_nodes[node] = node;
    }
    std::sort(ranked_nodes.begin(), ranked_nodes.end(), [&](NodeId lhs, NodeId rhs) {
        const Count lhs_score = static_cast<Count>(graph.out_edges(lhs).size()) +
                                static_cast<Count>(graph.incident_hyperedges(lhs).size());
        const Count rhs_score = static_cast<Count>(graph.out_edges(rhs).size()) +
                                static_cast<Count>(graph.incident_hyperedges(rhs).size());
        if (lhs_score != rhs_score) {
            return lhs_score > rhs_score;
        }
        return lhs < rhs;
    });

    const std::size_t elite_pool_size =
        std::max<std::size_t>(k, static_cast<std::size_t>(n * 3U / 10U));
    const std::size_t pool_size = std::min<std::size_t>(elite_pool_size, ranked_nodes.size());
    std::vector<NodeId> node_pool(ranked_nodes.begin(), ranked_nodes.begin() + pool_size);
    std::vector<double> node_weight(pool_size, 1.0);
    for (std::size_t i = 0; i < pool_size; ++i) {
        const NodeId node = node_pool[i];
        node_weight[i] = static_cast<double>(graph.out_edges(node).size() +
                                             graph.incident_hyperedges(node).size() + 1U);
    }

    std::vector<MOEAIndividual> population;
    population.reserve(population_size);
    ProgressTracker init_tracker(method_tag, "population_init", population_size, kProgressPercentStep);
    for (Count i = 0; i < population_size; ++i) {
        MOEAIndividual individual;
        if (i < population_size / 2U) {
            individual.seeds = random_seed_set(n, k, &random);
        } else {
            individual.seeds = weighted_seed_set(node_pool, node_weight, n, k, &random);
        }
        individual.fitness = evaluate_moea_fitness(
            graph, individual.seeds, fitness_simulations, config.seed, eval_counter++);
        population.push_back(std::move(individual));
        init_tracker.maybe_log(i + 1U);
    }
    std::sort(population.begin(), population.end(), better_individual);

    Count stale_generations = 0U;
    double global_best_fitness = population.front().fitness;
    const auto t_evolve = SteadyClock::now();

    for (Count generation = 0; generation < generations; ++generation) {
        std::vector<MOEAIndividual> offspring;
        offspring.reserve(offspring_size);

        for (Count i = 0; i < offspring_size; ++i) {
            const std::size_t parent_a_idx = tournament_pick_index(population, tournament_size, &random);
            const std::size_t parent_b_idx = tournament_pick_index(population, tournament_size, &random);

            std::vector<NodeId> child = population[parent_a_idx].seeds;
            if (random.uniform_real() < crossover_rate) {
                child = crossover_seed_set(
                    population[parent_a_idx].seeds,
                    population[parent_b_idx].seeds,
                    n,
                    k,
                    &random);
            }

            if (random.uniform_real() < mutation_rate) {
                if (random.uniform_real() < 0.5) {
                    apply_random_mutation(&child, n, &random);
                } else {
                    apply_hybrid_aware_mutation(&child, graph, &random);
                }
            }
            fill_missing_seeds(&child, n, k, &random);

            MOEAIndividual child_individual;
            child_individual.seeds = std::move(child);
            child_individual.fitness = evaluate_moea_fitness(
                graph, child_individual.seeds, fitness_simulations, config.seed, eval_counter++);
            offspring.push_back(std::move(child_individual));
        }

        population.insert(population.end(), offspring.begin(), offspring.end());
        std::sort(population.begin(), population.end(), better_individual);
        if (population.size() > population_size) {
            population.resize(population_size);
        }

        const double current_best = population.front().fitness;
        if (current_best > global_best_fitness + 1e-12) {
            global_best_fitness = current_best;
            stale_generations = 0U;
        } else {
            ++stale_generations;
        }

        double avg_fitness = 0.0;
        for (const auto& individual : population) {
            avg_fitness += individual.fitness;
        }
        avg_fitness /= static_cast<double>(population.size());

        const Count done = generation + 1U;
        const double elapsed = elapsed_seconds(t_evolve, SteadyClock::now());
        const double rate = (elapsed > 1e-12) ? (static_cast<double>(done) / elapsed) : 0.0;
        const double remaining = static_cast<double>(generations - done);
        const double eta = (rate > 1e-12) ? (remaining / rate) : 0.0;
        std::cout << "[baseline:" << method_tag << "] phase=moea_evolve gen="
                  << done << "/" << generations
                  << ", best_fitness=" << current_best
                  << ", avg_fitness=" << avg_fitness
                  << ", stale_gens=" << stale_generations
                  << ", fitness_evals=" << eval_counter
                  << ", elapsed_s=" << elapsed
                  << ", eta_s=" << eta
                  << ", total_elapsed_s=" << runtime_elapsed_seconds()
                  << std::endl;
    }

    std::sort(population.begin(), population.end(), better_individual);
    return population.front().seeds;
}

}  // namespace

MCGreedyBaseline::MCGreedyBaseline(const HybridHypergraph& graph) : graph_(graph) {}

IMResult MCGreedyBaseline::run(const IMConfig& config) const {
    const std::string method_tag = "mc-greedy";
    const NodeId n = graph_.num_nodes();
    const Budget steps = clipped_budget(config.k, n);
    if (config.mc_greedy_simulations == 0U) {
        throw std::invalid_argument("IMConfig.mc_greedy_simulations must be > 0");
    }

    const auto t0 = SteadyClock::now();
    log_phase_start(method_tag,
                    "select_seeds",
                    "k=" + std::to_string(steps) +
                        ", mc_per_candidate=" + std::to_string(config.mc_greedy_simulations));

    std::vector<std::uint8_t> chosen(n, 0U);
    std::vector<NodeId> seeds;
    seeds.reserve(steps);

    for (Budget step = 0; step < steps; ++step) {
        const auto t_step = SteadyClock::now();
        const Count total_candidates = static_cast<Count>(n - static_cast<NodeId>(seeds.size()));
        const std::vector<Count> candidate_checkpoints =
            build_progress_checkpoints(total_candidates, kProgressPercentStep);
        std::size_t checkpoint_idx = 0U;
        Count evaluated_candidates = 0U;

        NodeId best = kInvalidNodeId;
        double best_value = -1.0;

        for (NodeId v = 0; v < n; ++v) {
            if (chosen[v] != 0U) {
                continue;
            }
            auto trial = seeds;
            trial.push_back(v);
            const double value = estimate_htc_influence(graph_, trial, config.mc_greedy_simulations,
                                                        config.seed + static_cast<std::uint64_t>(step) * 1000003ULL +
                                                            static_cast<std::uint64_t>(v));

            if (best == kInvalidNodeId || value > best_value || (value == best_value && v < best)) {
                best = v;
                best_value = value;
            }

            ++evaluated_candidates;
            while (checkpoint_idx < candidate_checkpoints.size() &&
                   evaluated_candidates >= candidate_checkpoints[checkpoint_idx]) {
                const NodeId best_node_for_log = (best == kInvalidNodeId) ? 0U : best;
                log_progress_snapshot(
                    method_tag,
                    "candidate_eval",
                    candidate_checkpoints[checkpoint_idx],
                    total_candidates,
                    t_step,
                    "step=" + std::to_string(step + 1U) + "/" + std::to_string(steps) +
                        ", current_best_seed=" + std::to_string(best_node_for_log) +
                        ", current_best_gain=" + std::to_string(best_value));
                ++checkpoint_idx;
            }
        }

        if (best == kInvalidNodeId) {
            break;
        }
        chosen[best] = 1U;
        seeds.push_back(best);
        const double step_elapsed = elapsed_seconds(t_step, SteadyClock::now());
        std::cout << "[baseline:" << method_tag << "] phase=seed_select progress="
                  << seeds.size() << "/" << steps
                  << ", latest_seed=" << best
                  << ", selected_gain=" << best_value
                  << ", step_elapsed_s=" << step_elapsed
                  << ", total_elapsed_s=" << runtime_elapsed_seconds()
                  << std::endl;
    }
    log_phase_end(method_tag, "select_seeds", t0, "selected=" + std::to_string(seeds.size()));
    ensure_seed_budget(graph_, config.k, &seeds);
    if (seeds.size() != steps) {
        std::cout << "[baseline:" << method_tag
                  << "] phase=seed_select detail=seed_budget_adjusted"
                  << ", adjusted_size=" << seeds.size()
                  << ", total_elapsed_s=" << runtime_elapsed_seconds() << std::endl;
    }

    const auto t_algo_end = SteadyClock::now();
    const Count pre_eval_peak_rss_kb = current_peak_rss_kb();
    log_phase_end(method_tag, "run", t0, "total_selected=" + std::to_string(seeds.size()));
    IMResult result = evaluate_seed_set(
        graph_, method_tag, seeds, config.mc_simulations, config.seed + 17ULL);
    result.time_total_seconds = elapsed_seconds(t0, t_algo_end);
    result.seconds = result.time_total_seconds;
    result.peak_rss_kb = pre_eval_peak_rss_kb;
    return result;
}

URIMBaseline::URIMBaseline(const HybridHypergraph& graph) : graph_(graph) {}

IMResult URIMBaseline::run(const IMConfig& config) const {
    const std::string method_tag = "ur-im";
    const auto t0 = SteadyClock::now();
    log_phase_start(method_tag, "opim_proxy", "graph=upper-relaxed-aux");

    const ICGraph g_plus = build_upper_relaxed_aux_ic_graph(graph_);
    OPIMProxy proxy;
    const Budget target_k = clipped_budget(config.k, graph_.num_nodes());
    auto proxy_result =
        proxy.run(g_plus, target_k, config.epsilon, config.delta, config.seed, graph_.num_nodes());
    log_phase_end(method_tag,
                  "opim_proxy",
                  t0,
                  "rr_samples=" + std::to_string(proxy_result.rr_samples) +
                      ", upper_bound=" + std::to_string(proxy_result.upper_bound));
    ensure_seed_budget(graph_, config.k, &proxy_result.seeds);

    const auto t_algo_end = SteadyClock::now();
    const Count pre_eval_peak_rss_kb = current_peak_rss_kb();
    log_phase_end(method_tag, "run", t0);
    IMResult result = evaluate_seed_set(graph_,
                                        method_tag,
                                        proxy_result.seeds,
                                        config.mc_simulations,
                                        config.seed + 19ULL);
    result.rr_samples = proxy_result.rr_samples;
    result.upper_bound_B = proxy_result.upper_bound;
    result.time_total_seconds = elapsed_seconds(t0, t_algo_end);
    result.seconds = result.time_total_seconds;
    result.peak_rss_kb = pre_eval_peak_rss_kb;
    return result;
}

URIMHistBaseline::URIMHistBaseline(const HybridHypergraph& graph) : graph_(graph) {}

IMResult URIMHistBaseline::run(const IMConfig& config) const {
    const std::string method_tag = "ur-im-hist";
    const auto t0 = SteadyClock::now();
    log_phase_start(method_tag, "hist_proxy", "graph=upper-relaxed-aux");

    const ICGraph g_plus = build_upper_relaxed_aux_ic_graph(graph_);
    HISTProxy proxy;
    const Budget target_k = clipped_budget(config.k, graph_.num_nodes());
    auto proxy_result = proxy.run(
        g_plus, target_k, config.epsilon, config.delta, config.seed, graph_.num_nodes());
    log_phase_end(method_tag,
                  "hist_proxy",
                  t0,
                  "rr_samples=" + std::to_string(proxy_result.rr_samples) +
                      ", upper_bound=" + std::to_string(proxy_result.upper_bound));
    ensure_seed_budget(graph_, config.k, &proxy_result.seeds);

    const auto t_algo_end = SteadyClock::now();
    const Count pre_eval_peak_rss_kb = current_peak_rss_kb();
    log_phase_end(method_tag, "run", t0);
    IMResult result = evaluate_seed_set(graph_,
                                        method_tag,
                                        proxy_result.seeds,
                                        config.mc_simulations,
                                        config.seed + 37ULL);
    result.rr_samples = proxy_result.rr_samples;
    result.upper_bound_B = proxy_result.upper_bound;
    result.time_total_seconds = elapsed_seconds(t0, t_algo_end);
    result.seconds = result.time_total_seconds;
    result.peak_rss_kb = pre_eval_peak_rss_kb;
    return result;
}

EdgeOnlyBaseline::EdgeOnlyBaseline(const HybridHypergraph& graph) : graph_(graph) {}

IMResult EdgeOnlyBaseline::run(const IMConfig& config) const {
    const std::string method_tag = "edge-only";
    const auto t0 = SteadyClock::now();
    log_phase_start(method_tag, "opim_proxy", "graph=edge-only");

    const ICGraph g_edge = build_edge_only_ic_graph(graph_);
    OPIMProxy proxy;
    const Budget target_k = clipped_budget(config.k, graph_.num_nodes());
    auto proxy_result = proxy.run(g_edge, target_k, config.epsilon, config.delta, config.seed);
    log_phase_end(method_tag,
                  "opim_proxy",
                  t0,
                  "rr_samples=" + std::to_string(proxy_result.rr_samples) +
                      ", upper_bound=" + std::to_string(proxy_result.upper_bound));
    ensure_seed_budget(graph_, config.k, &proxy_result.seeds);

    const auto t_algo_end = SteadyClock::now();
    const Count pre_eval_peak_rss_kb = current_peak_rss_kb();
    log_phase_end(method_tag, "run", t0);
    IMResult result = evaluate_seed_set(graph_,
                                        method_tag,
                                        proxy_result.seeds,
                                        config.mc_simulations,
                                        config.seed + 23ULL);
    result.rr_samples = proxy_result.rr_samples;
    result.upper_bound_B = proxy_result.upper_bound;
    result.time_total_seconds = elapsed_seconds(t0, t_algo_end);
    result.seconds = result.time_total_seconds;
    result.peak_rss_kb = pre_eval_peak_rss_kb;
    return result;
}

EdgeOnlyHistBaseline::EdgeOnlyHistBaseline(const HybridHypergraph& graph) : graph_(graph) {}

IMResult EdgeOnlyHistBaseline::run(const IMConfig& config) const {
    const std::string method_tag = "edge-only-hist";
    const auto t0 = SteadyClock::now();
    log_phase_start(method_tag, "hist_proxy", "graph=edge-only");

    const ICGraph g_edge = build_edge_only_ic_graph(graph_);
    HISTProxy proxy;
    const Budget target_k = clipped_budget(config.k, graph_.num_nodes());
    auto proxy_result = proxy.run(g_edge, target_k, config.epsilon, config.delta, config.seed);
    log_phase_end(method_tag,
                  "hist_proxy",
                  t0,
                  "rr_samples=" + std::to_string(proxy_result.rr_samples) +
                      ", upper_bound=" + std::to_string(proxy_result.upper_bound));
    ensure_seed_budget(graph_, config.k, &proxy_result.seeds);

    const auto t_algo_end = SteadyClock::now();
    const Count pre_eval_peak_rss_kb = current_peak_rss_kb();
    log_phase_end(method_tag, "run", t0);
    IMResult result = evaluate_seed_set(graph_,
                                        method_tag,
                                        proxy_result.seeds,
                                        config.mc_simulations,
                                        config.seed + 41ULL);
    result.rr_samples = proxy_result.rr_samples;
    result.upper_bound_B = proxy_result.upper_bound;
    result.time_total_seconds = elapsed_seconds(t0, t_algo_end);
    result.seconds = result.time_total_seconds;
    result.peak_rss_kb = pre_eval_peak_rss_kb;
    return result;
}

CliqueExpansionBaseline::CliqueExpansionBaseline(const HybridHypergraph& graph) : graph_(graph) {}

IMResult CliqueExpansionBaseline::run(const IMConfig& config) const {
    const std::string method_tag = "htc-ce";
    const auto t0 = SteadyClock::now();
    log_phase_start(method_tag, "opim_proxy", "graph=clique-implicit");

    const Budget target_k = clipped_budget(config.k, graph_.num_nodes());
    auto proxy_result = run_opim_proxy_clique_implicit(
        graph_, target_k, config.epsilon, config.delta, config.seed);
    log_phase_end(method_tag,
                  "opim_proxy",
                  t0,
                  "rr_samples=" + std::to_string(proxy_result.rr_samples) +
                      ", upper_bound=" + std::to_string(proxy_result.upper_bound));
    ensure_seed_budget(graph_, config.k, &proxy_result.seeds);

    const auto t_algo_end = SteadyClock::now();
    const Count pre_eval_peak_rss_kb = current_peak_rss_kb();
    log_phase_end(method_tag, "run", t0);
    IMResult result = evaluate_seed_set(graph_,
                                        method_tag,
                                        proxy_result.seeds,
                                        config.mc_simulations,
                                        config.seed + 29ULL);
    result.rr_samples = proxy_result.rr_samples;
    result.upper_bound_B = proxy_result.upper_bound;
    result.time_total_seconds = elapsed_seconds(t0, t_algo_end);
    result.seconds = result.time_total_seconds;
    result.peak_rss_kb = pre_eval_peak_rss_kb;
    return result;
}

CliqueExpansionHistBaseline::CliqueExpansionHistBaseline(const HybridHypergraph& graph)
    : graph_(graph) {}

IMResult CliqueExpansionHistBaseline::run(const IMConfig& config) const {
    const std::string method_tag = "htc-ce-hist";
    const auto t0 = SteadyClock::now();
    log_phase_start(method_tag, "hist_proxy", "graph=clique-implicit");

    const Budget target_k = clipped_budget(config.k, graph_.num_nodes());
    auto proxy_result = run_hist_proxy_clique_implicit(
        graph_, target_k, config.epsilon, config.delta, config.seed);
    log_phase_end(method_tag,
                  "hist_proxy",
                  t0,
                  "rr_samples=" + std::to_string(proxy_result.rr_samples) +
                      ", upper_bound=" + std::to_string(proxy_result.upper_bound));
    ensure_seed_budget(graph_, config.k, &proxy_result.seeds);

    const auto t_algo_end = SteadyClock::now();
    const Count pre_eval_peak_rss_kb = current_peak_rss_kb();
    log_phase_end(method_tag, "run", t0);
    IMResult result = evaluate_seed_set(graph_,
                                        method_tag,
                                        proxy_result.seeds,
                                        config.mc_simulations,
                                        config.seed + 43ULL);
    result.rr_samples = proxy_result.rr_samples;
    result.upper_bound_B = proxy_result.upper_bound;
    result.time_total_seconds = elapsed_seconds(t0, t_algo_end);
    result.seconds = result.time_total_seconds;
    result.peak_rss_kb = pre_eval_peak_rss_kb;
    return result;
}

HybridDegreeBaseline::HybridDegreeBaseline(const HybridHypergraph& graph) : graph_(graph) {}

IMResult HybridDegreeBaseline::run(const IMConfig& config,
                                   double edge_weight,
                                   double hyperedge_weight) const {
    const std::string method_tag = "hybrid-degree";
    const auto t0 = SteadyClock::now();
    log_phase_start(method_tag,
                    "rank_nodes",
                    "edge_weight=" + std::to_string(edge_weight) +
                        ", hyperedge_weight=" + std::to_string(hyperedge_weight));

    std::vector<double> score(graph_.num_nodes(), 0.0);
    for (NodeId v = 0; v < graph_.num_nodes(); ++v) {
        const double out_deg = static_cast<double>(graph_.out_edges(v).size());
        const double hyp_deg = static_cast<double>(graph_.incident_hyperedges(v).size());
        score[v] = edge_weight * out_deg + hyperedge_weight * hyp_deg;
    }

    std::vector<NodeId> seeds = top_k_by_score(score, config.k);
    log_phase_end(method_tag, "rank_nodes", t0, "preliminary_selected=" + std::to_string(seeds.size()));
    ensure_seed_budget(graph_, config.k, &seeds);
    std::cout << "[baseline:" << method_tag << "] phase=seed_select done"
              << ", selected=" << seeds.size() << "/" << clipped_budget(config.k, graph_.num_nodes())
              << ", total_elapsed_s=" << runtime_elapsed_seconds()
              << std::endl;

    const auto t_algo_end = SteadyClock::now();
    const Count pre_eval_peak_rss_kb = current_peak_rss_kb();
    log_phase_end(method_tag, "run", t0);
    IMResult result = evaluate_seed_set(
        graph_, method_tag, seeds, config.mc_simulations, config.seed + 31ULL);
    result.time_total_seconds = elapsed_seconds(t0, t_algo_end);
    result.seconds = result.time_total_seconds;
    result.peak_rss_kb = pre_eval_peak_rss_kb;
    return result;
}

HCI1TMBaseline::HCI1TMBaseline(const HybridHypergraph& graph) : graph_(graph) {}

IMResult HCI1TMBaseline::run(const IMConfig& config) const {
    const std::string method_tag = "hci1-tm";
    const auto t0 = SteadyClock::now();
    log_phase_start(method_tag, "seed_select", "order=HCI1");
    std::vector<NodeId> seeds = run_hci_tm_selector(graph_, config.k, HCITMOrder::HCI1, method_tag);
    ensure_seed_budget(graph_, config.k, &seeds);
    log_phase_end(method_tag, "seed_select", t0, "selected=" + std::to_string(seeds.size()));
    const auto t_algo_end = SteadyClock::now();
    const Count pre_eval_peak_rss_kb = current_peak_rss_kb();
    log_phase_end(method_tag, "run", t0);
    IMResult result = evaluate_seed_set(
        graph_, method_tag, seeds, config.mc_simulations, config.seed + 47ULL);
    result.time_total_seconds = elapsed_seconds(t0, t_algo_end);
    result.seconds = result.time_total_seconds;
    result.peak_rss_kb = pre_eval_peak_rss_kb;
    return result;
}

HCI2TMBaseline::HCI2TMBaseline(const HybridHypergraph& graph) : graph_(graph) {}

IMResult HCI2TMBaseline::run(const IMConfig& config) const {
    const std::string method_tag = "hci2-tm";
    const auto t0 = SteadyClock::now();
    log_phase_start(method_tag, "seed_select", "order=HCI2");
    std::vector<NodeId> seeds = run_hci_tm_selector(graph_, config.k, HCITMOrder::HCI2, method_tag);
    ensure_seed_budget(graph_, config.k, &seeds);
    log_phase_end(method_tag, "seed_select", t0, "selected=" + std::to_string(seeds.size()));
    const auto t_algo_end = SteadyClock::now();
    const Count pre_eval_peak_rss_kb = current_peak_rss_kb();
    log_phase_end(method_tag, "run", t0);
    IMResult result = evaluate_seed_set(
        graph_, method_tag, seeds, config.mc_simulations, config.seed + 53ULL);
    result.time_total_seconds = elapsed_seconds(t0, t_algo_end);
    result.seconds = result.time_total_seconds;
    result.peak_rss_kb = pre_eval_peak_rss_kb;
    return result;
}

HNMOEAHTCBaseline::HNMOEAHTCBaseline(const HybridHypergraph& graph) : graph_(graph) {}

IMResult HNMOEAHTCBaseline::run(const IMConfig& config) const {
    const std::string method_tag = "hn-moea-htc";
    const auto t0 = SteadyClock::now();
    log_phase_start(method_tag,
                    "seed_select",
                    "population=" + std::to_string(config.moea_population_size) +
                        ", offspring=" + std::to_string(config.moea_offspring_size) +
                        ", generations=" + std::to_string(config.moea_generations));
    std::vector<NodeId> seeds = run_hn_moea_htc_selector(graph_, config, method_tag);
    ensure_seed_budget(graph_, config.k, &seeds);
    log_phase_end(method_tag, "seed_select", t0, "selected=" + std::to_string(seeds.size()));
    const auto t_algo_end = SteadyClock::now();
    const Count pre_eval_peak_rss_kb = current_peak_rss_kb();
    log_phase_end(method_tag, "run", t0);
    IMResult result = evaluate_seed_set(
        graph_, method_tag, seeds, config.mc_simulations, config.seed + 59ULL);
    result.time_total_seconds = elapsed_seconds(t0, t_algo_end);
    result.seconds = result.time_total_seconds;
    result.peak_rss_kb = pre_eval_peak_rss_kb;
    return result;
}

}  // namespace htc
