#include "htc/ic.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "htc/runtime_clock.hpp"

namespace htc {
namespace {

using SteadyClock = std::chrono::steady_clock;
constexpr Count kOpimProgressPercentStep = 10U;

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

std::vector<Count> build_progress_checkpoints(Count begin_count,
                                              Count end_count,
                                              Count percent_step) {
    std::vector<Count> checkpoints;
    if (end_count <= begin_count || percent_step == 0U) {
        return checkpoints;
    }
    const Count span = end_count - begin_count;
    const Count step = std::min<Count>(percent_step, 100U);
    for (Count pct = step; pct <= 100U; pct += step) {
        Count checkpoint = begin_count + ceil_to_count(
                                             (static_cast<long double>(span) *
                                              static_cast<long double>(pct)) /
                                             100.0L);
        checkpoint = std::max<Count>(begin_count + 1U, std::min<Count>(checkpoint, end_count));
        if (checkpoints.empty() || checkpoints.back() != checkpoint) {
            checkpoints.push_back(checkpoint);
        }
    }
    if (checkpoints.empty() || checkpoints.back() != end_count) {
        checkpoints.push_back(end_count);
    }
    return checkpoints;
}

void log_opim_rr_progress(Count iter,
                          Count num_iter,
                          const char* stream_tag,
                          Count begin_count,
                          Count completed,
                          Count target,
                          SteadyClock::time_point begin) {
    const Count phase_completed = (completed > begin_count) ? (completed - begin_count) : 0U;
    const Count phase_target = (target > begin_count) ? (target - begin_count) : 0U;
    const double elapsed = elapsed_seconds(begin, SteadyClock::now());
    const double rate =
        (elapsed > 1e-12) ? (static_cast<double>(phase_completed) / elapsed) : 0.0;
    const double remaining = static_cast<double>((target > completed) ? (target - completed) : 0U);
    const double eta = (rate > 1e-12) ? (remaining / rate) : 0.0;
    const double pct = (phase_target > 0U)
                           ? (static_cast<double>(phase_completed) * 100.0 /
                              static_cast<double>(phase_target))
                           : 100.0;

    std::cout << "[opim-c] rr progress: iter=" << iter << "/" << num_iter
              << ", stream=" << stream_tag
              << ", progress=" << phase_completed << "/" << phase_target
              << " (" << pct << "%)"
              << ", elapsed_s=" << elapsed
              << ", eta_s=" << eta
              << ", rate=" << rate << "/s"
              << ", total_elapsed_s=" << runtime_elapsed_seconds()
              << std::endl;
}

std::uint64_t pair_key(NodeId u, NodeId v) {
    return (static_cast<std::uint64_t>(u) << 32U) | static_cast<std::uint64_t>(v);
}

void accumulate_prob(std::unordered_map<std::uint64_t, Prob>* pair_prob,
                     NodeId u,
                     NodeId v,
                     Prob p) {
    if (pair_prob == nullptr) {
        throw std::invalid_argument("pair_prob cannot be null");
    }
    if (p < 0.0 || p > 1.0) {
        throw std::invalid_argument("accumulated probability out of [0,1]");
    }

    const std::uint64_t key = pair_key(u, v);
    auto it = pair_prob->find(key);
    if (it == pair_prob->end()) {
        pair_prob->emplace(key, p);
        return;
    }

    const Prob old_p = it->second;
    it->second = 1.0 - (1.0 - old_p) * (1.0 - p);
}

Count covered_rr_count(const std::vector<std::vector<Count>>& node_to_rr,
                       const std::vector<NodeId>& seeds,
                       Count rr_count) {
    if (rr_count == 0U) {
        return 0U;
    }

    std::vector<std::uint8_t> covered(static_cast<std::size_t>(rr_count), 0U);
    Count total = 0U;
    for (NodeId v : seeds) {
        if (v >= node_to_rr.size()) {
            throw std::invalid_argument("seed node out of range for RR index");
        }
        for (Count rid : node_to_rr[v]) {
            if (rid >= rr_count) {
                throw std::invalid_argument("RR index out of range in node_to_rr");
            }
            auto& flag = covered[static_cast<std::size_t>(rid)];
            if (flag == 0U) {
                flag = 1U;
                ++total;
            }
        }
    }
    return total;
}

double pow2(double value) {
    return value * value;
}

double logcnk(Count n, Count k) {
    if (k > n) {
        throw std::invalid_argument("logcnk requires k <= n");
    }
    k = std::min(k, n - k);
    long double res = 0.0L;
    for (Count i = 1U; i <= k; ++i) {
        const long double numer = static_cast<long double>(n - k + i);
        const long double denom = static_cast<long double>(i);
        res += std::log(numer / denom);
    }
    return static_cast<double>(res);
}

void make_min_heap(std::vector<Count>* values) {
    if (values == nullptr) {
        throw std::invalid_argument("values cannot be null");
    }
    std::make_heap(values->begin(), values->end(), std::greater<Count>());
}

void min_heap_replace_min_value(std::vector<Count>* values, Count replacement) {
    if (values == nullptr || values->empty()) {
        throw std::invalid_argument("values cannot be null or empty");
    }
    std::pop_heap(values->begin(), values->end(), std::greater<Count>());
    values->back() = replacement;
    std::push_heap(values->begin(), values->end(), std::greater<Count>());
}

[[maybe_unused]] void append_rr_sets_until(const RRSampler& sampler,
                                           Random& random,
                                           Count target_count,
                                           NodeId num_nodes,
                                           std::vector<RRSet>* rr_sets,
                                           std::vector<std::vector<Count>>* node_to_rr) {
    if (rr_sets == nullptr || node_to_rr == nullptr) {
        throw std::invalid_argument("RR set containers cannot be null");
    }
    if (node_to_rr->size() != num_nodes) {
        throw std::invalid_argument("node_to_rr size mismatch");
    }

    while (rr_sets->size() < static_cast<std::size_t>(target_count)) {
        if (rr_sets->size() >= static_cast<std::size_t>(std::numeric_limits<Count>::max())) {
            throw std::overflow_error("RR set count exceeds Count range");
        }
        RRSet rr = sampler.sample(random);
        const Count rid = static_cast<Count>(rr_sets->size());
        for (NodeId node : rr.nodes) {
            if (node >= num_nodes) {
                throw std::invalid_argument("RR-set contains node out of range");
            }
            (*node_to_rr)[node].push_back(rid);
        }
        rr_sets->push_back(std::move(rr));
    }
}

RRSet sample_rr_set_with_projection(const ICGraph& graph,
                                    NodeId root,
                                    Random& random,
                                    NodeId kept_node_count) {
    if (root >= graph.num_nodes()) {
        throw std::out_of_range("RR root out of range");
    }
    if (kept_node_count == 0U || kept_node_count > graph.num_nodes()) {
        throw std::invalid_argument("invalid kept_node_count for projected RR sampling");
    }

    RRSet rr;
    rr.root = root;

    std::vector<std::uint8_t> visited(graph.num_nodes(), 0U);
    std::queue<NodeId> queue;
    queue.push(root);
    visited[root] = 1U;
    if (root < kept_node_count) {
        rr.nodes.push_back(root);
    }

    while (!queue.empty()) {
        const NodeId v = queue.front();
        queue.pop();
        for (EdgeId eid : graph.in_edges(v)) {
            const auto& e = graph.edge(eid);
            if (visited[e.src] != 0U) {
                continue;
            }
            if (!random.bernoulli(e.prob)) {
                continue;
            }
            visited[e.src] = 1U;
            queue.push(e.src);
            if (e.src < kept_node_count) {
                rr.nodes.push_back(e.src);
            }
        }
    }

    return rr;
}

void append_rr_sets_until_projected(const ICGraph& graph,
                                    Random& random,
                                    Count target_count,
                                    NodeId root_node_count,
                                    NodeId kept_node_count,
                                    std::vector<RRSet>* rr_sets,
                                    std::vector<std::vector<Count>>* node_to_rr) {
    if (rr_sets == nullptr || node_to_rr == nullptr) {
        throw std::invalid_argument("RR set containers cannot be null");
    }
    if (root_node_count == 0U || root_node_count > graph.num_nodes()) {
        throw std::invalid_argument("invalid root_node_count for projected RR sampling");
    }
    if (kept_node_count == 0U || kept_node_count > graph.num_nodes()) {
        throw std::invalid_argument("invalid kept_node_count for projected RR sampling");
    }
    if (node_to_rr->size() != kept_node_count) {
        throw std::invalid_argument("projected RR node_to_rr size mismatch");
    }

    while (rr_sets->size() < static_cast<std::size_t>(target_count)) {
        if (rr_sets->size() >= static_cast<std::size_t>(std::numeric_limits<Count>::max())) {
            throw std::overflow_error("RR set count exceeds Count range");
        }
        const NodeId root = random.uniform_node(0U, root_node_count - 1U);
        RRSet rr = sample_rr_set_with_projection(graph, root, random, kept_node_count);
        const Count rid = static_cast<Count>(rr_sets->size());
        for (NodeId node : rr.nodes) {
            if (node >= kept_node_count) {
                throw std::invalid_argument("projected RR-set contains node out of range");
            }
            (*node_to_rr)[node].push_back(rid);
        }
        rr_sets->push_back(std::move(rr));
    }
}

void append_rr_sets_until_projected_with_progress(const ICGraph& graph,
                                                  Random& random,
                                                  Count target_count,
                                                  NodeId root_node_count,
                                                  NodeId kept_node_count,
                                                  std::vector<RRSet>* rr_sets,
                                                  std::vector<std::vector<Count>>* node_to_rr,
                                                  bool enable_progress_log,
                                                  Count iter,
                                                  Count num_iter,
                                                  const char* stream_tag,
                                                  SteadyClock::time_point begin) {
    if (rr_sets == nullptr || node_to_rr == nullptr) {
        throw std::invalid_argument("RR set containers cannot be null");
    }
    if (root_node_count == 0U || root_node_count > graph.num_nodes()) {
        throw std::invalid_argument("invalid root_node_count for projected RR sampling");
    }
    if (kept_node_count == 0U || kept_node_count > graph.num_nodes()) {
        throw std::invalid_argument("invalid kept_node_count for projected RR sampling");
    }
    if (node_to_rr->size() != kept_node_count) {
        throw std::invalid_argument("projected RR node_to_rr size mismatch");
    }

    std::size_t next_checkpoint_idx = 0U;
    std::vector<Count> checkpoints;
    const Count begin_count = static_cast<Count>(rr_sets->size());
    if (enable_progress_log) {
        checkpoints = build_progress_checkpoints(begin_count,
                                                 target_count,
                                                 kOpimProgressPercentStep);
    }

    while (rr_sets->size() < static_cast<std::size_t>(target_count)) {
        if (rr_sets->size() >= static_cast<std::size_t>(std::numeric_limits<Count>::max())) {
            throw std::overflow_error("RR set count exceeds Count range");
        }
        const NodeId root = random.uniform_node(0U, root_node_count - 1U);
        RRSet rr = sample_rr_set_with_projection(graph, root, random, kept_node_count);
        const Count rid = static_cast<Count>(rr_sets->size());
        for (NodeId node : rr.nodes) {
            if (node >= kept_node_count) {
                throw std::invalid_argument("projected RR-set contains node out of range");
            }
            (*node_to_rr)[node].push_back(rid);
        }
        rr_sets->push_back(std::move(rr));

        if (!enable_progress_log) {
            continue;
        }
        while (next_checkpoint_idx < checkpoints.size() &&
               rr_sets->size() >= static_cast<std::size_t>(checkpoints[next_checkpoint_idx])) {
            const Count completed = checkpoints[next_checkpoint_idx];
            log_opim_rr_progress(
                iter, num_iter, stream_tag, begin_count, completed, target_count, begin);
            ++next_checkpoint_idx;
        }
    }
}

Prob clique_expansion_edge_prob(const Hyperedge& h) {
    if (h.nodes.size() < 2U) {
        return 0.0;
    }
    const double denom = static_cast<double>(h.nodes.size() - 1U);
    const double numer = static_cast<double>(h.nodes.size() - h.threshold);
    if (denom <= 0.0) {
        return 0.0;
    }
    return numer / denom;
}

RRSet sample_rr_set_with_projection_clique_implicit(const HybridHypergraph& graph,
                                                    NodeId root,
                                                    Random& random,
                                                    NodeId kept_node_count) {
    if (root >= graph.num_nodes()) {
        throw std::out_of_range("RR root out of range");
    }
    if (kept_node_count == 0U || kept_node_count > graph.num_nodes()) {
        throw std::invalid_argument("invalid kept_node_count for projected implicit clique RR");
    }

    RRSet rr;
    rr.root = root;

    std::vector<std::uint8_t> visited(graph.num_nodes(), 0U);
    std::queue<NodeId> queue;
    queue.push(root);
    visited[root] = 1U;
    if (root < kept_node_count) {
        rr.nodes.push_back(root);
    }

    while (!queue.empty()) {
        const NodeId v = queue.front();
        queue.pop();

        for (EdgeId eid : graph.in_edges(v)) {
            const auto& e = graph.edge(eid);
            if (visited[e.src] != 0U) {
                continue;
            }
            if (!random.bernoulli(e.prob)) {
                continue;
            }
            visited[e.src] = 1U;
            queue.push(e.src);
            if (e.src < kept_node_count) {
                rr.nodes.push_back(e.src);
            }
        }

        for (HyperedgeId hid : graph.incident_hyperedges(v)) {
            const auto& h = graph.hyperedge(hid);
            const Prob p_ce = clique_expansion_edge_prob(h);
            if (p_ce <= 0.0) {
                continue;
            }
            for (NodeId u : h.nodes) {
                if (u == v || visited[u] != 0U) {
                    continue;
                }
                if (!random.bernoulli(p_ce)) {
                    continue;
                }
                visited[u] = 1U;
                queue.push(u);
                if (u < kept_node_count) {
                    rr.nodes.push_back(u);
                }
            }
        }
    }

    return rr;
}

void append_rr_sets_until_projected_clique_implicit(const HybridHypergraph& graph,
                                                    Random& random,
                                                    Count target_count,
                                                    NodeId root_node_count,
                                                    NodeId kept_node_count,
                                                    std::vector<RRSet>* rr_sets,
                                                    std::vector<std::vector<Count>>* node_to_rr) {
    if (rr_sets == nullptr || node_to_rr == nullptr) {
        throw std::invalid_argument("RR set containers cannot be null");
    }
    if (root_node_count == 0U || root_node_count > graph.num_nodes()) {
        throw std::invalid_argument("invalid root_node_count for projected implicit clique RR");
    }
    if (kept_node_count == 0U || kept_node_count > graph.num_nodes()) {
        throw std::invalid_argument("invalid kept_node_count for projected implicit clique RR");
    }
    if (node_to_rr->size() != kept_node_count) {
        throw std::invalid_argument("projected RR node_to_rr size mismatch");
    }

    while (rr_sets->size() < static_cast<std::size_t>(target_count)) {
        if (rr_sets->size() >= static_cast<std::size_t>(std::numeric_limits<Count>::max())) {
            throw std::overflow_error("RR set count exceeds Count range");
        }
        const NodeId root = random.uniform_node(0U, root_node_count - 1U);
        RRSet rr = sample_rr_set_with_projection_clique_implicit(graph, root, random, kept_node_count);
        const Count rid = static_cast<Count>(rr_sets->size());
        for (NodeId node : rr.nodes) {
            if (node >= kept_node_count) {
                throw std::invalid_argument("projected RR-set contains node out of range");
            }
            (*node_to_rr)[node].push_back(rid);
        }
        rr_sets->push_back(std::move(rr));
    }
}

struct RRSetEarlyStopSample {
    RRSet rr;
    bool hit_sentinel = false;
};

RRSetEarlyStopSample sample_rr_set_with_sentinel_root(const ICGraph& graph,
                                                      NodeId root,
                                                      Random& random,
                                                      const std::vector<std::uint8_t>& sentinel_nodes) {
    if (root >= graph.num_nodes()) {
        throw std::out_of_range("RR root out of range");
    }
    if (sentinel_nodes.size() != graph.num_nodes()) {
        throw std::invalid_argument("sentinel_nodes size mismatch");
    }

    RRSetEarlyStopSample sample;
    sample.rr.root = root;

    std::vector<std::uint8_t> visited(graph.num_nodes(), 0U);
    std::queue<NodeId> queue;
    queue.push(root);
    visited[root] = 1U;
    sample.rr.nodes.push_back(root);

    if (sentinel_nodes[root] != 0U) {
        sample.hit_sentinel = true;
        return sample;
    }

    while (!queue.empty()) {
        const NodeId v = queue.front();
        queue.pop();

        for (EdgeId eid : graph.in_edges(v)) {
            const auto& e = graph.edge(eid);
            if (visited[e.src] != 0U) {
                continue;
            }
            if (!random.bernoulli(e.prob)) {
                continue;
            }
            visited[e.src] = 1U;
            sample.rr.nodes.push_back(e.src);
            if (sentinel_nodes[e.src] != 0U) {
                sample.hit_sentinel = true;
                return sample;
            }
            queue.push(e.src);
        }
    }

    return sample;
}

RRSetEarlyStopSample sample_rr_set_with_sentinel_root_projected(
    const ICGraph& graph,
    NodeId root,
    Random& random,
    const std::vector<std::uint8_t>& sentinel_nodes,
    NodeId kept_node_count) {
    if (root >= graph.num_nodes()) {
        throw std::out_of_range("RR root out of range");
    }
    if (kept_node_count == 0U || kept_node_count > graph.num_nodes()) {
        throw std::invalid_argument("kept_node_count out of range for projected sentinel RR");
    }
    if (sentinel_nodes.size() != kept_node_count) {
        throw std::invalid_argument("sentinel_nodes size mismatch for projected sentinel RR");
    }

    RRSetEarlyStopSample sample;
    sample.rr.root = root;

    std::vector<std::uint8_t> visited(graph.num_nodes(), 0U);
    std::queue<NodeId> queue;
    queue.push(root);
    visited[root] = 1U;
    if (root < kept_node_count) {
        sample.rr.nodes.push_back(root);
        if (sentinel_nodes[root] != 0U) {
            sample.hit_sentinel = true;
            return sample;
        }
    }

    while (!queue.empty()) {
        const NodeId v = queue.front();
        queue.pop();

        for (EdgeId eid : graph.in_edges(v)) {
            const auto& e = graph.edge(eid);
            if (visited[e.src] != 0U) {
                continue;
            }
            if (!random.bernoulli(e.prob)) {
                continue;
            }
            visited[e.src] = 1U;
            queue.push(e.src);

            if (e.src >= kept_node_count) {
                continue;
            }
            sample.rr.nodes.push_back(e.src);
            if (sentinel_nodes[e.src] != 0U) {
                sample.hit_sentinel = true;
                return sample;
            }
        }
    }

    return sample;
}

RRSetEarlyStopSample sample_rr_set_with_sentinel_root_projected_clique_implicit(
    const HybridHypergraph& graph,
    NodeId root,
    Random& random,
    const std::vector<std::uint8_t>& sentinel_nodes,
    NodeId kept_node_count) {
    if (root >= graph.num_nodes()) {
        throw std::out_of_range("RR root out of range");
    }
    if (kept_node_count == 0U || kept_node_count > graph.num_nodes()) {
        throw std::invalid_argument("kept_node_count out of range for projected implicit clique sentinel RR");
    }
    if (sentinel_nodes.size() != kept_node_count) {
        throw std::invalid_argument("sentinel_nodes size mismatch for projected implicit clique sentinel RR");
    }

    RRSetEarlyStopSample sample;
    sample.rr.root = root;

    std::vector<std::uint8_t> visited(graph.num_nodes(), 0U);
    std::queue<NodeId> queue;
    queue.push(root);
    visited[root] = 1U;
    if (root < kept_node_count) {
        sample.rr.nodes.push_back(root);
        if (sentinel_nodes[root] != 0U) {
            sample.hit_sentinel = true;
            return sample;
        }
    }

    while (!queue.empty()) {
        const NodeId v = queue.front();
        queue.pop();

        for (EdgeId eid : graph.in_edges(v)) {
            const auto& e = graph.edge(eid);
            if (visited[e.src] != 0U) {
                continue;
            }
            if (!random.bernoulli(e.prob)) {
                continue;
            }
            visited[e.src] = 1U;
            queue.push(e.src);

            if (e.src >= kept_node_count) {
                continue;
            }
            sample.rr.nodes.push_back(e.src);
            if (sentinel_nodes[e.src] != 0U) {
                sample.hit_sentinel = true;
                return sample;
            }
        }

        for (HyperedgeId hid : graph.incident_hyperedges(v)) {
            const auto& h = graph.hyperedge(hid);
            const Prob p_ce = clique_expansion_edge_prob(h);
            if (p_ce <= 0.0) {
                continue;
            }
            for (NodeId u : h.nodes) {
                if (u == v || visited[u] != 0U) {
                    continue;
                }
                if (!random.bernoulli(p_ce)) {
                    continue;
                }
                visited[u] = 1U;
                queue.push(u);
                if (u >= kept_node_count) {
                    continue;
                }
                sample.rr.nodes.push_back(u);
                if (sentinel_nodes[u] != 0U) {
                    sample.hit_sentinel = true;
                    return sample;
                }
            }
        }
    }

    return sample;
}

[[maybe_unused]] void append_rr_sets_until_with_sentinel(
    const ICGraph& graph,
    Random& random,
    Count target_count,
    NodeId num_nodes,
    const std::vector<std::uint8_t>& sentinel_nodes,
    std::vector<RRSet>* rr_sets,
    std::vector<std::vector<Count>>* node_to_rr,
    Count* hit_sentinel_samples) {
    if (rr_sets == nullptr || node_to_rr == nullptr) {
        throw std::invalid_argument("RR set containers cannot be null");
    }
    if (sentinel_nodes.size() != num_nodes) {
        throw std::invalid_argument("sentinel_nodes size mismatch");
    }
    if (node_to_rr->size() != num_nodes) {
        throw std::invalid_argument("node_to_rr size mismatch");
    }

    while (rr_sets->size() < static_cast<std::size_t>(target_count)) {
        if (rr_sets->size() >= static_cast<std::size_t>(std::numeric_limits<Count>::max())) {
            throw std::overflow_error("RR set count exceeds Count range");
        }
        const NodeId root = random.uniform_node(0U, num_nodes - 1U);
        RRSetEarlyStopSample sample =
            sample_rr_set_with_sentinel_root(graph, root, random, sentinel_nodes);
        const Count rid = static_cast<Count>(rr_sets->size());
        for (NodeId node : sample.rr.nodes) {
            if (node >= num_nodes) {
                throw std::invalid_argument("RR-set contains node out of range");
            }
            (*node_to_rr)[node].push_back(rid);
        }
        if (sample.hit_sentinel && hit_sentinel_samples != nullptr) {
            ++(*hit_sentinel_samples);
        }
        rr_sets->push_back(std::move(sample.rr));
    }
}

void append_rr_sets_until_with_sentinel_projected(
    const ICGraph& graph,
    Random& random,
    Count target_count,
    NodeId root_node_count,
    NodeId kept_node_count,
    const std::vector<std::uint8_t>& sentinel_nodes,
    std::vector<RRSet>* rr_sets,
    std::vector<std::vector<Count>>* node_to_rr,
    Count* hit_sentinel_samples) {
    if (rr_sets == nullptr || node_to_rr == nullptr) {
        throw std::invalid_argument("RR set containers cannot be null");
    }
    if (root_node_count == 0U || root_node_count > graph.num_nodes()) {
        throw std::invalid_argument("root_node_count out of range for projected sentinel RR");
    }
    if (kept_node_count == 0U || kept_node_count > graph.num_nodes()) {
        throw std::invalid_argument("kept_node_count out of range for projected sentinel RR");
    }
    if (sentinel_nodes.size() != kept_node_count) {
        throw std::invalid_argument("sentinel_nodes size mismatch for projected sentinel RR");
    }
    if (node_to_rr->size() != kept_node_count) {
        throw std::invalid_argument("node_to_rr size mismatch for projected sentinel RR");
    }

    while (rr_sets->size() < static_cast<std::size_t>(target_count)) {
        if (rr_sets->size() >= static_cast<std::size_t>(std::numeric_limits<Count>::max())) {
            throw std::overflow_error("RR set count exceeds Count range");
        }
        const NodeId root = random.uniform_node(0U, root_node_count - 1U);
        RRSetEarlyStopSample sample = sample_rr_set_with_sentinel_root_projected(
            graph, root, random, sentinel_nodes, kept_node_count);
        const Count rid = static_cast<Count>(rr_sets->size());
        for (NodeId node : sample.rr.nodes) {
            if (node >= kept_node_count) {
                throw std::invalid_argument("projected sentinel RR-set contains node out of range");
            }
            (*node_to_rr)[node].push_back(rid);
        }
        if (sample.hit_sentinel && hit_sentinel_samples != nullptr) {
            ++(*hit_sentinel_samples);
        }
        rr_sets->push_back(std::move(sample.rr));
    }
}

void append_rr_sets_until_with_sentinel_projected_clique_implicit(
    const HybridHypergraph& graph,
    Random& random,
    Count target_count,
    NodeId root_node_count,
    NodeId kept_node_count,
    const std::vector<std::uint8_t>& sentinel_nodes,
    std::vector<RRSet>* rr_sets,
    std::vector<std::vector<Count>>* node_to_rr,
    Count* hit_sentinel_samples) {
    if (rr_sets == nullptr || node_to_rr == nullptr) {
        throw std::invalid_argument("RR set containers cannot be null");
    }
    if (root_node_count == 0U || root_node_count > graph.num_nodes()) {
        throw std::invalid_argument("root_node_count out of range for projected implicit clique sentinel RR");
    }
    if (kept_node_count == 0U || kept_node_count > graph.num_nodes()) {
        throw std::invalid_argument("kept_node_count out of range for projected implicit clique sentinel RR");
    }
    if (sentinel_nodes.size() != kept_node_count) {
        throw std::invalid_argument("sentinel_nodes size mismatch for projected implicit clique sentinel RR");
    }
    if (node_to_rr->size() != kept_node_count) {
        throw std::invalid_argument("node_to_rr size mismatch for projected implicit clique sentinel RR");
    }

    while (rr_sets->size() < static_cast<std::size_t>(target_count)) {
        if (rr_sets->size() >= static_cast<std::size_t>(std::numeric_limits<Count>::max())) {
            throw std::overflow_error("RR set count exceeds Count range");
        }
        const NodeId root = random.uniform_node(0U, root_node_count - 1U);
        RRSetEarlyStopSample sample = sample_rr_set_with_sentinel_root_projected_clique_implicit(
            graph, root, random, sentinel_nodes, kept_node_count);
        const Count rid = static_cast<Count>(rr_sets->size());
        for (NodeId node : sample.rr.nodes) {
            if (node >= kept_node_count) {
                throw std::invalid_argument("projected sentinel RR-set contains node out of range");
            }
            (*node_to_rr)[node].push_back(rid);
        }
        if (sample.hit_sentinel && hit_sentinel_samples != nullptr) {
            ++(*hit_sentinel_samples);
        }
        rr_sets->push_back(std::move(sample.rr));
    }
}

struct MaxCoverResult {
    std::vector<NodeId> seeds;
    Count covered_rr = 0U;
    double bound_last = std::numeric_limits<double>::infinity();
    double bound_min = std::numeric_limits<double>::infinity();
};

[[maybe_unused]] MaxCoverResult max_cover_with_prefix(const std::vector<std::vector<Count>>& node_to_rr,
                                                      const std::vector<RRSet>& rr_sets,
                                                      NodeId num_nodes,
                                                      Budget k,
                                                      const std::vector<NodeId>& prefix) {
    MaxCoverResult result;
    if (k == 0U || num_nodes == 0U || rr_sets.empty()) {
        return result;
    }

    if (node_to_rr.size() != num_nodes) {
        throw std::invalid_argument("node_to_rr size mismatch");
    }

    std::vector<std::uint8_t> chosen(num_nodes, 0U);
    std::vector<std::uint8_t> covered(rr_sets.size(), 0U);
    std::vector<Count> marginal(num_nodes, 0U);

    result.seeds.reserve(k);
    Count covered_rr = 0U;
    for (NodeId v : prefix) {
        if (v >= num_nodes) {
            throw std::invalid_argument("prefix seed out of range");
        }
        if (chosen[v] != 0U) {
            continue;
        }
        chosen[v] = 1U;
        result.seeds.push_back(v);
        if (result.seeds.size() >= k) {
            break;
        }
        for (Count rid : node_to_rr[v]) {
            if (rid >= rr_sets.size()) {
                throw std::invalid_argument("RR index out of range in node_to_rr");
            }
            auto& flag = covered[static_cast<std::size_t>(rid)];
            if (flag == 0U) {
                flag = 1U;
                ++covered_rr;
            }
        }
    }

    for (NodeId v = 0; v < num_nodes; ++v) {
        if (chosen[v] != 0U) {
            continue;
        }
        Count gain = 0U;
        for (Count rid : node_to_rr[v]) {
            if (covered[static_cast<std::size_t>(rid)] == 0U) {
                ++gain;
            }
        }
        marginal[v] = gain;
    }

    while (result.seeds.size() < k) {
        NodeId best = kInvalidNodeId;
        Count best_gain = 0U;
        for (NodeId v = 0; v < num_nodes; ++v) {
            if (chosen[v] != 0U) {
                continue;
            }
            const Count gain = marginal[v];
            if (best == kInvalidNodeId || gain > best_gain || (gain == best_gain && v < best)) {
                best = v;
                best_gain = gain;
            }
        }

        if (best == kInvalidNodeId) {
            break;
        }

        chosen[best] = 1U;
        result.seeds.push_back(best);
        covered_rr += best_gain;

        for (Count rid : node_to_rr[best]) {
            if (rid >= rr_sets.size()) {
                throw std::invalid_argument("RR index out of range in node_to_rr");
            }
            auto& flag = covered[static_cast<std::size_t>(rid)];
            if (flag != 0U) {
                continue;
            }
            flag = 1U;
            for (NodeId node : rr_sets[static_cast<std::size_t>(rid)].nodes) {
                if (chosen[node] != 0U) {
                    continue;
                }
                if (marginal[node] > 0U) {
                    --marginal[node];
                }
            }
        }
    }

    Count optimistic_covered = covered_rr;
    const Budget remaining_budget = k - static_cast<Budget>(result.seeds.size());
    if (remaining_budget > 0U) {
        std::vector<Count> remain_marginals;
        remain_marginals.reserve(num_nodes);
        for (NodeId v = 0; v < num_nodes; ++v) {
            if (chosen[v] == 0U) {
                remain_marginals.push_back(marginal[v]);
            }
        }
        std::sort(remain_marginals.begin(), remain_marginals.end(), std::greater<Count>());
        const std::size_t take = std::min<std::size_t>(remain_marginals.size(), remaining_budget);
        for (std::size_t i = 0; i < take; ++i) {
            optimistic_covered += remain_marginals[i];
        }
    }

    result.covered_rr = covered_rr;
    result.bound_last = static_cast<double>(optimistic_covered) * static_cast<double>(num_nodes) /
                        static_cast<double>(rr_sets.size());
    result.bound_min = result.bound_last;
    return result;
}

MaxCoverResult max_cover_lazy(const std::vector<std::vector<Count>>& node_to_rr,
                              const std::vector<RRSet>& rr_sets,
                              NodeId num_nodes,
                              Budget k) {
    MaxCoverResult result;
    if (k == 0U || num_nodes == 0U || rr_sets.empty()) {
        return result;
    }

    std::vector<Count> coverage(num_nodes, 0U);
    Count max_deg = 0U;
    for (NodeId v = 0; v < num_nodes; ++v) {
        const Count deg = static_cast<Count>(node_to_rr[v].size());
        coverage[v] = deg;
        if (deg > max_deg) {
            max_deg = deg;
        }
    }
    if (max_deg == 0U) {
        return result;
    }

    std::vector<std::vector<NodeId>> deg_map(static_cast<std::size_t>(max_deg + 1U));
    for (NodeId v = 0; v < num_nodes; ++v) {
        const Count deg = coverage[v];
        if (deg != 0U) {
            deg_map[static_cast<std::size_t>(deg)].push_back(v);
        }
    }

    Count sum_inf = 0U;
    std::vector<std::uint8_t> rr_mark(rr_sets.size(), 0U);

    result.seeds.reserve(k);
    for (Count deg = max_deg; deg > 0U; --deg) {
        auto& vec_node = deg_map[static_cast<std::size_t>(deg)];
        for (std::size_t idx = vec_node.size(); idx > 0U; --idx) {
            const NodeId argmax_idx = vec_node[idx - 1U];
            const Count curr_deg = coverage[argmax_idx];
            if (deg > curr_deg) {
                deg_map[static_cast<std::size_t>(curr_deg)].push_back(argmax_idx);
                continue;
            }

            Count topk = k;
            Count deg_bound = deg;
            std::vector<Count> vec_bound(static_cast<std::size_t>(k), 0U);

            std::size_t idx_bound = idx;
            while (topk > 0U && idx_bound > 0U) {
                --idx_bound;
                const NodeId node = deg_map[static_cast<std::size_t>(deg_bound)][idx_bound];
                vec_bound[static_cast<std::size_t>(--topk)] = coverage[node];
            }
            while (topk > 0U && deg_bound > 1U) {
                --deg_bound;
                idx_bound = deg_map[static_cast<std::size_t>(deg_bound)].size();
                while (topk > 0U && idx_bound > 0U) {
                    --idx_bound;
                    const NodeId node = deg_map[static_cast<std::size_t>(deg_bound)][idx_bound];
                    vec_bound[static_cast<std::size_t>(--topk)] = coverage[node];
                }
            }

            make_min_heap(&vec_bound);
            bool flag = (topk == 0U);
            while (flag && idx_bound > 0U) {
                --idx_bound;
                const NodeId node = deg_map[static_cast<std::size_t>(deg_bound)][idx_bound];
                const Count curr_deg_bound = coverage[node];
                if (vec_bound[0] >= deg_bound) {
                    flag = false;
                } else if (vec_bound[0] < curr_deg_bound) {
                    min_heap_replace_min_value(&vec_bound, curr_deg_bound);
                }
            }
            while (flag && deg_bound > 1U) {
                --deg_bound;
                idx_bound = deg_map[static_cast<std::size_t>(deg_bound)].size();
                while (flag && idx_bound > 0U) {
                    --idx_bound;
                    const NodeId node = deg_map[static_cast<std::size_t>(deg_bound)][idx_bound];
                    const Count curr_deg_bound = coverage[node];
                    if (vec_bound[0] >= deg_bound) {
                        flag = false;
                    } else if (vec_bound[0] < curr_deg_bound) {
                        min_heap_replace_min_value(&vec_bound, curr_deg_bound);
                    }
                }
            }

            const Count bound_sum =
                std::accumulate(vec_bound.begin(), vec_bound.end(), Count{0U}) + sum_inf;
            result.bound_last = static_cast<double>(bound_sum) * static_cast<double>(num_nodes) /
                                static_cast<double>(rr_sets.size());
            result.bound_min = std::min(result.bound_min, result.bound_last);

            if (result.seeds.size() >= k) {
                result.covered_rr = sum_inf;
                return result;
            }

            sum_inf += curr_deg;
            result.seeds.push_back(argmax_idx);
            coverage[argmax_idx] = 0U;
            for (Count rid : node_to_rr[argmax_idx]) {
                auto& mark = rr_mark[static_cast<std::size_t>(rid)];
                if (mark != 0U) {
                    continue;
                }
                mark = 1U;
                for (NodeId node_idx : rr_sets[static_cast<std::size_t>(rid)].nodes) {
                    if (coverage[node_idx] == 0U) {
                        continue;
                    }
                    --coverage[node_idx];
                }
            }
        }
        deg_map.pop_back();
    }

    result.covered_rr = sum_inf;
    return result;
}

MaxCoverResult max_cover_topk(const std::vector<std::vector<Count>>& node_to_rr,
                              const std::vector<RRSet>& rr_sets,
                              NodeId num_nodes,
                              Budget k) {
    MaxCoverResult result;
    if (k == 0U || num_nodes == 0U || rr_sets.empty()) {
        return result;
    }

    const std::size_t k_size = static_cast<std::size_t>(k);
    std::vector<Count> coverage(num_nodes, 0U);
    Count max_deg = 0U;
    for (NodeId v = 0; v < num_nodes; ++v) {
        const Count deg = static_cast<Count>(node_to_rr[v].size());
        coverage[v] = deg;
        if (deg > max_deg) {
            max_deg = deg;
        }
    }

    std::vector<std::vector<NodeId>> deg_map(static_cast<std::size_t>(max_deg + 1U));
    for (NodeId v = 0; v < num_nodes; ++v) {
        deg_map[static_cast<std::size_t>(coverage[v])].push_back(v);
    }

    std::vector<NodeId> sorted_node(num_nodes, kInvalidNodeId);
    std::vector<std::size_t> node_position(num_nodes, 0U);
    std::vector<std::size_t> degree_position(static_cast<std::size_t>(max_deg + 2U), 0U);

    std::size_t idx_sort = 0U;
    for (Count deg = 0U; deg <= max_deg; ++deg) {
        const auto& nodes = deg_map[static_cast<std::size_t>(deg)];
        degree_position[static_cast<std::size_t>(deg + 1U)] =
            degree_position[static_cast<std::size_t>(deg)] + nodes.size();
        for (NodeId node : nodes) {
            node_position[node] = idx_sort;
            sorted_node[idx_sort] = node;
            ++idx_sort;
        }
    }

    std::vector<std::uint8_t> rr_mark(rr_sets.size(), 0U);

    Count sum_topk = 0U;
    const std::size_t boundary = static_cast<std::size_t>(num_nodes) - k_size;
    for (Count deg = max_deg + 1U; deg > 0U; --deg) {
        const Count curr_deg = deg - 1U;
        const std::size_t start = degree_position[static_cast<std::size_t>(curr_deg)];
        const std::size_t end = degree_position[static_cast<std::size_t>(curr_deg + 1U)];
        if (start <= boundary) {
            if (end > boundary) {
                sum_topk += curr_deg * static_cast<Count>(end - boundary);
            }
            break;
        }
        sum_topk += curr_deg * static_cast<Count>(end - start);
    }

    result.bound_min = static_cast<double>(sum_topk);
    Count sum_inf = 0U;
    result.seeds.reserve(k);

    for (Budget step = 0U; step < k; ++step) {
        if (sorted_node.empty()) {
            break;
        }

        const NodeId seed = sorted_node.back();
        sorted_node.pop_back();
        const std::size_t new_num_v = sorted_node.size();

        if (new_num_v >= k_size) {
            const std::size_t idx = new_num_v - k_size;
            sum_topk += coverage[sorted_node[idx]] - coverage[seed];
        } else {
            sum_topk = (sum_topk > coverage[seed]) ? (sum_topk - coverage[seed]) : 0U;
        }

        sum_inf += coverage[seed];
        result.seeds.push_back(seed);
        coverage[seed] = 0U;

        const std::size_t topk_start = (new_num_v > k_size) ? (new_num_v - k_size) : 0U;
        for (Count rid : node_to_rr[seed]) {
            if (rid >= rr_sets.size()) {
                throw std::invalid_argument("RR index out of range in node_to_rr");
            }
            auto& mark = rr_mark[static_cast<std::size_t>(rid)];
            if (mark != 0U) {
                continue;
            }
            mark = 1U;

            for (NodeId node_idx : rr_sets[static_cast<std::size_t>(rid)].nodes) {
                if (node_idx >= num_nodes) {
                    throw std::invalid_argument("RR-set contains node out of range");
                }
                if (coverage[node_idx] == 0U) {
                    continue;
                }

                const std::size_t curr_pos = node_position[node_idx];
                const Count curr_deg = coverage[node_idx];
                const std::size_t start_pos = degree_position[static_cast<std::size_t>(curr_deg)];
                if (curr_pos >= sorted_node.size() || start_pos >= sorted_node.size()) {
                    throw std::logic_error("topk max-cover position out of range");
                }

                const NodeId start_node = sorted_node[start_pos];
                std::swap(sorted_node[curr_pos], sorted_node[start_pos]);
                node_position[node_idx] = start_pos;
                node_position[start_node] = curr_pos;
                degree_position[static_cast<std::size_t>(curr_deg)]++;
                --coverage[node_idx];

                if (start_pos >= topk_start && sum_topk > 0U) {
                    --sum_topk;
                }
            }
        }

        result.bound_last = static_cast<double>(sum_inf + sum_topk);
        result.bound_min = std::min(result.bound_min, result.bound_last);
    }

    const double scale = static_cast<double>(num_nodes) / static_cast<double>(rr_sets.size());
    result.bound_min *= scale;
    result.bound_last *= scale;
    result.covered_rr = sum_inf;
    return result;
}

MaxCoverResult max_cover(const std::vector<std::vector<Count>>& node_to_rr,
                         const std::vector<RRSet>& rr_sets,
                         NodeId num_nodes,
                         Budget k) {
    if (k >= 1000U) {
        return max_cover_topk(node_to_rr, rr_sets, num_nodes, k);
    }
    return max_cover_lazy(node_to_rr, rr_sets, num_nodes, k);
}

Count doubled_rr_target(Count base, Count idx) {
    if (idx >= std::numeric_limits<Count>::digits) {
        return std::numeric_limits<Count>::max();
    }
    if (base > (std::numeric_limits<Count>::max() >> idx)) {
        return std::numeric_limits<Count>::max();
    }
    return base << idx;
}

}  // namespace

ICGraph::ICGraph(NodeId n) : n_(n) {
    in_edges_.resize(n_);
    out_edges_.resize(n_);
}

NodeId ICGraph::num_nodes() const {
    return n_;
}

EdgeId ICGraph::num_edges() const {
    if (edges_.size() >= static_cast<std::size_t>(kInvalidEdgeId)) {
        throw std::overflow_error("ICGraph edges exceed EdgeId range");
    }
    return static_cast<EdgeId>(edges_.size());
}

EdgeId ICGraph::add_edge(NodeId u, NodeId v, Prob p) {
    validate_node(u);
    validate_node(v);
    validate_probability(p);
    if (edges_.size() >= static_cast<std::size_t>(kInvalidEdgeId)) {
        throw std::overflow_error("cannot add more IC edges: EdgeId overflow");
    }

    const EdgeId id = static_cast<EdgeId>(edges_.size());
    edges_.push_back(DirectedEdge{u, v, p});
    return id;
}

void ICGraph::build_indices() {
    in_edges_.assign(n_, {});
    out_edges_.assign(n_, {});

    for (std::size_t i = 0; i < edges_.size(); ++i) {
        const EdgeId eid = static_cast<EdgeId>(i);
        const auto& e = edges_[i];
        out_edges_[e.src].push_back(eid);
        in_edges_[e.dst].push_back(eid);
    }
}

void ICGraph::validate() const {
    if (n_ == kInvalidNodeId) {
        throw std::invalid_argument("ICGraph num_nodes cannot be kInvalidNodeId");
    }

    for (const auto& e : edges_) {
        if (e.src >= n_ || e.dst >= n_) {
            throw std::invalid_argument("IC edge endpoint out of range");
        }
        validate_probability(e.prob);
    }

    if (in_edges_.size() != n_ || out_edges_.size() != n_) {
        throw std::invalid_argument("ICGraph indices size mismatch; call build_indices()");
    }

    for (NodeId v = 0; v < n_; ++v) {
        for (EdgeId eid : in_edges_[v]) {
            if (eid >= edges_.size()) {
                throw std::invalid_argument("IC in-edge index out of range");
            }
            if (edges_[eid].dst != v) {
                throw std::invalid_argument("IC in-edge index inconsistent with destination");
            }
        }
        for (EdgeId eid : out_edges_[v]) {
            if (eid >= edges_.size()) {
                throw std::invalid_argument("IC out-edge index out of range");
            }
            if (edges_[eid].src != v) {
                throw std::invalid_argument("IC out-edge index inconsistent with source");
            }
        }
    }
}

const DirectedEdge& ICGraph::edge(EdgeId id) const {
    if (id >= edges_.size()) {
        throw std::out_of_range("IC edge id out of range");
    }
    return edges_[id];
}

const std::vector<EdgeId>& ICGraph::in_edges(NodeId v) const {
    validate_node(v);
    return in_edges_[v];
}

const std::vector<EdgeId>& ICGraph::out_edges(NodeId u) const {
    validate_node(u);
    return out_edges_[u];
}

void ICGraph::validate_probability(Prob p) {
    if (p < 0.0 || p > 1.0) {
        throw std::invalid_argument("IC edge probability must be in [0,1]");
    }
}

void ICGraph::validate_node(NodeId v) const {
    if (v >= n_) {
        throw std::out_of_range("IC node id out of range");
    }
}

RRSampler::RRSampler(const ICGraph& graph) : graph_(graph) {}

RRSet RRSampler::sample(Random& random) const {
    if (graph_.num_nodes() == 0U) {
        throw std::invalid_argument("cannot sample RR-set from empty graph");
    }
    const NodeId root = random.uniform_node(0U, graph_.num_nodes() - 1U);
    return sample_with_root(root, random);
}

RRSet RRSampler::sample_with_root(NodeId root, Random& random) const {
    if (root >= graph_.num_nodes()) {
        throw std::out_of_range("RR root out of range");
    }

    RRSet rr;
    rr.root = root;

    std::vector<std::uint8_t> visited(graph_.num_nodes(), 0U);
    std::queue<NodeId> queue;
    queue.push(root);
    visited[root] = 1U;
    rr.nodes.push_back(root);

    while (!queue.empty()) {
        const NodeId v = queue.front();
        queue.pop();

        for (EdgeId eid : graph_.in_edges(v)) {
            const auto& e = graph_.edge(eid);
            if (visited[e.src] != 0U) {
                continue;
            }
            if (!random.bernoulli(e.prob)) {
                continue;
            }
            visited[e.src] = 1U;
            queue.push(e.src);
            rr.nodes.push_back(e.src);
        }
    }

    return rr;
}

std::vector<NodeId> CoverageGreedy::select(const std::vector<RRSet>& rr_sets,
                                           NodeId num_nodes,
                                           Budget k) const {
    std::vector<NodeId> seeds;
    if (k == 0U || num_nodes == 0U) {
        return seeds;
    }

    seeds.reserve(k);
    std::vector<std::vector<Count>> node_to_rr(num_nodes);
    for (Count rid = 0; rid < rr_sets.size(); ++rid) {
        for (NodeId node : rr_sets[static_cast<std::size_t>(rid)].nodes) {
            if (node >= num_nodes) {
                throw std::invalid_argument("RR-set contains node out of range");
            }
            node_to_rr[node].push_back(rid);
        }
    }

    std::vector<std::uint8_t> chosen(num_nodes, 0U);
    std::vector<std::uint8_t> covered(rr_sets.size(), 0U);

    for (Budget step = 0; step < k; ++step) {
        NodeId best = kInvalidNodeId;
        Count best_gain = 0;

        for (NodeId v = 0; v < num_nodes; ++v) {
            if (chosen[v] != 0U) {
                continue;
            }

            Count gain = 0;
            for (Count rid : node_to_rr[v]) {
                if (covered[static_cast<std::size_t>(rid)] == 0U) {
                    ++gain;
                }
            }

            if (best == kInvalidNodeId || gain > best_gain || (gain == best_gain && v < best)) {
                best = v;
                best_gain = gain;
            }
        }

        if (best == kInvalidNodeId) {
            break;
        }

        chosen[best] = 1U;
        seeds.push_back(best);
        for (Count rid : node_to_rr[best]) {
            covered[static_cast<std::size_t>(rid)] = 1U;
        }
    }

    return seeds;
}

OPIMProxyResult OPIMProxy::run(const ICGraph& graph,
                               Budget k,
                               double epsilon,
                               double delta,
                               std::uint64_t seed,
                               NodeId active_node_count) const {
    if (k == 0U) {
        throw std::invalid_argument("OPIMProxy k must be > 0");
    }
    if (graph.num_nodes() == 0U) {
        throw std::invalid_argument("OPIMProxy cannot run on empty graph");
    }
    if (epsilon <= 0.0 || epsilon >= 1.0) {
        throw std::invalid_argument("OPIMProxy epsilon must be in (0,1)");
    }
    if (delta <= 0.0 || delta >= 1.0) {
        throw std::invalid_argument("OPIMProxy delta must be in (0,1)");
    }
    const NodeId n_nodes = (active_node_count == 0U) ? graph.num_nodes() : active_node_count;
    if (n_nodes == 0U || n_nodes > graph.num_nodes()) {
        throw std::invalid_argument("OPIMProxy active_node_count must be in [1, graph.num_nodes()]");
    }
    if (k > n_nodes) {
        throw std::invalid_argument("OPIMProxy k must be <= active node count");
    }
    const bool enable_progress_log = (active_node_count != 0U);
    const auto t_begin = SteadyClock::now();
    Random random(seed);
    const double n = static_cast<double>(n_nodes);
    const double approx = 1.0 - 1.0 / std::exp(1.0);
    const double alpha = std::sqrt(std::log(6.0 / delta));
    const double beta = std::sqrt(approx * (logcnk(n_nodes, k) + std::log(6.0 / delta)));
    const double term = approx * alpha + beta;

    Count num_r_base = static_cast<Count>(2.0 * pow2(term));
    if (num_r_base == 0U) {
        num_r_base = 1U;
    }
    Count max_num_r = static_cast<Count>(2.0 * n * pow2(term) /
                                         (static_cast<double>(k) * pow2(epsilon))) + 1U;
    if (max_num_r < num_r_base) {
        max_num_r = num_r_base;
    }

    const Count ratio = std::max<Count>(Count{1U}, max_num_r / num_r_base);
    const Count num_iter = static_cast<Count>(std::log2(static_cast<double>(ratio))) + 1U;
    const double a1 = std::log(static_cast<double>(num_iter) * 3.0 / delta);
    const double a2 = std::log(static_cast<double>(num_iter) * 3.0 / delta);

    std::vector<RRSet> rr_sets_r1;
    std::vector<RRSet> rr_sets_r2;
    std::vector<std::vector<Count>> node_to_rr_r1(n_nodes);
    std::vector<std::vector<Count>> node_to_rr_r2(n_nodes);

    OPIMProxyResult best;
    best.upper_bound = n / approx;
    if (enable_progress_log) {
        std::cout << "[opim-c] started: k=" << k
                  << ", epsilon=" << epsilon
                  << ", delta=" << delta
                  << ", rr_base=" << num_r_base
                  << ", rr_max=" << max_num_r
                  << ", iter_max=" << num_iter
                  << ", total_elapsed_s=" << runtime_elapsed_seconds()
                  << std::endl;
    }
    for (Count iter = 0U; iter < num_iter; ++iter) {
        const auto t_iter = SteadyClock::now();
        const Count target_num_r = std::min(max_num_r, doubled_rr_target(num_r_base, iter));
        if (enable_progress_log) {
            std::cout << "[opim-c] iter started: " << (iter + 1U) << "/" << num_iter
                      << ", target_rr=" << target_num_r
                      << ", total_elapsed_s=" << runtime_elapsed_seconds()
                      << std::endl;
        }
        append_rr_sets_until_projected_with_progress(graph,
                                                     random,
                                                     target_num_r,
                                                     n_nodes,
                                                     n_nodes,
                                                     &rr_sets_r1,
                                                     &node_to_rr_r1,
                                                     enable_progress_log,
                                                     iter + 1U,
                                                     num_iter,
                                                     "R1",
                                                     t_iter);
        append_rr_sets_until_projected_with_progress(graph,
                                                     random,
                                                     target_num_r,
                                                     n_nodes,
                                                     n_nodes,
                                                     &rr_sets_r2,
                                                     &node_to_rr_r2,
                                                     enable_progress_log,
                                                     iter + 1U,
                                                     num_iter,
                                                     "R2",
                                                     t_iter);

        const Count num_r = static_cast<Count>(rr_sets_r1.size());
        const auto cover_result = max_cover(node_to_rr_r1, rr_sets_r1, n_nodes, k);

        const Count vldt_covered = covered_rr_count(node_to_rr_r2, cover_result.seeds, num_r);
        const double inf_vldt = static_cast<double>(vldt_covered) * n / static_cast<double>(num_r);
        const double deg_vldt = inf_vldt * static_cast<double>(num_r) / n;

        const double upper_bound = cover_result.bound_min;

        const double upper_deg_opt = upper_bound * static_cast<double>(num_r) / n;
        const double lower_select = pow2(std::sqrt(deg_vldt + a1 * 2.0 / 9.0) - std::sqrt(a1 / 2.0))
                                    - a1 / 18.0;
        const double upper_opt = pow2(std::sqrt(upper_deg_opt + a2 / 2.0) + std::sqrt(a2 / 2.0));
        const double approx_opimc = (upper_opt > 0.0) ? (lower_select / upper_opt) : -1.0;

        best.seeds = cover_result.seeds;
        best.upper_bound = upper_bound;
        best.rr_samples = num_r * 2U;

        if (enable_progress_log) {
            const double approx_target = approx - epsilon;
            std::cout << "[opim-c] iter done: " << (iter + 1U) << "/" << num_iter
                      << ", rr_samples=" << best.rr_samples
                      << ", upper_bound=" << upper_bound
                      << ", approx=" << approx_opimc
                      << ", target=" << approx_target
                      << ", elapsed_s=" << elapsed_seconds(t_iter, SteadyClock::now())
                      << ", total_elapsed_s=" << runtime_elapsed_seconds()
                      << std::endl;
        }

        if (approx_opimc >= approx - epsilon) {
            if (enable_progress_log) {
                std::cout << "[opim-c] converged: iter=" << (iter + 1U)
                          << ", rr_samples=" << best.rr_samples
                          << ", elapsed_s=" << elapsed_seconds(t_begin, SteadyClock::now())
                          << ", total_elapsed_s=" << runtime_elapsed_seconds()
                          << std::endl;
            }
            return best;
        }
    }

    if (best.seeds.empty()) {
        best.seeds.push_back(0U);
        best.rr_samples = static_cast<Count>(rr_sets_r1.size() * 2U);
        best.upper_bound = n / approx;
    }
    if (enable_progress_log) {
        std::cout << "[opim-c] done: rr_samples=" << best.rr_samples
                  << ", upper_bound=" << best.upper_bound
                  << ", elapsed_s=" << elapsed_seconds(t_begin, SteadyClock::now())
                  << ", total_elapsed_s=" << runtime_elapsed_seconds()
                  << std::endl;
    }
    return best;
}

OPIMProxyResult run_opim_proxy_clique_implicit(const HybridHypergraph& graph,
                                               Budget k,
                                               double epsilon,
                                               double delta,
                                               std::uint64_t seed,
                                               NodeId active_node_count) {
    if (k == 0U) {
        throw std::invalid_argument("run_opim_proxy_clique_implicit k must be > 0");
    }
    if (graph.num_nodes() == 0U) {
        throw std::invalid_argument("run_opim_proxy_clique_implicit cannot run on empty graph");
    }
    if (epsilon <= 0.0 || epsilon >= 1.0) {
        throw std::invalid_argument("run_opim_proxy_clique_implicit epsilon must be in (0,1)");
    }
    if (delta <= 0.0 || delta >= 1.0) {
        throw std::invalid_argument("run_opim_proxy_clique_implicit delta must be in (0,1)");
    }

    const NodeId n_nodes = (active_node_count == 0U) ? graph.num_nodes() : active_node_count;
    if (n_nodes == 0U || n_nodes > graph.num_nodes()) {
        throw std::invalid_argument(
            "run_opim_proxy_clique_implicit active_node_count must be in [1, graph.num_nodes()]");
    }
    if (k > n_nodes) {
        throw std::invalid_argument("run_opim_proxy_clique_implicit k must be <= active node count");
    }

    Random random(seed);
    const double n = static_cast<double>(n_nodes);
    const double approx = 1.0 - 1.0 / std::exp(1.0);
    const double alpha = std::sqrt(std::log(6.0 / delta));
    const double beta = std::sqrt(approx * (logcnk(n_nodes, k) + std::log(6.0 / delta)));
    const double term = approx * alpha + beta;

    Count num_r_base = static_cast<Count>(2.0 * pow2(term));
    if (num_r_base == 0U) {
        num_r_base = 1U;
    }
    Count max_num_r = static_cast<Count>(2.0 * n * pow2(term) /
                                         (static_cast<double>(k) * pow2(epsilon))) + 1U;
    if (max_num_r < num_r_base) {
        max_num_r = num_r_base;
    }

    const Count ratio = std::max<Count>(Count{1U}, max_num_r / num_r_base);
    const Count num_iter = static_cast<Count>(std::log2(static_cast<double>(ratio))) + 1U;
    const double a1 = std::log(static_cast<double>(num_iter) * 3.0 / delta);
    const double a2 = std::log(static_cast<double>(num_iter) * 3.0 / delta);

    std::vector<RRSet> rr_sets_r1;
    std::vector<RRSet> rr_sets_r2;
    std::vector<std::vector<Count>> node_to_rr_r1(n_nodes);
    std::vector<std::vector<Count>> node_to_rr_r2(n_nodes);

    OPIMProxyResult best;
    best.upper_bound = n / approx;
    for (Count iter = 0U; iter < num_iter; ++iter) {
        const Count target_num_r = std::min(max_num_r, doubled_rr_target(num_r_base, iter));
        append_rr_sets_until_projected_clique_implicit(
            graph, random, target_num_r, n_nodes, n_nodes, &rr_sets_r1, &node_to_rr_r1);
        append_rr_sets_until_projected_clique_implicit(
            graph, random, target_num_r, n_nodes, n_nodes, &rr_sets_r2, &node_to_rr_r2);

        const Count num_r = static_cast<Count>(rr_sets_r1.size());
        const MaxCoverResult cover_result = max_cover(node_to_rr_r1, rr_sets_r1, n_nodes, k);

        const Count vldt_covered = covered_rr_count(node_to_rr_r2, cover_result.seeds, num_r);
        const double inf_vldt = static_cast<double>(vldt_covered) * n / static_cast<double>(num_r);
        const double deg_vldt = inf_vldt * static_cast<double>(num_r) / n;

        const double upper_bound = cover_result.bound_min;
        const double upper_deg_opt = upper_bound * static_cast<double>(num_r) / n;
        const double lower_select = pow2(std::sqrt(deg_vldt + a1 * 2.0 / 9.0) - std::sqrt(a1 / 2.0))
                                    - a1 / 18.0;
        const double upper_opt = pow2(std::sqrt(upper_deg_opt + a2 / 2.0) + std::sqrt(a2 / 2.0));
        const double approx_opimc = (upper_opt > 0.0) ? (lower_select / upper_opt) : -1.0;

        best.seeds = cover_result.seeds;
        best.upper_bound = upper_bound;
        best.rr_samples = num_r * 2U;

        if (approx_opimc >= approx - epsilon) {
            return best;
        }
    }

    if (best.seeds.empty()) {
        best.seeds.push_back(0U);
        best.rr_samples = static_cast<Count>(rr_sets_r1.size() * 2U);
        best.upper_bound = n / approx;
    }
    return best;
}

namespace {

struct HistGreedyResult {
    std::vector<NodeId> seeds;
    std::vector<Count> prefix_covered_rr;
    Count covered_rr = 0U;
    double bound_last = std::numeric_limits<double>::infinity();
    double bound_min = std::numeric_limits<double>::infinity();
};

struct EarlyStopEvalResult {
    double influence = 0.0;
    double avg_rr_size = 0.0;
    Count hits = 0U;
    Count samples = 0U;
};

double rr_average_size(const std::vector<RRSet>& rr_sets) {
    if (rr_sets.empty()) {
        return 0.0;
    }
    long double total = 0.0L;
    for (const auto& rr : rr_sets) {
        total += static_cast<long double>(rr.nodes.size());
    }
    return static_cast<double>(total / static_cast<long double>(rr_sets.size()));
}

double calculate_inf_early_stop(Count hits, Count rr_count, NodeId n_nodes) {
    if (rr_count == 0U) {
        return 0.0;
    }
    return static_cast<double>(hits) * static_cast<double>(n_nodes) / static_cast<double>(rr_count);
}

Count compute_num_iter(Count max_num_r, Count num_r_base) {
    if (num_r_base == 0U) {
        return 1U;
    }
    const Count ratio = std::max<Count>(Count{1U}, max_num_r / num_r_base);
    return static_cast<Count>(std::log2(static_cast<double>(ratio))) + 1U;
}

std::vector<std::uint8_t> make_node_mask(NodeId n_nodes,
                                         const std::vector<NodeId>& nodes,
                                         const char* tag) {
    std::vector<std::uint8_t> mask(n_nodes, 0U);
    for (NodeId v : nodes) {
        if (v >= n_nodes) {
            throw std::invalid_argument(std::string(tag) + " seed out of range");
        }
        mask[v] = 1U;
    }
    return mask;
}

int decide_multiple(double full_rr_size, double trunc_rr_size, Count num_rrsets) {
    if (num_rrsets < 100U || trunc_rr_size <= 0.0) {
        return 1;
    }
    const int ratio = static_cast<int>(full_rr_size / trunc_rr_size);
    if (ratio >= 32) {
        return 8;
    }
    if (ratio >= 16) {
        return 4;
    }
    if (ratio >= 4) {
        return 2;
    }
    return 1;
}

Count clique_out_degree_proxy(const HybridHypergraph& graph, NodeId v) {
    Count proxy = static_cast<Count>(graph.out_edges(v).size());
    for (HyperedgeId hid : graph.incident_hyperedges(v)) {
        const auto& h = graph.hyperedge(hid);
        proxy += static_cast<Count>(h.nodes.size() - 1U);
    }
    return proxy;
}

void reset_rr_index(NodeId n_nodes,
                    std::vector<RRSet>* rr_sets,
                    std::vector<std::vector<Count>>* node_to_rr,
                    Count* hit_sentinel_samples) {
    if (rr_sets == nullptr || node_to_rr == nullptr) {
        throw std::invalid_argument("RR containers cannot be null");
    }
    rr_sets->clear();
    node_to_rr->assign(n_nodes, {});
    if (hit_sentinel_samples != nullptr) {
        *hit_sentinel_samples = 0U;
    }
}

EarlyStopEvalResult eval_seed_set_inf_early_stop(const ICGraph& graph,
                                                 Random& random,
                                                 Count num_samples,
                                                 const std::vector<std::uint8_t>& seed_nodes,
                                                 NodeId root_node_count,
                                                 NodeId kept_node_count) {
    EarlyStopEvalResult result;
    result.samples = num_samples;
    if (num_samples == 0U) {
        return result;
    }
    if (seed_nodes.size() != kept_node_count) {
        throw std::invalid_argument("seed_nodes size mismatch in projected early-stop eval");
    }
    if (root_node_count == 0U || root_node_count > graph.num_nodes()) {
        throw std::invalid_argument("invalid root_node_count in projected early-stop eval");
    }
    if (kept_node_count == 0U || kept_node_count > graph.num_nodes()) {
        throw std::invalid_argument("invalid kept_node_count in projected early-stop eval");
    }
    long double total_rr_size = 0.0L;
    for (Count i = 0U; i < num_samples; ++i) {
        const NodeId root = random.uniform_node(0U, root_node_count - 1U);
        RRSetEarlyStopSample sample = sample_rr_set_with_sentinel_root_projected(
            graph, root, random, seed_nodes, kept_node_count);
        if (sample.hit_sentinel) {
            ++result.hits;
        }
        total_rr_size += static_cast<long double>(sample.rr.nodes.size());
    }
    result.influence = calculate_inf_early_stop(result.hits, num_samples, kept_node_count);
    result.avg_rr_size = static_cast<double>(total_rr_size / static_cast<long double>(num_samples));
    return result;
}

EarlyStopEvalResult eval_seed_set_inf_early_stop_clique_implicit(
    const HybridHypergraph& graph,
    Random& random,
    Count num_samples,
    const std::vector<std::uint8_t>& seed_nodes,
    NodeId root_node_count,
    NodeId kept_node_count) {
    EarlyStopEvalResult result;
    result.samples = num_samples;
    if (num_samples == 0U) {
        return result;
    }
    if (seed_nodes.size() != kept_node_count) {
        throw std::invalid_argument("seed_nodes size mismatch in projected implicit clique early-stop eval");
    }
    if (root_node_count == 0U || root_node_count > graph.num_nodes()) {
        throw std::invalid_argument("invalid root_node_count in projected implicit clique early-stop eval");
    }
    if (kept_node_count == 0U || kept_node_count > graph.num_nodes()) {
        throw std::invalid_argument("invalid kept_node_count in projected implicit clique early-stop eval");
    }
    long double total_rr_size = 0.0L;
    for (Count i = 0U; i < num_samples; ++i) {
        const NodeId root = random.uniform_node(0U, root_node_count - 1U);
        RRSetEarlyStopSample sample = sample_rr_set_with_sentinel_root_projected_clique_implicit(
            graph, root, random, seed_nodes, kept_node_count);
        if (sample.hit_sentinel) {
            ++result.hits;
        }
        total_rr_size += static_cast<long double>(sample.rr.nodes.size());
    }
    result.influence = calculate_inf_early_stop(result.hits, num_samples, kept_node_count);
    result.avg_rr_size = static_cast<double>(total_rr_size / static_cast<long double>(num_samples));
    return result;
}

HistGreedyResult max_cover_hist_lazy(const std::vector<std::vector<Count>>& node_to_rr,
                                     const std::vector<RRSet>& rr_sets,
                                     const std::vector<Count>& out_degree,
                                     NodeId num_nodes,
                                     Budget target_size,
                                     Budget bound_target_size,
                                     const std::vector<NodeId>& prefix,
                                     bool record_prefix_covered) {
    HistGreedyResult result;
    if (target_size == 0U || num_nodes == 0U || rr_sets.empty()) {
        return result;
    }
    if (node_to_rr.size() != num_nodes) {
        throw std::invalid_argument("node_to_rr size mismatch");
    }
    if (out_degree.size() != num_nodes) {
        throw std::invalid_argument("out_degree size mismatch");
    }

    std::vector<Count> coverage(num_nodes, 0U);
    Count max_deg = 0U;
    for (NodeId v = 0; v < num_nodes; ++v) {
        const Count deg = static_cast<Count>(node_to_rr[v].size());
        coverage[v] = deg;
        if (deg > max_deg) {
            max_deg = deg;
        }
    }

    std::vector<std::uint8_t> chosen(num_nodes, 0U);
    std::vector<std::uint8_t> rr_mark(rr_sets.size(), 0U);
    Count sum_inf = 0U;

    auto apply_seed = [&](NodeId seed) {
        if (seed >= num_nodes) {
            throw std::invalid_argument("seed out of range");
        }
        if (chosen[seed] != 0U) {
            return;
        }
        chosen[seed] = 1U;
        result.seeds.push_back(seed);
        sum_inf += coverage[seed];
        if (record_prefix_covered) {
            result.prefix_covered_rr.push_back(sum_inf);
        }
        coverage[seed] = 0U;

        for (Count rid : node_to_rr[seed]) {
            if (rid >= rr_sets.size()) {
                throw std::invalid_argument("RR index out of range in node_to_rr");
            }
            auto& mark = rr_mark[static_cast<std::size_t>(rid)];
            if (mark != 0U) {
                continue;
            }
            mark = 1U;
            for (NodeId node_idx : rr_sets[static_cast<std::size_t>(rid)].nodes) {
                if (node_idx >= num_nodes) {
                    throw std::invalid_argument("RR-set contains node out of range");
                }
                if (coverage[node_idx] == 0U) {
                    continue;
                }
                --coverage[node_idx];
            }
        }
    };

    result.seeds.reserve(target_size);
    for (NodeId v : prefix) {
        if (result.seeds.size() >= target_size) {
            break;
        }
        apply_seed(v);
    }
    if (result.seeds.size() >= target_size) {
        result.covered_rr = sum_inf;
        return result;
    }
    if (max_deg == 0U) {
        result.covered_rr = sum_inf;
        return result;
    }

    std::vector<std::vector<NodeId>> deg_map(static_cast<std::size_t>(max_deg + 1U));
    for (NodeId v = 0; v < num_nodes; ++v) {
        const Count deg = coverage[v];
        if (deg != 0U && chosen[v] == 0U) {
            deg_map[static_cast<std::size_t>(deg)].push_back(v);
        }
    }

    for (Count deg = max_deg; deg > 0U; --deg) {
        auto& orig_vec_node = deg_map[static_cast<std::size_t>(deg)];
        std::vector<std::pair<Count, NodeId>> vec_pair;
        vec_pair.reserve(orig_vec_node.size());

        for (std::size_t idx = orig_vec_node.size(); idx > 0U; --idx) {
            const NodeId node = orig_vec_node[idx - 1U];
            const Count node_coverage = coverage[node];

            if (deg > node_coverage) {
                deg_map[static_cast<std::size_t>(node_coverage)].push_back(node);
                continue;
            }
            if (chosen[node] != 0U || node_coverage == 0U) {
                continue;
            }
            vec_pair.emplace_back(out_degree[node], node);
        }

        deg_map.pop_back();
        if (vec_pair.empty()) {
            continue;
        }

        std::sort(vec_pair.begin(), vec_pair.end());
        std::vector<NodeId> vec_node;
        vec_node.reserve(vec_pair.size());
        for (const auto& node_pair : vec_pair) {
            vec_node.push_back(node_pair.second);
        }
        deg_map.push_back(std::move(vec_node));
        auto& sorted_vec_node = deg_map[static_cast<std::size_t>(deg)];

        for (std::size_t idx = sorted_vec_node.size(); idx > 0U; --idx) {
            const NodeId argmax_idx = sorted_vec_node[idx - 1U];
            const Count curr_deg = coverage[argmax_idx];
            if (deg > curr_deg) {
                deg_map[static_cast<std::size_t>(curr_deg)].push_back(argmax_idx);
                continue;
            }

            if (bound_target_size > 0U) {
                Count topk = bound_target_size;
                Count deg_bound = deg;
                std::vector<Count> vec_bound(static_cast<std::size_t>(bound_target_size), 0U);

                std::size_t idx_bound = idx;
                while (topk > 0U && idx_bound > 0U) {
                    --idx_bound;
                    const NodeId node = deg_map[static_cast<std::size_t>(deg_bound)][idx_bound];
                    vec_bound[static_cast<std::size_t>(--topk)] = coverage[node];
                }
                while (topk > 0U && deg_bound > 1U) {
                    --deg_bound;
                    idx_bound = deg_map[static_cast<std::size_t>(deg_bound)].size();
                    while (topk > 0U && idx_bound > 0U) {
                        --idx_bound;
                        const NodeId node = deg_map[static_cast<std::size_t>(deg_bound)][idx_bound];
                        vec_bound[static_cast<std::size_t>(--topk)] = coverage[node];
                    }
                }

                make_min_heap(&vec_bound);
                bool flag = (topk == 0U);
                while (flag && idx_bound > 0U) {
                    --idx_bound;
                    const NodeId node = deg_map[static_cast<std::size_t>(deg_bound)][idx_bound];
                    const Count curr_deg_bound = coverage[node];
                    if (vec_bound[0] >= deg_bound) {
                        flag = false;
                    } else if (vec_bound[0] < curr_deg_bound) {
                        min_heap_replace_min_value(&vec_bound, curr_deg_bound);
                    }
                }
                while (flag && deg_bound > 1U) {
                    --deg_bound;
                    idx_bound = deg_map[static_cast<std::size_t>(deg_bound)].size();
                    while (flag && idx_bound > 0U) {
                        --idx_bound;
                        const NodeId node = deg_map[static_cast<std::size_t>(deg_bound)][idx_bound];
                        const Count curr_deg_bound = coverage[node];
                        if (vec_bound[0] >= deg_bound) {
                            flag = false;
                        } else if (vec_bound[0] < curr_deg_bound) {
                            min_heap_replace_min_value(&vec_bound, curr_deg_bound);
                        }
                    }
                }

                const Count bound_sum =
                    std::accumulate(vec_bound.begin(), vec_bound.end(), Count{0U}) + sum_inf;
                result.bound_last =
                    static_cast<double>(bound_sum) * static_cast<double>(num_nodes) /
                    static_cast<double>(rr_sets.size());
                result.bound_min = std::min(result.bound_min, result.bound_last);
            }

            if (result.seeds.size() >= target_size) {
                result.covered_rr = sum_inf;
                return result;
            }

            apply_seed(argmax_idx);
        }

        deg_map.pop_back();
    }

    result.covered_rr = sum_inf;
    return result;
}

class HistRunner {
public:
    HistRunner(const ICGraph& graph,
               Budget k,
               double epsilon,
               double delta,
               std::uint64_t seed,
               NodeId active_node_count)
        : graph_(graph),
          k_(k),
          epsilon_(epsilon),
          delta_(delta),
          random_(seed ^ 0xbf58476d1ce4e5b9ULL),
          rr_sampler_(graph),
          n_nodes_((active_node_count == 0U) ? graph.num_nodes() : active_node_count),
          n_(static_cast<double>((active_node_count == 0U) ? graph.num_nodes() : active_node_count)),
          approx_(1.0 - 1.0 / std::exp(1.0)),
          base_num_rrsets_(std::max<Count>(1U, static_cast<Count>(3.0 * std::log(1.0 / delta)))) {
        if (n_nodes_ == 0U || n_nodes_ > graph_.num_nodes()) {
            throw std::invalid_argument("HISTProxy active_node_count must be in [1, graph.num_nodes()]");
        }
        out_degree_.resize(n_nodes_, 0U);
        for (NodeId v = 0; v < n_nodes_; ++v) {
            out_degree_[v] = static_cast<Count>(graph_.out_edges(v).size());
        }
        node_to_rr_r1_.assign(n_nodes_, {});
        node_to_rr_r2_.assign(n_nodes_, {});
    }

    OPIMProxyResult run() {
        const double epsilon_phase = std::max(1e-9, epsilon_ * 0.5);
        const double delta_phase = std::max(1e-12, delta_ * 0.5);

        std::vector<NodeId> sentinel = find_dynamic_sub(epsilon_phase, delta_phase);
        if (sentinel.empty()) {
            sentinel.push_back(0U);
        }
        if (sentinel.size() > k_) {
            sentinel.resize(k_);
        }

        if (sentinel.size() >= k_) {
            OPIMProxyResult result;
            result.seeds = sentinel;
            result.upper_bound = n_ / approx_;
            result.rr_samples = static_cast<Count>(rr_sets_r1_.size() * 2U);
            return result;
        }
        return find_rem_set(sentinel, epsilon_phase, epsilon_, delta_phase);
    }

private:
    std::vector<NodeId> find_dynamic_sub(double epsilon, double delta) {
        reset_rr_index(n_nodes_, &rr_sets_r1_, &node_to_rr_r1_, nullptr);
        reset_rr_index(n_nodes_, &rr_sets_r2_, &node_to_rr_r2_, &hit_sentinel_r2_);

        const double x = (k_ > 1U) ? (1.0 - 1.0 / static_cast<double>(k_)) : 0.0;
        Budget min_sub_size = 1U;
        if (k_ > 1U && x > 0.0 && x < 1.0) {
            const double numer = std::log(std::max(1e-12, 1.0 - epsilon));
            const double denom = std::log(x);
            if (std::isfinite(numer) && std::isfinite(denom) && denom != 0.0) {
                min_sub_size = static_cast<Budget>(std::ceil(numer / denom));
                if (min_sub_size == 0U) {
                    min_sub_size = 1U;
                }
                if (min_sub_size > k_) {
                    min_sub_size = k_;
                }
            }
        }

        const double alpha = std::sqrt(std::log(6.0 / delta));
        const double beta = std::sqrt(approx_ * (logcnk(n_nodes_, k_) + std::log(6.0 / delta)));
        Count max_num_r = static_cast<Count>(2.0 * n_ * pow2(approx_ * alpha + beta) /
                                             static_cast<double>(k_) / pow2(epsilon)) + 1U;
        if (max_num_r < base_num_rrsets_) {
            max_num_r = base_num_rrsets_;
        }

        const Count num_iter = compute_num_iter(max_num_r, base_num_rrsets_);
        const double a1 = std::log(static_cast<double>(num_iter) * 3.0 / delta);
        const double a2 = std::log(static_cast<double>(num_iter) * 6.0 / delta);

        std::vector<NodeId> sentinel;
        bool first_round = true;
        int multiple = 1;

        for (Count iter = 0U; iter < num_iter; ++iter) {
            const Count target_num_r = std::min(max_num_r, doubled_rr_target(base_num_rrsets_, iter));
            append_rr_sets_until_projected(
                graph_, random_, target_num_r, n_nodes_, n_nodes_, &rr_sets_r1_, &node_to_rr_r1_);

            const Count num_r = static_cast<Count>(rr_sets_r1_.size());
            Budget target_size = first_round ? (k_ / 4U) : (k_ / 8U);
            first_round = false;
            if (target_size == 0U) {
                target_size = 1U;
            }

            const HistGreedyResult cover = max_cover_hist_lazy(node_to_rr_r1_,
                                                               rr_sets_r1_,
                                                               out_degree_,
                                                               n_nodes_,
                                                               target_size,
                                                               k_,
                                                               {},
                                                               true);
            std::vector<NodeId> sentinel_candidates = cover.seeds;
            std::vector<Count> sentinel_prefix = cover.prefix_covered_rr;

            // Align with HIST source heuristic: if sentinel greedy is not yet in a
            // high-confidence regime, trim trailing low-contribution seeds and
            // refill by out-degree from trimmed candidates.
            if (!(sentinel_candidates.size() == target_size && num_r > 1000U)) {
                const Count threshold = static_cast<Count>(0.9 * static_cast<double>(cover.covered_rr));
                std::vector<NodeId> trimmed_nodes;
                while (sentinel_candidates.size() > 1U && sentinel_prefix.size() > 1U) {
                    const std::size_t i = sentinel_candidates.size() - 1U;
                    if (sentinel_prefix[i - 1U] >= threshold) {
                        trimmed_nodes.push_back(sentinel_candidates.back());
                        sentinel_candidates.pop_back();
                        sentinel_prefix.pop_back();
                    } else {
                        break;
                    }
                }
                std::sort(trimmed_nodes.begin(), trimmed_nodes.end(), [&](NodeId a, NodeId b) {
                    if (out_degree_[a] != out_degree_[b]) {
                        return out_degree_[a] > out_degree_[b];
                    }
                    return a < b;
                });
                std::vector<std::uint8_t> in_sentinel(n_nodes_, 0U);
                for (NodeId v : sentinel_candidates) {
                    in_sentinel[v] = 1U;
                }
                for (NodeId v : trimmed_nodes) {
                    if (sentinel_candidates.size() >= target_size) {
                        break;
                    }
                    if (in_sentinel[v] != 0U) {
                        continue;
                    }
                    in_sentinel[v] = 1U;
                    sentinel_candidates.push_back(v);
                    sentinel_prefix.push_back(cover.covered_rr);
                }
            }

            if (sentinel_candidates.size() < target_size || sentinel_prefix.empty()) {
                continue;
            }

            int last_pos = 0;
            bool found = false;
            for (int i = static_cast<int>(sentinel_prefix.size()) - 1; i >= 0; --i) {
                const double lower_deg = static_cast<double>(sentinel_prefix[static_cast<std::size_t>(i)]);
                const double upper_deg = cover.bound_min;
                const double a = std::log(static_cast<double>(num_iter) * 6.0 / delta);
                const double lower =
                    pow2(std::sqrt(lower_deg + a * 2.0 / 9.0) - std::sqrt(a / 2.0)) - a / 18.0;
                double upper =
                    pow2(std::sqrt(upper_deg + a1 / 2.0) + std::sqrt(a1 / 2.0));
                upper = std::min(upper, static_cast<double>(num_r));
                const double vec_approx = (upper > 0.0) ? (lower / upper) : -1.0;
                const double calc_approx =
                    (1.0 - std::pow(x, static_cast<double>(i + 1)) - epsilon) * 1.2;
                if (vec_approx > calc_approx) {
                    found = true;
                    last_pos = i;
                    break;
                }
            }
            if (!found) {
                last_pos = 0;
            }

            Count set_size = static_cast<Count>(last_pos + 1);
            set_size = std::max<Count>(set_size, 10U);
            set_size = std::min<Count>(set_size, static_cast<Count>(target_size));
            set_size = std::max<Count>(set_size, static_cast<Count>(min_sub_size));
            set_size = std::min<Count>(set_size, static_cast<Count>(sentinel_candidates.size()));
            sentinel.assign(sentinel_candidates.begin(),
                            sentinel_candidates.begin() + static_cast<std::size_t>(set_size));

            const auto sentinel_nodes = make_node_mask(n_nodes_, sentinel, "HISTProxy sentinel");
            reset_rr_index(n_nodes_, &rr_sets_r2_, &node_to_rr_r2_, &hit_sentinel_r2_);
            const Count target_vldt = num_r * static_cast<Count>(multiple);
            append_rr_sets_until_with_sentinel_projected(graph_,
                                                         random_,
                                                         target_vldt,
                                                         n_nodes_,
                                                         n_nodes_,
                                                         sentinel_nodes,
                                                         &rr_sets_r2_,
                                                         &node_to_rr_r2_,
                                                         &hit_sentinel_r2_);

            const double inf_vldt =
                calculate_inf_early_stop(hit_sentinel_r2_, static_cast<Count>(rr_sets_r2_.size()), n_nodes_);
            const double deg_vldt = inf_vldt * static_cast<double>(multiple) *
                                    static_cast<double>(num_r) / n_;
            double upper_bound = cover.bound_min;
            if (!(upper_bound > 0.0) || !std::isfinite(upper_bound)) {
                upper_bound = n_ / approx_;
            }

            const double upper_deg_opt = upper_bound * static_cast<double>(num_r) / n_;
            double lower_select =
                (pow2(std::sqrt(deg_vldt + a2 * 2.0 / 9.0) - std::sqrt(a2 / 2.0)) - a2 / 18.0) /
                static_cast<double>(multiple);
            if (lower_select < 0.0) {
                lower_select =
                    static_cast<double>(sentinel.size()) / n_ * static_cast<double>(num_r) *
                    static_cast<double>(multiple);
            }

            double upper_opt = pow2(std::sqrt(upper_deg_opt + a1 / 2.0) + std::sqrt(a1 / 2.0));
            upper_opt = std::min(upper_opt, static_cast<double>(num_r));
            const double curr_approx = (upper_opt > 0.0) ? (lower_select / upper_opt) : -1.0;
            const double approx_sub = (x > 0.0) ? (1.0 - std::pow(x, static_cast<double>(sentinel.size()))) : 1.0;
            const double target_approx = approx_sub - epsilon;

            if (curr_approx >= target_approx) {
                return sentinel;
            }
            if (num_r < 100U) {
                continue;
            }

            const double full_rr_size = rr_average_size(rr_sets_r1_);
            const double trunc_rr_size = rr_average_size(rr_sets_r2_);
            if (trunc_rr_size <= 0.0 || (full_rr_size / trunc_rr_size) < 2.0) {
                continue;
            }

            const double lower_threshold =
                (upper_opt * n_ / static_cast<double>(num_r)) * target_approx;
            if ((inf_vldt / static_cast<double>(multiple)) > lower_threshold && lower_threshold > 0.0) {
                const double new_approx =
                    increase_r2(sentinel_nodes, a2, upper_opt, target_approx);
                if (new_approx > target_approx) {
                    return sentinel;
                }
            }
        }
        return sentinel;
    }

    double increase_r2(const std::vector<std::uint8_t>& sentinel_nodes,
                       double a,
                       double upper_opt,
                       double target_approx) {
        const Count vldt_rrsets = static_cast<Count>(rr_sets_r2_.size());
        const Count r1_rrsets = static_cast<Count>(rr_sets_r1_.size());
        if (vldt_rrsets == 0U || r1_rrsets == 0U || upper_opt <= 0.0) {
            return 0.0;
        }

        const int multiple = 4;
        const double inf_vldt = calculate_inf_early_stop(hit_sentinel_r2_, vldt_rrsets, n_nodes_);
        const double deg_vldt = inf_vldt * static_cast<double>(vldt_rrsets) / n_;
        const double lower_select =
            pow2(std::sqrt(deg_vldt * static_cast<double>(multiple) + a * 2.0 / 9.0) -
                 std::sqrt(a / 2.0)) - a / 18.0;
        const double estimate_approx =
            (lower_select / (static_cast<double>(multiple) * static_cast<double>(vldt_rrsets))) /
            (upper_opt / static_cast<double>(r1_rrsets));
        if (estimate_approx < target_approx) {
            return 0.0;
        }

        Count target_rrsets = vldt_rrsets;
        if (target_rrsets <= (std::numeric_limits<Count>::max() / static_cast<Count>(multiple))) {
            target_rrsets *= static_cast<Count>(multiple);
        } else {
            target_rrsets = std::numeric_limits<Count>::max();
        }

            append_rr_sets_until_with_sentinel_projected(graph_,
                                                         random_,
                                                         target_rrsets,
                                                         n_nodes_,
                                                         n_nodes_,
                                                         sentinel_nodes,
                                                         &rr_sets_r2_,
                                                         &node_to_rr_r2_,
                                                         &hit_sentinel_r2_);

        const Count new_vldt_rrsets = static_cast<Count>(rr_sets_r2_.size());
        if (new_vldt_rrsets == 0U) {
            return 0.0;
        }
        const double new_inf_vldt =
            calculate_inf_early_stop(hit_sentinel_r2_, new_vldt_rrsets, n_nodes_);
        const double new_deg_vldt = new_inf_vldt * static_cast<double>(new_vldt_rrsets) / n_;
        const double new_lower_select =
            pow2(std::sqrt(new_deg_vldt + a * 2.0 / 9.0) - std::sqrt(a / 2.0)) - a / 18.0;
        const double new_approx =
            (new_lower_select / static_cast<double>(new_vldt_rrsets)) /
            (upper_opt / static_cast<double>(r1_rrsets));
        return (new_approx > target_approx) ? new_approx : 0.0;
    }

    OPIMProxyResult find_rem_set(const std::vector<NodeId>& sentinel,
                                 double epsilon,
                                 double target_epsilon,
                                 double delta) {
        reset_rr_index(n_nodes_, &rr_sets_r1_, &node_to_rr_r1_, nullptr);

        const Count sub_seed_size = static_cast<Count>(sentinel.size());
        const double delta_upper = delta / 3.0;
        const double alpha = std::sqrt(std::log(3.0 / delta_upper));
        const double beta = std::sqrt(
            approx_ *
            (logcnk(n_nodes_, k_ - static_cast<Budget>(sub_seed_size)) + std::log(3.0 / delta_upper)));

        Count max_num_r = static_cast<Count>(2.0 * n_ * pow2(alpha + beta) /
                                             static_cast<double>(k_) / pow2(epsilon)) + 1U;
        if (max_num_r < base_num_rrsets_) {
            max_num_r = base_num_rrsets_;
        }
        const Count num_iter = compute_num_iter(max_num_r, base_num_rrsets_);
        const double a2 = std::log(static_cast<double>(num_iter) * 3.0 / delta);

        OPIMProxyResult best;
        best.seeds = sentinel;
        best.upper_bound = n_ / approx_;
        best.rr_samples = 0U;

        int multiple = 1;
        const auto sentinel_nodes = make_node_mask(n_nodes_, sentinel, "HISTProxy sentinel");

        for (Count iter = 0U; iter < num_iter; ++iter) {
            const Count target_num_r = std::min(max_num_r, doubled_rr_target(base_num_rrsets_, iter));
            append_rr_sets_until_with_sentinel_projected(graph_,
                                                         random_,
                                                         target_num_r,
                                                         n_nodes_,
                                                         n_nodes_,
                                                         sentinel_nodes,
                                                         &rr_sets_r1_,
                                                         &node_to_rr_r1_,
                                                         nullptr);

            const Count num_r = static_cast<Count>(rr_sets_r1_.size());
            const HistGreedyResult cover = max_cover_hist_lazy(node_to_rr_r1_,
                                                               rr_sets_r1_,
                                                               out_degree_,
                                                               n_nodes_,
                                                               k_,
                                                               k_,
                                                               sentinel,
                                                               false);
            const auto conn_nodes = make_node_mask(n_nodes_, cover.seeds, "HISTProxy final");
            const Count eval_samples = num_r * static_cast<Count>(multiple);
            const EarlyStopEvalResult eval =
                eval_seed_set_inf_early_stop(
                    graph_, random_, eval_samples, conn_nodes, n_nodes_, n_nodes_);

            const double deg_vldt =
                eval.influence * static_cast<double>(multiple) * static_cast<double>(num_r) / n_;
            double upper_bound = cover.bound_min;
            if (!(upper_bound > 0.0) || !std::isfinite(upper_bound)) {
                upper_bound = n_ / approx_;
            }
            const double upper_deg_opt = upper_bound * static_cast<double>(num_r) / n_;
            const double lower_select =
                (pow2(std::sqrt(deg_vldt + a2 * 2.0 / 9.0) - std::sqrt(a2 / 2.0)) - a2 / 18.0) /
                static_cast<double>(multiple);
            const double upper_opt =
                pow2(std::sqrt(upper_deg_opt + a2 / 2.0) + std::sqrt(a2 / 2.0));
            const double curr_approx = (upper_opt > 0.0) ? (lower_select / upper_opt) : -1.0;

            best.seeds = cover.seeds;
            best.upper_bound = upper_bound;
            best.rr_samples = num_r * 2U;

            const double full_rr_size = rr_average_size(rr_sets_r1_);
            multiple = decide_multiple(full_rr_size, eval.avg_rr_size, num_r);
            if (curr_approx >= approx_ - target_epsilon) {
                return best;
            }
        }

        return best;
    }

    const ICGraph& graph_;
    Budget k_;
    double epsilon_;
    double delta_;
    Random random_;
    RRSampler rr_sampler_;
    NodeId n_nodes_;
    double n_;
    double approx_;
    Count base_num_rrsets_;
    std::vector<Count> out_degree_;
    std::vector<RRSet> rr_sets_r1_;
    std::vector<RRSet> rr_sets_r2_;
    std::vector<std::vector<Count>> node_to_rr_r1_;
    std::vector<std::vector<Count>> node_to_rr_r2_;
    Count hit_sentinel_r2_ = 0U;
};

class HistRunnerCliqueImplicit {
public:
    HistRunnerCliqueImplicit(const HybridHypergraph& graph,
                             Budget k,
                             double epsilon,
                             double delta,
                             std::uint64_t seed,
                             NodeId active_node_count)
        : graph_(graph),
          k_(k),
          epsilon_(epsilon),
          delta_(delta),
          random_(seed ^ 0xbf58476d1ce4e5b9ULL),
          n_nodes_((active_node_count == 0U) ? graph.num_nodes() : active_node_count),
          n_(static_cast<double>((active_node_count == 0U) ? graph.num_nodes() : active_node_count)),
          approx_(1.0 - 1.0 / std::exp(1.0)),
          base_num_rrsets_(std::max<Count>(1U, static_cast<Count>(3.0 * std::log(1.0 / delta)))) {
        if (n_nodes_ == 0U || n_nodes_ > graph_.num_nodes()) {
            throw std::invalid_argument(
                "run_hist_proxy_clique_implicit active_node_count must be in [1, graph.num_nodes()]");
        }
        out_degree_.resize(n_nodes_, 0U);
        for (NodeId v = 0; v < n_nodes_; ++v) {
            out_degree_[v] = clique_out_degree_proxy(graph_, v);
        }
        node_to_rr_r1_.assign(n_nodes_, {});
        node_to_rr_r2_.assign(n_nodes_, {});
    }

    OPIMProxyResult run() {
        const double epsilon_phase = std::max(1e-9, epsilon_ * 0.5);
        const double delta_phase = std::max(1e-12, delta_ * 0.5);

        std::vector<NodeId> sentinel = find_dynamic_sub(epsilon_phase, delta_phase);
        if (sentinel.empty()) {
            sentinel.push_back(0U);
        }
        if (sentinel.size() > k_) {
            sentinel.resize(k_);
        }

        if (sentinel.size() >= k_) {
            OPIMProxyResult result;
            result.seeds = sentinel;
            result.upper_bound = n_ / approx_;
            result.rr_samples = static_cast<Count>(rr_sets_r1_.size() * 2U);
            return result;
        }
        return find_rem_set(sentinel, epsilon_phase, epsilon_, delta_phase);
    }

private:
    std::vector<NodeId> find_dynamic_sub(double epsilon, double delta) {
        reset_rr_index(n_nodes_, &rr_sets_r1_, &node_to_rr_r1_, nullptr);
        reset_rr_index(n_nodes_, &rr_sets_r2_, &node_to_rr_r2_, &hit_sentinel_r2_);

        const double x = (k_ > 1U) ? (1.0 - 1.0 / static_cast<double>(k_)) : 0.0;
        Budget min_sub_size = 1U;
        if (k_ > 1U && x > 0.0 && x < 1.0) {
            const double numer = std::log(std::max(1e-12, 1.0 - epsilon));
            const double denom = std::log(x);
            if (std::isfinite(numer) && std::isfinite(denom) && denom != 0.0) {
                min_sub_size = static_cast<Budget>(std::ceil(numer / denom));
                if (min_sub_size == 0U) {
                    min_sub_size = 1U;
                }
                if (min_sub_size > k_) {
                    min_sub_size = k_;
                }
            }
        }

        const double alpha = std::sqrt(std::log(6.0 / delta));
        const double beta = std::sqrt(approx_ * (logcnk(n_nodes_, k_) + std::log(6.0 / delta)));
        Count max_num_r = static_cast<Count>(2.0 * n_ * pow2(approx_ * alpha + beta) /
                                             static_cast<double>(k_) / pow2(epsilon)) + 1U;
        if (max_num_r < base_num_rrsets_) {
            max_num_r = base_num_rrsets_;
        }

        const Count num_iter = compute_num_iter(max_num_r, base_num_rrsets_);
        const double a1 = std::log(static_cast<double>(num_iter) * 3.0 / delta);
        const double a2 = std::log(static_cast<double>(num_iter) * 6.0 / delta);

        std::vector<NodeId> sentinel;
        bool first_round = true;
        int multiple = 1;

        for (Count iter = 0U; iter < num_iter; ++iter) {
            const Count target_num_r = std::min(max_num_r, doubled_rr_target(base_num_rrsets_, iter));
            append_rr_sets_until_projected_clique_implicit(
                graph_, random_, target_num_r, n_nodes_, n_nodes_, &rr_sets_r1_, &node_to_rr_r1_);

            const Count num_r = static_cast<Count>(rr_sets_r1_.size());
            Budget target_size = first_round ? (k_ / 4U) : (k_ / 8U);
            first_round = false;
            if (target_size == 0U) {
                target_size = 1U;
            }

            const HistGreedyResult cover = max_cover_hist_lazy(node_to_rr_r1_,
                                                               rr_sets_r1_,
                                                               out_degree_,
                                                               n_nodes_,
                                                               target_size,
                                                               k_,
                                                               {},
                                                               true);
            std::vector<NodeId> sentinel_candidates = cover.seeds;
            std::vector<Count> sentinel_prefix = cover.prefix_covered_rr;

            // Align with HIST source heuristic: if sentinel greedy is not yet in a
            // high-confidence regime, trim trailing low-contribution seeds and
            // refill by out-degree from trimmed candidates.
            if (!(sentinel_candidates.size() == target_size && num_r > 1000U)) {
                const Count threshold = static_cast<Count>(0.9 * static_cast<double>(cover.covered_rr));
                std::vector<NodeId> trimmed_nodes;
                while (sentinel_candidates.size() > 1U && sentinel_prefix.size() > 1U) {
                    const std::size_t i = sentinel_candidates.size() - 1U;
                    if (sentinel_prefix[i - 1U] >= threshold) {
                        trimmed_nodes.push_back(sentinel_candidates.back());
                        sentinel_candidates.pop_back();
                        sentinel_prefix.pop_back();
                    } else {
                        break;
                    }
                }
                std::sort(trimmed_nodes.begin(), trimmed_nodes.end(), [&](NodeId a, NodeId b) {
                    if (out_degree_[a] != out_degree_[b]) {
                        return out_degree_[a] > out_degree_[b];
                    }
                    return a < b;
                });
                std::vector<std::uint8_t> in_sentinel(n_nodes_, 0U);
                for (NodeId v : sentinel_candidates) {
                    in_sentinel[v] = 1U;
                }
                for (NodeId v : trimmed_nodes) {
                    if (sentinel_candidates.size() >= target_size) {
                        break;
                    }
                    if (in_sentinel[v] != 0U) {
                        continue;
                    }
                    in_sentinel[v] = 1U;
                    sentinel_candidates.push_back(v);
                    sentinel_prefix.push_back(cover.covered_rr);
                }
            }

            if (sentinel_candidates.size() < target_size || sentinel_prefix.empty()) {
                continue;
            }

            int last_pos = 0;
            bool found = false;
            for (int i = static_cast<int>(sentinel_prefix.size()) - 1; i >= 0; --i) {
                const double lower_deg = static_cast<double>(sentinel_prefix[static_cast<std::size_t>(i)]);
                const double upper_deg = cover.bound_min;
                const double a = std::log(static_cast<double>(num_iter) * 6.0 / delta);
                const double lower =
                    pow2(std::sqrt(lower_deg + a * 2.0 / 9.0) - std::sqrt(a / 2.0)) - a / 18.0;
                double upper =
                    pow2(std::sqrt(upper_deg + a1 / 2.0) + std::sqrt(a1 / 2.0));
                upper = std::min(upper, static_cast<double>(num_r));
                const double vec_approx = (upper > 0.0) ? (lower / upper) : -1.0;
                const double calc_approx =
                    (1.0 - std::pow(x, static_cast<double>(i + 1)) - epsilon) * 1.2;
                if (vec_approx > calc_approx) {
                    found = true;
                    last_pos = i;
                    break;
                }
            }
            if (!found) {
                last_pos = 0;
            }

            Count set_size = static_cast<Count>(last_pos + 1);
            set_size = std::max<Count>(set_size, 10U);
            set_size = std::min<Count>(set_size, static_cast<Count>(target_size));
            set_size = std::max<Count>(set_size, static_cast<Count>(min_sub_size));
            set_size = std::min<Count>(set_size, static_cast<Count>(sentinel_candidates.size()));
            sentinel.assign(sentinel_candidates.begin(),
                            sentinel_candidates.begin() + static_cast<std::size_t>(set_size));

            const auto sentinel_nodes = make_node_mask(n_nodes_, sentinel, "HISTProxy sentinel");
            reset_rr_index(n_nodes_, &rr_sets_r2_, &node_to_rr_r2_, &hit_sentinel_r2_);
            const Count target_vldt = num_r * static_cast<Count>(multiple);
            append_rr_sets_until_with_sentinel_projected_clique_implicit(graph_,
                                                                         random_,
                                                                         target_vldt,
                                                                         n_nodes_,
                                                                         n_nodes_,
                                                                         sentinel_nodes,
                                                                         &rr_sets_r2_,
                                                                         &node_to_rr_r2_,
                                                                         &hit_sentinel_r2_);

            const double inf_vldt =
                calculate_inf_early_stop(hit_sentinel_r2_, static_cast<Count>(rr_sets_r2_.size()), n_nodes_);
            const double deg_vldt = inf_vldt * static_cast<double>(multiple) *
                                    static_cast<double>(num_r) / n_;
            double upper_bound = cover.bound_min;
            if (!(upper_bound > 0.0) || !std::isfinite(upper_bound)) {
                upper_bound = n_ / approx_;
            }

            const double upper_deg_opt = upper_bound * static_cast<double>(num_r) / n_;
            double lower_select =
                (pow2(std::sqrt(deg_vldt + a2 * 2.0 / 9.0) - std::sqrt(a2 / 2.0)) - a2 / 18.0) /
                static_cast<double>(multiple);
            if (lower_select < 0.0) {
                lower_select =
                    static_cast<double>(sentinel.size()) / n_ * static_cast<double>(num_r) *
                    static_cast<double>(multiple);
            }

            double upper_opt = pow2(std::sqrt(upper_deg_opt + a1 / 2.0) + std::sqrt(a1 / 2.0));
            upper_opt = std::min(upper_opt, static_cast<double>(num_r));
            const double curr_approx = (upper_opt > 0.0) ? (lower_select / upper_opt) : -1.0;
            const double approx_sub = (x > 0.0) ? (1.0 - std::pow(x, static_cast<double>(sentinel.size()))) : 1.0;
            const double target_approx = approx_sub - epsilon;

            if (curr_approx >= target_approx) {
                return sentinel;
            }
            if (num_r < 100U) {
                continue;
            }

            const double full_rr_size = rr_average_size(rr_sets_r1_);
            const double trunc_rr_size = rr_average_size(rr_sets_r2_);
            if (trunc_rr_size <= 0.0 || (full_rr_size / trunc_rr_size) < 2.0) {
                continue;
            }

            const double lower_threshold =
                (upper_opt * n_ / static_cast<double>(num_r)) * target_approx;
            if ((inf_vldt / static_cast<double>(multiple)) > lower_threshold && lower_threshold > 0.0) {
                const double new_approx =
                    increase_r2(sentinel_nodes, a2, upper_opt, target_approx);
                if (new_approx > target_approx) {
                    return sentinel;
                }
            }
        }
        return sentinel;
    }

    double increase_r2(const std::vector<std::uint8_t>& sentinel_nodes,
                       double a,
                       double upper_opt,
                       double target_approx) {
        const Count vldt_rrsets = static_cast<Count>(rr_sets_r2_.size());
        const Count r1_rrsets = static_cast<Count>(rr_sets_r1_.size());
        if (vldt_rrsets == 0U || r1_rrsets == 0U || upper_opt <= 0.0) {
            return 0.0;
        }

        const int multiple = 4;
        const double inf_vldt = calculate_inf_early_stop(hit_sentinel_r2_, vldt_rrsets, n_nodes_);
        const double deg_vldt = inf_vldt * static_cast<double>(vldt_rrsets) / n_;
        const double lower_select =
            pow2(std::sqrt(deg_vldt * static_cast<double>(multiple) + a * 2.0 / 9.0) -
                 std::sqrt(a / 2.0)) - a / 18.0;
        const double estimate_approx =
            (lower_select / (static_cast<double>(multiple) * static_cast<double>(vldt_rrsets))) /
            (upper_opt / static_cast<double>(r1_rrsets));
        if (estimate_approx < target_approx) {
            return 0.0;
        }

        Count target_rrsets = vldt_rrsets;
        if (target_rrsets <= (std::numeric_limits<Count>::max() / static_cast<Count>(multiple))) {
            target_rrsets *= static_cast<Count>(multiple);
        } else {
            target_rrsets = std::numeric_limits<Count>::max();
        }

        append_rr_sets_until_with_sentinel_projected_clique_implicit(graph_,
                                                                      random_,
                                                                      target_rrsets,
                                                                      n_nodes_,
                                                                      n_nodes_,
                                                                      sentinel_nodes,
                                                                      &rr_sets_r2_,
                                                                      &node_to_rr_r2_,
                                                                      &hit_sentinel_r2_);

        const Count new_vldt_rrsets = static_cast<Count>(rr_sets_r2_.size());
        if (new_vldt_rrsets == 0U) {
            return 0.0;
        }
        const double new_inf_vldt =
            calculate_inf_early_stop(hit_sentinel_r2_, new_vldt_rrsets, n_nodes_);
        const double new_deg_vldt = new_inf_vldt * static_cast<double>(new_vldt_rrsets) / n_;
        const double new_lower_select =
            pow2(std::sqrt(new_deg_vldt + a * 2.0 / 9.0) - std::sqrt(a / 2.0)) - a / 18.0;
        const double new_approx =
            (new_lower_select / static_cast<double>(new_vldt_rrsets)) /
            (upper_opt / static_cast<double>(r1_rrsets));
        return (new_approx > target_approx) ? new_approx : 0.0;
    }

    OPIMProxyResult find_rem_set(const std::vector<NodeId>& sentinel,
                                 double epsilon,
                                 double target_epsilon,
                                 double delta) {
        reset_rr_index(n_nodes_, &rr_sets_r1_, &node_to_rr_r1_, nullptr);

        const Count sub_seed_size = static_cast<Count>(sentinel.size());
        const double delta_upper = delta / 3.0;
        const double alpha = std::sqrt(std::log(3.0 / delta_upper));
        const double beta = std::sqrt(
            approx_ *
            (logcnk(n_nodes_, k_ - static_cast<Budget>(sub_seed_size)) + std::log(3.0 / delta_upper)));

        Count max_num_r = static_cast<Count>(2.0 * n_ * pow2(alpha + beta) /
                                             static_cast<double>(k_) / pow2(epsilon)) + 1U;
        if (max_num_r < base_num_rrsets_) {
            max_num_r = base_num_rrsets_;
        }
        const Count num_iter = compute_num_iter(max_num_r, base_num_rrsets_);
        const double a2 = std::log(static_cast<double>(num_iter) * 3.0 / delta);

        OPIMProxyResult best;
        best.seeds = sentinel;
        best.upper_bound = n_ / approx_;
        best.rr_samples = 0U;

        int multiple = 1;
        const auto sentinel_nodes = make_node_mask(n_nodes_, sentinel, "HISTProxy sentinel");

        for (Count iter = 0U; iter < num_iter; ++iter) {
            const Count target_num_r = std::min(max_num_r, doubled_rr_target(base_num_rrsets_, iter));
            append_rr_sets_until_with_sentinel_projected_clique_implicit(graph_,
                                                                         random_,
                                                                         target_num_r,
                                                                         n_nodes_,
                                                                         n_nodes_,
                                                                         sentinel_nodes,
                                                                         &rr_sets_r1_,
                                                                         &node_to_rr_r1_,
                                                                         nullptr);

            const Count num_r = static_cast<Count>(rr_sets_r1_.size());
            const HistGreedyResult cover = max_cover_hist_lazy(node_to_rr_r1_,
                                                               rr_sets_r1_,
                                                               out_degree_,
                                                               n_nodes_,
                                                               k_,
                                                               k_,
                                                               sentinel,
                                                               false);
            const auto conn_nodes = make_node_mask(n_nodes_, cover.seeds, "HISTProxy final");
            const Count eval_samples = num_r * static_cast<Count>(multiple);
            const EarlyStopEvalResult eval = eval_seed_set_inf_early_stop_clique_implicit(
                graph_, random_, eval_samples, conn_nodes, n_nodes_, n_nodes_);

            const double deg_vldt =
                eval.influence * static_cast<double>(multiple) * static_cast<double>(num_r) / n_;
            double upper_bound = cover.bound_min;
            if (!(upper_bound > 0.0) || !std::isfinite(upper_bound)) {
                upper_bound = n_ / approx_;
            }
            const double upper_deg_opt = upper_bound * static_cast<double>(num_r) / n_;
            const double lower_select =
                (pow2(std::sqrt(deg_vldt + a2 * 2.0 / 9.0) - std::sqrt(a2 / 2.0)) - a2 / 18.0) /
                static_cast<double>(multiple);
            const double upper_opt =
                pow2(std::sqrt(upper_deg_opt + a2 / 2.0) + std::sqrt(a2 / 2.0));
            const double curr_approx = (upper_opt > 0.0) ? (lower_select / upper_opt) : -1.0;

            best.seeds = cover.seeds;
            best.upper_bound = upper_bound;
            best.rr_samples = num_r * 2U;

            const double full_rr_size = rr_average_size(rr_sets_r1_);
            multiple = decide_multiple(full_rr_size, eval.avg_rr_size, num_r);
            if (curr_approx >= approx_ - target_epsilon) {
                return best;
            }
        }

        return best;
    }

    const HybridHypergraph& graph_;
    Budget k_;
    double epsilon_;
    double delta_;
    Random random_;
    NodeId n_nodes_;
    double n_;
    double approx_;
    Count base_num_rrsets_;
    std::vector<Count> out_degree_;
    std::vector<RRSet> rr_sets_r1_;
    std::vector<RRSet> rr_sets_r2_;
    std::vector<std::vector<Count>> node_to_rr_r1_;
    std::vector<std::vector<Count>> node_to_rr_r2_;
    Count hit_sentinel_r2_ = 0U;
};

}  // namespace

OPIMProxyResult HISTProxy::run(const ICGraph& graph,
                               Budget k,
                               double epsilon,
                               double delta,
                               std::uint64_t seed,
                               NodeId active_node_count) const {
    if (k == 0U) {
        throw std::invalid_argument("HISTProxy k must be > 0");
    }
    if (graph.num_nodes() == 0U) {
        throw std::invalid_argument("HISTProxy cannot run on empty graph");
    }
    if (epsilon <= 0.0 || epsilon >= 1.0) {
        throw std::invalid_argument("HISTProxy epsilon must be in (0,1)");
    }
    if (delta <= 0.0 || delta >= 1.0) {
        throw std::invalid_argument("HISTProxy delta must be in (0,1)");
    }
    const NodeId n_nodes = (active_node_count == 0U) ? graph.num_nodes() : active_node_count;
    if (n_nodes == 0U || n_nodes > graph.num_nodes()) {
        throw std::invalid_argument("HISTProxy active_node_count must be in [1, graph.num_nodes()]");
    }
    if (k > n_nodes) {
        throw std::invalid_argument("HISTProxy k must be <= active node count");
    }
    HistRunner runner(graph, k, epsilon, delta, seed, n_nodes);
    OPIMProxyResult result = runner.run();
    if (result.seeds.empty()) {
        result.seeds.push_back(0U);
    }
    if (result.seeds.size() > k) {
        result.seeds.resize(k);
    }
    if (result.rr_samples == 0U) {
        result.rr_samples = 2U;
    }
    if (!(result.upper_bound > 0.0) || !std::isfinite(result.upper_bound)) {
        const double approx = 1.0 - 1.0 / std::exp(1.0);
        result.upper_bound = static_cast<double>(n_nodes) / approx;
    }
    return result;
}

OPIMProxyResult run_hist_proxy_clique_implicit(const HybridHypergraph& graph,
                                               Budget k,
                                               double epsilon,
                                               double delta,
                                               std::uint64_t seed,
                                               NodeId active_node_count) {
    if (k == 0U) {
        throw std::invalid_argument("run_hist_proxy_clique_implicit k must be > 0");
    }
    if (graph.num_nodes() == 0U) {
        throw std::invalid_argument("run_hist_proxy_clique_implicit cannot run on empty graph");
    }
    if (epsilon <= 0.0 || epsilon >= 1.0) {
        throw std::invalid_argument("run_hist_proxy_clique_implicit epsilon must be in (0,1)");
    }
    if (delta <= 0.0 || delta >= 1.0) {
        throw std::invalid_argument("run_hist_proxy_clique_implicit delta must be in (0,1)");
    }

    const NodeId n_nodes = (active_node_count == 0U) ? graph.num_nodes() : active_node_count;
    if (n_nodes == 0U || n_nodes > graph.num_nodes()) {
        throw std::invalid_argument(
            "run_hist_proxy_clique_implicit active_node_count must be in [1, graph.num_nodes()]");
    }
    if (k > n_nodes) {
        throw std::invalid_argument("run_hist_proxy_clique_implicit k must be <= active node count");
    }

    HistRunnerCliqueImplicit runner(graph, k, epsilon, delta, seed, n_nodes);
    OPIMProxyResult result = runner.run();
    if (result.seeds.empty()) {
        result.seeds.push_back(0U);
    }
    if (result.seeds.size() > k) {
        result.seeds.resize(k);
    }
    if (result.rr_samples == 0U) {
        result.rr_samples = 2U;
    }
    if (!(result.upper_bound > 0.0) || !std::isfinite(result.upper_bound)) {
        const double approx = 1.0 - 1.0 / std::exp(1.0);
        result.upper_bound = static_cast<double>(n_nodes) / approx;
    }
    return result;
}

ICGraph build_upper_relaxed_aux_ic_graph(const HybridHypergraph& graph) {
    const std::uint64_t n_users = static_cast<std::uint64_t>(graph.num_nodes());
    const std::uint64_t n_hyperedges = static_cast<std::uint64_t>(graph.num_hyperedges());
    const std::uint64_t total_nodes = n_users + n_hyperedges;
    if (total_nodes >= static_cast<std::uint64_t>(kInvalidNodeId)) {
        throw std::overflow_error("upper-relaxed auxiliary graph node count exceeds NodeId range");
    }

    ICGraph g(static_cast<NodeId>(total_nodes));
    for (const auto& e : graph.edges()) {
        g.add_edge(e.src, e.dst, e.prob);
    }

    for (const auto& h : graph.hyperedges()) {
        const NodeId aux = static_cast<NodeId>(n_users + static_cast<std::uint64_t>(h.id));
        for (NodeId u : h.nodes) {
            g.add_edge(u, aux, 1.0);
            g.add_edge(aux, u, 1.0);
        }
    }
    g.build_indices();
    g.validate();
    return g;
}

ICGraph build_upper_relaxed_ic_graph(const HybridHypergraph& graph) {
    std::unordered_map<std::uint64_t, Prob> pair_prob;
    pair_prob.reserve(static_cast<std::size_t>(graph.num_edges()) +
                      static_cast<std::size_t>(graph.num_hyperedges()) * 8U);

    for (const auto& e : graph.edges()) {
        accumulate_prob(&pair_prob, e.src, e.dst, e.prob);
    }

    for (const auto& h : graph.hyperedges()) {
        for (NodeId u : h.nodes) {
            for (NodeId v : h.nodes) {
                if (u == v) {
                    continue;
                }
                accumulate_prob(&pair_prob, u, v, 1.0);
            }
        }
    }

    ICGraph g(graph.num_nodes());
    for (const auto& kv : pair_prob) {
        const NodeId u = static_cast<NodeId>(kv.first >> 32U);
        const NodeId v = static_cast<NodeId>(kv.first & 0xffffffffULL);
        g.add_edge(u, v, kv.second);
    }
    g.build_indices();
    g.validate();
    return g;
}

ICGraph build_edge_only_ic_graph(const HybridHypergraph& graph) {
    ICGraph g(graph.num_nodes());
    for (const auto& e : graph.edges()) {
        g.add_edge(e.src, e.dst, e.prob);
    }
    g.build_indices();
    g.validate();
    return g;
}

ICGraph build_clique_expansion_ic_graph(const HybridHypergraph& graph) {
    std::unordered_map<std::uint64_t, Prob> pair_prob;
    pair_prob.reserve(static_cast<std::size_t>(graph.num_edges()) +
                      static_cast<std::size_t>(graph.num_hyperedges()) * 8U);

    for (const auto& e : graph.edges()) {
        accumulate_prob(&pair_prob, e.src, e.dst, e.prob);
    }

    for (const auto& h : graph.hyperedges()) {
        const double denom = static_cast<double>(h.nodes.size() - 1U);
        const double numer = static_cast<double>(h.nodes.size() - h.threshold);
        const Prob p_ce = numer / denom;

        for (NodeId u : h.nodes) {
            for (NodeId v : h.nodes) {
                if (u == v) {
                    continue;
                }
                accumulate_prob(&pair_prob, u, v, p_ce);
            }
        }
    }

    ICGraph g(graph.num_nodes());
    for (const auto& kv : pair_prob) {
        const NodeId u = static_cast<NodeId>(kv.first >> 32U);
        const NodeId v = static_cast<NodeId>(kv.first & 0xffffffffULL);
        g.add_edge(u, v, kv.second);
    }
    g.build_indices();
    g.validate();
    return g;
}

}  // namespace htc
