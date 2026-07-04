#include "htc/cascade.hpp"

#include <cstddef>
#include <cstdint>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

namespace htc {

NodeId HTCCascadeSimulator::simulate_spread(const HybridHypergraph& graph,
                                            const std::vector<NodeId>& seeds,
                                            Random* random,
                                            const char* seed_out_of_range_context) {
    if (random == nullptr) {
        throw std::invalid_argument("random cannot be null");
    }

    const char* out_of_range_context =
        (seed_out_of_range_context == nullptr) ? "HTCCascadeSimulator" : seed_out_of_range_context;

    std::vector<std::uint8_t> active(graph.num_nodes(), 0U);
    std::vector<std::uint32_t> hyper_active_count(graph.num_hyperedges(), 0U);
    std::queue<NodeId> queue;

    auto activate = [&](NodeId v) {
        if (active[v] != 0U) {
            return;
        }
        active[v] = 1U;
        queue.push(v);
        for (HyperedgeId hid : graph.incident_hyperedges(v)) {
            ++hyper_active_count[hid];
        }
    };

    for (NodeId seed : seeds) {
        if (seed >= graph.num_nodes()) {
            throw std::out_of_range(std::string("seed out of range in ") + out_of_range_context);
        }
        activate(seed);
    }

    while (!queue.empty()) {
        const NodeId u = queue.front();
        queue.pop();

        for (EdgeId eid : graph.out_edges(u)) {
            const auto& e = graph.edge(eid);
            if (active[e.dst] != 0U) {
                continue;
            }
            if (random->bernoulli(e.prob)) {
                activate(e.dst);
            }
        }

        for (HyperedgeId hid : graph.incident_hyperedges(u)) {
            const auto& h = graph.hyperedge(hid);
            if (hyper_active_count[hid] < h.threshold) {
                continue;
            }
            for (NodeId v : h.nodes) {
                if (active[v] == 0U) {
                    activate(v);
                }
            }
        }
    }

    NodeId spread = 0;
    for (std::uint8_t x : active) {
        spread += (x != 0U) ? 1U : 0U;
    }
    return spread;
}

NodeId simulate_htc_spread(const HybridHypergraph& graph,
                           const std::vector<NodeId>& seeds,
                           Random* random,
                           const char* seed_out_of_range_context) {
    return HTCCascadeSimulator::simulate_spread(
        graph, seeds, random, seed_out_of_range_context);
}

}  // namespace htc
