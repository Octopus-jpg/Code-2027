#include "htc/synthetic.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "htc/random.hpp"

namespace htc {
namespace {

std::uint64_t pair_key(NodeId u, NodeId v) {
    return (static_cast<std::uint64_t>(u) << 32U) | static_cast<std::uint64_t>(v);
}

}  // namespace

HybridHypergraph generate_synthetic_hybrid_hypergraph(const SyntheticConfig& config) {
    if (config.num_nodes == 0U) {
        throw std::invalid_argument("SyntheticConfig.num_nodes must be > 0");
    }
    if (config.num_nodes == kInvalidNodeId) {
        throw std::invalid_argument("SyntheticConfig.num_nodes cannot be kInvalidNodeId");
    }
    if (config.min_hyperedge_size < 2U) {
        throw std::invalid_argument("SyntheticConfig.min_hyperedge_size must be >= 2");
    }
    if (config.min_hyperedge_size > config.max_hyperedge_size) {
        throw std::invalid_argument("SyntheticConfig.min_hyperedge_size must be <= max_hyperedge_size");
    }
    if (config.min_edge_prob < 0.0 || config.min_edge_prob > 1.0 ||
        config.max_edge_prob < 0.0 || config.max_edge_prob > 1.0 ||
        config.min_edge_prob > config.max_edge_prob) {
        throw std::invalid_argument("SyntheticConfig edge probability range is invalid");
    }

    const std::uint64_t n = static_cast<std::uint64_t>(config.num_nodes);
    const std::uint64_t max_possible_edges = n * (n - 1U);
    Count target_edges = std::min<Count>(config.num_edges, max_possible_edges);

    HybridHypergraph graph(config.num_nodes);
    Random random(config.seed);

    std::unordered_set<std::uint64_t> edge_set;
    edge_set.reserve(static_cast<std::size_t>(target_edges * 1.3 + 64));

    while (graph.num_edges() < target_edges) {
        const NodeId u = random.uniform_node(0U, config.num_nodes - 1U);
        const NodeId v = random.uniform_node(0U, config.num_nodes - 1U);
        if (u == v) {
            continue;
        }

        const std::uint64_t key = pair_key(u, v);
        if (!edge_set.insert(key).second) {
            continue;
        }

        const Prob p = config.min_edge_prob +
                       (config.max_edge_prob - config.min_edge_prob) * random.uniform_real();
        graph.add_edge(u, v, p);
    }

    std::vector<NodeId> node_buf(config.num_nodes);
    for (NodeId i = 0; i < config.num_nodes; ++i) {
        node_buf[i] = i;
    }

    for (Count hidx = 0; hidx < config.num_hyperedges; ++hidx) {
        const std::uint32_t max_size =
            std::min<std::uint32_t>(config.max_hyperedge_size, static_cast<std::uint32_t>(config.num_nodes));
        const std::uint32_t min_size =
            std::min<std::uint32_t>(config.min_hyperedge_size, max_size);
        if (max_size < 2U) {
            break;
        }

        const std::uint32_t size = static_cast<std::uint32_t>(
            random.uniform_count(static_cast<Count>(min_size), static_cast<Count>(max_size)));

        for (NodeId i = 0; i < size; ++i) {
            const NodeId j = random.uniform_node(i, config.num_nodes - 1U);
            std::swap(node_buf[i], node_buf[j]);
        }

        std::vector<NodeId> nodes;
        nodes.reserve(size);
        for (std::uint32_t i = 0; i < size; ++i) {
            nodes.push_back(node_buf[i]);
        }

        const std::uint32_t threshold = static_cast<std::uint32_t>(
            random.uniform_count(1U, static_cast<Count>(size - 1U)));
        graph.add_hyperedge(std::move(nodes), threshold);
    }

    graph.build_indices();
    graph.validate();
    return graph;
}

}  // namespace htc
