#pragma once

#include "htc/graph.hpp"
#include "htc/types.hpp"

namespace htc {

struct SyntheticConfig {
    NodeId num_nodes = 1000;
    Count num_edges = 10000;
    Count num_hyperedges = 2000;
    std::uint32_t min_hyperedge_size = 3;
    std::uint32_t max_hyperedge_size = 6;
    Prob min_edge_prob = 0.01;
    Prob max_edge_prob = 0.1;
    std::uint64_t seed = 1;
};

HybridHypergraph generate_synthetic_hybrid_hypergraph(const SyntheticConfig& config);

}  // namespace htc
