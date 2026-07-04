#pragma once

#include <vector>

#include "htc/graph.hpp"
#include "htc/random.hpp"
#include "htc/types.hpp"

namespace htc {

class HTCCascadeSimulator {
public:
    static NodeId simulate_spread(
        const HybridHypergraph& graph,
        const std::vector<NodeId>& seeds,
        Random* random,
        const char* seed_out_of_range_context = "HTCCascadeSimulator");
};

NodeId simulate_htc_spread(const HybridHypergraph& graph,
                           const std::vector<NodeId>& seeds,
                           Random* random,
                           const char* seed_out_of_range_context = "HTCCascadeSimulator");

}  // namespace htc
