#pragma once

#include <random>

#include "htc/types.hpp"

namespace htc {

class Random {
public:
    explicit Random(std::uint64_t seed);

    bool bernoulli(Prob p);
    NodeId uniform_node(NodeId low, NodeId high_inclusive);
    Count uniform_count(Count low, Count high_inclusive);
    double uniform_real();

private:
    std::mt19937_64 rng_;
};

}  // namespace htc
