#include "htc/random.hpp"

#include <stdexcept>

namespace htc {

Random::Random(std::uint64_t seed) : rng_(seed) {}

bool Random::bernoulli(Prob p) {
    if (p < 0.0 || p > 1.0) {
        throw std::invalid_argument("bernoulli probability must be in [0, 1]");
    }
    std::bernoulli_distribution dist(p);
    return dist(rng_);
}

NodeId Random::uniform_node(NodeId low, NodeId high_inclusive) {
    if (low > high_inclusive) {
        throw std::invalid_argument("uniform_node low must be <= high");
    }
    std::uniform_int_distribution<NodeId> dist(low, high_inclusive);
    return dist(rng_);
}

Count Random::uniform_count(Count low, Count high_inclusive) {
    if (low > high_inclusive) {
        throw std::invalid_argument("uniform_count low must be <= high");
    }
    std::uniform_int_distribution<Count> dist(low, high_inclusive);
    return dist(rng_);
}

double Random::uniform_real() {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng_);
}

}  // namespace htc
