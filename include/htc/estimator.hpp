#pragma once

#include <vector>

#include "htc/graph.hpp"
#include "htc/types.hpp"

namespace htc {

Count compute_forward_mc_samples(NodeId n,
                                 Count seed_set_size,
                                 double epsilon,
                                 double delta);

class RTWEstimator {
public:
    explicit RTWEstimator(const HybridHypergraph& graph);

    IEResult estimate_adaptive(const std::vector<NodeId>& seeds, const IEConfig& config) const;
    IEResult estimate_fixed(const std::vector<NodeId>& seeds, Count theta, std::uint64_t seed) const;

private:
    const HybridHypergraph& graph_;
};

class MCEstimator {
public:
    explicit MCEstimator(const HybridHypergraph& graph);

    IEResult estimate(const std::vector<NodeId>& seeds, Count simulations, std::uint64_t seed) const;

private:
    const HybridHypergraph& graph_;
};

class URIEstimator {
public:
    explicit URIEstimator(const HybridHypergraph& graph);

    IEResult estimate_adaptive(const std::vector<NodeId>& seeds, const IEConfig& config) const;

private:
    const HybridHypergraph& graph_;
};

}  // namespace htc
