#pragma once

#include <vector>

#include "htc/graph.hpp"
#include "htc/random.hpp"
#include "htc/types.hpp"

namespace htc {

class ICGraph {
public:
    explicit ICGraph(NodeId n = 0);

    NodeId num_nodes() const;
    EdgeId num_edges() const;

    EdgeId add_edge(NodeId u, NodeId v, Prob p);
    void build_indices();
    void validate() const;

    const DirectedEdge& edge(EdgeId id) const;
    const std::vector<EdgeId>& in_edges(NodeId v) const;
    const std::vector<EdgeId>& out_edges(NodeId u) const;

private:
    static void validate_probability(Prob p);
    void validate_node(NodeId v) const;

    NodeId n_ = 0;
    std::vector<DirectedEdge> edges_;
    std::vector<std::vector<EdgeId>> in_edges_;
    std::vector<std::vector<EdgeId>> out_edges_;
};

struct RRSet {
    NodeId root = kInvalidNodeId;
    std::vector<NodeId> nodes;
};

class RRSampler {
public:
    explicit RRSampler(const ICGraph& graph);

    RRSet sample(Random& random) const;
    RRSet sample_with_root(NodeId root, Random& random) const;

private:
    const ICGraph& graph_;
};

class CoverageGreedy {
public:
    std::vector<NodeId> select(const std::vector<RRSet>& rr_sets, NodeId num_nodes, Budget k) const;
};

struct OPIMProxyResult {
    std::vector<NodeId> seeds;
    double upper_bound = 0.0;
    Count rr_samples = 0;
};

class OPIMProxy {
public:
    OPIMProxyResult run(const ICGraph& graph,
                        Budget k,
                        double epsilon,
                        double delta,
                        std::uint64_t seed,
                        NodeId active_node_count = 0U) const;
};

class HISTProxy {
public:
    OPIMProxyResult run(const ICGraph& graph,
                        Budget k,
                        double epsilon,
                        double delta,
                        std::uint64_t seed,
                        NodeId active_node_count = 0U) const;
};

OPIMProxyResult run_opim_proxy_clique_implicit(const HybridHypergraph& graph,
                                               Budget k,
                                               double epsilon,
                                               double delta,
                                               std::uint64_t seed,
                                               NodeId active_node_count = 0U);

OPIMProxyResult run_hist_proxy_clique_implicit(const HybridHypergraph& graph,
                                               Budget k,
                                               double epsilon,
                                               double delta,
                                               std::uint64_t seed,
                                               NodeId active_node_count = 0U);

ICGraph build_upper_relaxed_ic_graph(const HybridHypergraph& graph);
ICGraph build_upper_relaxed_aux_ic_graph(const HybridHypergraph& graph);
ICGraph build_edge_only_ic_graph(const HybridHypergraph& graph);
ICGraph build_clique_expansion_ic_graph(const HybridHypergraph& graph);

}  // namespace htc
