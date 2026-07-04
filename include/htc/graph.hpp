#pragma once

#include <vector>

#include "htc/types.hpp"

namespace htc {

struct DirectedEdge {
    NodeId src = kInvalidNodeId;
    NodeId dst = kInvalidNodeId;
    Prob prob = 0.0;
};

struct Hyperedge {
    HyperedgeId id = kInvalidHyperedgeId;
    std::vector<NodeId> nodes;
    std::uint32_t threshold = 1;
};

class HybridHypergraph {
public:
    explicit HybridHypergraph(NodeId n = 0);

    NodeId num_nodes() const;
    EdgeId num_edges() const;
    HyperedgeId num_hyperedges() const;

    EdgeId add_edge(NodeId u, NodeId v, Prob p);
    HyperedgeId add_hyperedge(std::vector<NodeId> nodes, std::uint32_t threshold);

    void build_indices();
    void validate() const;

    const DirectedEdge& edge(EdgeId id) const;
    const Hyperedge& hyperedge(HyperedgeId id) const;

    const std::vector<DirectedEdge>& edges() const;
    const std::vector<Hyperedge>& hyperedges() const;

    const std::vector<EdgeId>& in_edges(NodeId v) const;
    const std::vector<EdgeId>& out_edges(NodeId u) const;
    const std::vector<HyperedgeId>& incident_hyperedges(NodeId v) const;

private:
    static void validate_probability(Prob p);
    void validate_node(NodeId v) const;

    NodeId n_ = 0;
    std::vector<DirectedEdge> edges_;
    std::vector<Hyperedge> hyperedges_;
    std::vector<std::vector<EdgeId>> in_edges_;
    std::vector<std::vector<EdgeId>> out_edges_;
    std::vector<std::vector<HyperedgeId>> incident_hyperedges_;
};

}  // namespace htc
