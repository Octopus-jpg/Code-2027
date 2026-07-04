#include "htc/graph.hpp"

#include <algorithm>
#include <stdexcept>

namespace htc {

HybridHypergraph::HybridHypergraph(NodeId n) : n_(n) {
    in_edges_.resize(n_);
    out_edges_.resize(n_);
    incident_hyperedges_.resize(n_);
}

NodeId HybridHypergraph::num_nodes() const {
    return n_;
}

EdgeId HybridHypergraph::num_edges() const {
    if (edges_.size() >= static_cast<std::size_t>(kInvalidEdgeId)) {
        throw std::overflow_error("number of edges exceeds EdgeId range");
    }
    return static_cast<EdgeId>(edges_.size());
}

HyperedgeId HybridHypergraph::num_hyperedges() const {
    if (hyperedges_.size() >= static_cast<std::size_t>(kInvalidHyperedgeId)) {
        throw std::overflow_error("number of hyperedges exceeds HyperedgeId range");
    }
    return static_cast<HyperedgeId>(hyperedges_.size());
}

EdgeId HybridHypergraph::add_edge(NodeId u, NodeId v, Prob p) {
    validate_node(u);
    validate_node(v);
    validate_probability(p);
    if (edges_.size() >= static_cast<std::size_t>(kInvalidEdgeId)) {
        throw std::overflow_error("cannot add more edges: EdgeId overflow");
    }

    const EdgeId id = static_cast<EdgeId>(edges_.size());
    edges_.push_back(DirectedEdge{u, v, p});
    return id;
}

HyperedgeId HybridHypergraph::add_hyperedge(std::vector<NodeId> nodes, std::uint32_t threshold) {
    if (nodes.size() < 2U) {
        throw std::invalid_argument("hyperedge must contain at least two nodes");
    }
    for (NodeId v : nodes) {
        validate_node(v);
    }

    std::sort(nodes.begin(), nodes.end());
    nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());

    if (nodes.size() < 2U) {
        throw std::invalid_argument("hyperedge must contain at least two distinct nodes");
    }
    if (threshold < 1U || threshold >= nodes.size()) {
        throw std::invalid_argument("hyperedge threshold must satisfy 1 <= threshold <= |h|-1");
    }
    if (hyperedges_.size() >= static_cast<std::size_t>(kInvalidHyperedgeId)) {
        throw std::overflow_error("cannot add more hyperedges: HyperedgeId overflow");
    }

    const HyperedgeId id = static_cast<HyperedgeId>(hyperedges_.size());
    hyperedges_.push_back(Hyperedge{id, std::move(nodes), threshold});
    return id;
}

void HybridHypergraph::build_indices() {
    in_edges_.assign(n_, {});
    out_edges_.assign(n_, {});
    incident_hyperedges_.assign(n_, {});

    for (std::size_t i = 0; i < edges_.size(); ++i) {
        const auto& e = edges_[i];
        const EdgeId eid = static_cast<EdgeId>(i);
        out_edges_[e.src].push_back(eid);
        in_edges_[e.dst].push_back(eid);
    }

    for (std::size_t i = 0; i < hyperedges_.size(); ++i) {
        const auto& h = hyperedges_[i];
        const HyperedgeId hid = static_cast<HyperedgeId>(i);
        for (NodeId v : h.nodes) {
            incident_hyperedges_[v].push_back(hid);
        }
    }
}

void HybridHypergraph::validate() const {
    if (n_ == kInvalidNodeId) {
        throw std::invalid_argument("num_nodes cannot be kInvalidNodeId");
    }

    for (std::size_t i = 0; i < edges_.size(); ++i) {
        const auto& e = edges_[i];
        if (e.src >= n_ || e.dst >= n_) {
            throw std::invalid_argument("edge endpoint out of range");
        }
        validate_probability(e.prob);
    }

    for (std::size_t i = 0; i < hyperedges_.size(); ++i) {
        const auto& h = hyperedges_[i];
        if (h.id != static_cast<HyperedgeId>(i)) {
            throw std::invalid_argument("hyperedge id mismatch");
        }
        if (h.nodes.size() < 2U) {
            throw std::invalid_argument("hyperedge must contain at least two nodes");
        }
        if (h.threshold < 1U || h.threshold >= h.nodes.size()) {
            throw std::invalid_argument("invalid hyperedge threshold");
        }
        for (NodeId v : h.nodes) {
            if (v >= n_) {
                throw std::invalid_argument("hyperedge node out of range");
            }
        }
        if (!std::is_sorted(h.nodes.begin(), h.nodes.end())) {
            throw std::invalid_argument("hyperedge nodes must be sorted");
        }
        if (std::adjacent_find(h.nodes.begin(), h.nodes.end()) != h.nodes.end()) {
            throw std::invalid_argument("hyperedge nodes must be unique");
        }
    }

    if (in_edges_.size() != n_ || out_edges_.size() != n_ || incident_hyperedges_.size() != n_) {
        throw std::invalid_argument("index vectors size mismatch; call build_indices() after edits");
    }

    for (NodeId v = 0; v < n_; ++v) {
        for (EdgeId eid : in_edges_[v]) {
            if (eid >= edges_.size()) {
                throw std::invalid_argument("in-edge index out of range");
            }
            if (edges_[eid].dst != v) {
                throw std::invalid_argument("in-edge index inconsistent with edge destination");
            }
        }
        for (EdgeId eid : out_edges_[v]) {
            if (eid >= edges_.size()) {
                throw std::invalid_argument("out-edge index out of range");
            }
            if (edges_[eid].src != v) {
                throw std::invalid_argument("out-edge index inconsistent with edge source");
            }
        }
        for (HyperedgeId hid : incident_hyperedges_[v]) {
            if (hid >= hyperedges_.size()) {
                throw std::invalid_argument("incident hyperedge index out of range");
            }
            const auto& nodes = hyperedges_[hid].nodes;
            if (!std::binary_search(nodes.begin(), nodes.end(), v)) {
                throw std::invalid_argument("incident hyperedge index inconsistent with hyperedge nodes");
            }
        }
    }
}

const DirectedEdge& HybridHypergraph::edge(EdgeId id) const {
    if (id >= edges_.size()) {
        throw std::out_of_range("edge id out of range");
    }
    return edges_[id];
}

const Hyperedge& HybridHypergraph::hyperedge(HyperedgeId id) const {
    if (id >= hyperedges_.size()) {
        throw std::out_of_range("hyperedge id out of range");
    }
    return hyperedges_[id];
}

const std::vector<DirectedEdge>& HybridHypergraph::edges() const {
    return edges_;
}

const std::vector<Hyperedge>& HybridHypergraph::hyperedges() const {
    return hyperedges_;
}

const std::vector<EdgeId>& HybridHypergraph::in_edges(NodeId v) const {
    validate_node(v);
    return in_edges_[v];
}

const std::vector<EdgeId>& HybridHypergraph::out_edges(NodeId u) const {
    validate_node(u);
    return out_edges_[u];
}

const std::vector<HyperedgeId>& HybridHypergraph::incident_hyperedges(NodeId v) const {
    validate_node(v);
    return incident_hyperedges_[v];
}

void HybridHypergraph::validate_probability(Prob p) {
    if (p < 0.0 || p > 1.0) {
        throw std::invalid_argument("edge probability must be in [0, 1]");
    }
}

void HybridHypergraph::validate_node(NodeId v) const {
    if (v >= n_) {
        throw std::out_of_range("node id out of range");
    }
}

}  // namespace htc
