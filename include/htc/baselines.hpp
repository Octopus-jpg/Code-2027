#pragma once

#include "htc/graph.hpp"
#include "htc/ic.hpp"
#include "htc/types.hpp"

namespace htc {

class MCGreedyBaseline {
public:
    explicit MCGreedyBaseline(const HybridHypergraph& graph);
    IMResult run(const IMConfig& config) const;

private:
    const HybridHypergraph& graph_;
};

class URIMBaseline {
public:
    explicit URIMBaseline(const HybridHypergraph& graph);
    IMResult run(const IMConfig& config) const;

private:
    const HybridHypergraph& graph_;
};

class URIMHistBaseline {
public:
    explicit URIMHistBaseline(const HybridHypergraph& graph);
    IMResult run(const IMConfig& config) const;

private:
    const HybridHypergraph& graph_;
};

class EdgeOnlyBaseline {
public:
    explicit EdgeOnlyBaseline(const HybridHypergraph& graph);
    IMResult run(const IMConfig& config) const;

private:
    const HybridHypergraph& graph_;
};

class EdgeOnlyHistBaseline {
public:
    explicit EdgeOnlyHistBaseline(const HybridHypergraph& graph);
    IMResult run(const IMConfig& config) const;

private:
    const HybridHypergraph& graph_;
};

class CliqueExpansionBaseline {
public:
    explicit CliqueExpansionBaseline(const HybridHypergraph& graph);
    IMResult run(const IMConfig& config) const;

private:
    const HybridHypergraph& graph_;
};

class CliqueExpansionHistBaseline {
public:
    explicit CliqueExpansionHistBaseline(const HybridHypergraph& graph);
    IMResult run(const IMConfig& config) const;

private:
    const HybridHypergraph& graph_;
};

class HybridDegreeBaseline {
public:
    explicit HybridDegreeBaseline(const HybridHypergraph& graph);
    IMResult run(const IMConfig& config,
                 double edge_weight = 1.0,
                 double hyperedge_weight = 1.0) const;

private:
    const HybridHypergraph& graph_;
};

class HCI1TMBaseline {
public:
    explicit HCI1TMBaseline(const HybridHypergraph& graph);
    IMResult run(const IMConfig& config) const;

private:
    const HybridHypergraph& graph_;
};

class HCI2TMBaseline {
public:
    explicit HCI2TMBaseline(const HybridHypergraph& graph);
    IMResult run(const IMConfig& config) const;

private:
    const HybridHypergraph& graph_;
};

class HNMOEAHTCBaseline {
public:
    explicit HNMOEAHTCBaseline(const HybridHypergraph& graph);
    IMResult run(const IMConfig& config) const;

private:
    const HybridHypergraph& graph_;
};

}  // namespace htc
