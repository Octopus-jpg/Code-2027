#pragma once

#include "htc/graph.hpp"
#include "htc/rtw.hpp"
#include "htc/types.hpp"

namespace htc {

class RTWGreedy {
public:
    explicit RTWGreedy(const HybridHypergraph& graph);

    IMResult run(const IMConfig& config) const;
    IMResult run_practical(const IMConfig& config) const;

private:
    const HybridHypergraph& graph_;
};

}  // namespace htc
