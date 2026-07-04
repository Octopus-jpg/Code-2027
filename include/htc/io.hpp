#pragma once

#include <string>
#include <vector>

#include "htc/graph.hpp"
#include "htc/types.hpp"

namespace htc {

HybridHypergraph load_hybrid_hypergraph_from_dir(const std::string& dir);
HybridHypergraph load_hybrid_hypergraph_from_files(const std::string& edge_file,
                                                   const std::string& hyperedge_file,
                                                   const std::string& hyperedge_threshold_file);
std::vector<NodeId> parse_seed_list(const std::string& text);
void append_ie_result_jsonl(const std::string& path,
                            const std::string& data_dir,
                            const std::vector<NodeId>& seeds,
                            const IEConfig& config,
                            const IEResult& result);
void append_im_result_jsonl(const std::string& path,
                            const std::string& data_dir,
                            const std::string& method,
                            const IMConfig& config,
                            const IMResult& result);

}  // namespace htc
