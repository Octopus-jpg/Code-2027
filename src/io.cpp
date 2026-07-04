#include "htc/io.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace htc {
namespace {

std::string trim(std::string s) {
    const auto is_not_space = [](unsigned char ch) { return !std::isspace(ch); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), is_not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), is_not_space).base(), s.end());
    return s;
}

std::vector<std::string> read_data_lines(const std::filesystem::path& file_path) {
    std::ifstream fin(file_path);
    if (!fin.is_open()) {
        throw std::runtime_error("cannot open file: " + file_path.string());
    }

    std::vector<std::string> lines;
    std::string raw;
    while (std::getline(fin, raw)) {
        const auto pos = raw.find('#');
        if (pos != std::string::npos) {
            raw.erase(pos);
        }
        std::string line = trim(raw);
        if (!line.empty()) {
            lines.push_back(std::move(line));
        }
    }
    return lines;
}

struct ListFileData {
    std::optional<std::string> header_comment;
    std::vector<std::string> data_lines;
};

ListFileData read_list_file(const std::filesystem::path& file_path) {
    std::ifstream fin(file_path);
    if (!fin.is_open()) {
        throw std::runtime_error("cannot open file: " + file_path.string());
    }

    ListFileData result;
    std::string raw;
    while (std::getline(fin, raw)) {
        std::string line = trim(raw);
        if (line.empty()) {
            continue;
        }

        if (line.front() == '#') {
            if (!result.header_comment.has_value()) {
                std::string header = trim(line.substr(1));
                if (!header.empty()) {
                    result.header_comment = std::move(header);
                }
            }
            continue;
        }

        const auto comment_pos = line.find('#');
        if (comment_pos != std::string::npos) {
            line = trim(line.substr(0, comment_pos));
        }
        if (!line.empty()) {
            result.data_lines.push_back(std::move(line));
        }
    }
    return result;
}

NodeId parse_node_id(const std::string& token, const char* context) {
    std::uint64_t value = 0;
    std::istringstream iss(token);
    if (!(iss >> value) || !(iss >> std::ws).eof()) {
        throw std::runtime_error(std::string("invalid node id in ") + context + ": " + token);
    }
    if (value >= static_cast<std::uint64_t>(kInvalidNodeId)) {
        throw std::runtime_error(std::string("node id overflow in ") + context + ": " + token);
    }
    return static_cast<NodeId>(value);
}

std::uint64_t parse_u64(const std::string& token, const char* context) {
    std::uint64_t value = 0;
    std::istringstream iss(token);
    if (!(iss >> value) || !(iss >> std::ws).eof()) {
        throw std::runtime_error(std::string("invalid integer in ") + context + ": " + token);
    }
    return value;
}

std::uint32_t parse_u32(const std::string& token, const char* context) {
    std::uint64_t value = 0;
    std::istringstream iss(token);
    if (!(iss >> value) || !(iss >> std::ws).eof()) {
        throw std::runtime_error(std::string("invalid integer in ") + context + ": " + token);
    }
    if (value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error(std::string("uint32 overflow in ") + context + ": " + token);
    }
    return static_cast<std::uint32_t>(value);
}

Prob parse_prob(const std::string& token, const char* context) {
    Prob p = 0.0;
    std::istringstream iss(token);
    if (!(iss >> p) || !(iss >> std::ws).eof()) {
        throw std::runtime_error(std::string("invalid probability in ") + context + ": " + token);
    }
    if (p < 0.0 || p > 1.0) {
        throw std::runtime_error(std::string("probability out of range in ") + context + ": " + token);
    }
    return p;
}

std::vector<std::string> split_tokens(const std::string& line) {
    std::vector<std::string> tokens;
    std::istringstream iss(line);
    std::string token;
    while (iss >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

struct EdgeFileHeader {
    std::uint32_t num_nodes = 0;
    Count num_edges = 0;
};

EdgeFileHeader parse_edge_file_header(const std::string& header,
                                      const std::string& file_label) {
    const std::vector<std::string> tokens = split_tokens(header);
    if (tokens.size() != 2U) {
        throw std::runtime_error(
            file_label + " header must be '# <node_number> <edge_number>'");
    }

    const std::string node_ctx = file_label + " header node_number";
    const std::string edge_ctx = file_label + " header edge_number";
    EdgeFileHeader parsed;
    parsed.num_nodes = parse_u32(tokens[0], node_ctx.c_str());
    parsed.num_edges = parse_u64(tokens[1], edge_ctx.c_str());
    return parsed;
}

Count parse_hyperedge_count_header(const std::string& header,
                                   const std::string& file_label) {
    const std::vector<std::string> tokens = split_tokens(header);
    if (tokens.size() != 1U) {
        throw std::runtime_error(file_label + " header must be '# <hyperedge_number>'");
    }
    const std::string count_ctx = file_label + " header hyperedge_number";
    return parse_u64(tokens[0], count_ctx.c_str());
}

void write_json_string(std::ostream& out, const std::string& value) {
    out << '"';
    for (unsigned char ch : value) {
        switch (ch) {
            case '"':
                out << "\\\"";
                break;
            case '\\':
                out << "\\\\";
                break;
            case '\b':
                out << "\\b";
                break;
            case '\f':
                out << "\\f";
                break;
            case '\n':
                out << "\\n";
                break;
            case '\r':
                out << "\\r";
                break;
            case '\t':
                out << "\\t";
                break;
            default:
                if (ch < 0x20U) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(ch) << std::dec << std::setfill(' ');
                } else {
                    out << static_cast<char>(ch);
                }
                break;
        }
    }
    out << '"';
}

void write_node_array(std::ostream& out, const std::vector<NodeId>& nodes) {
    out << '[';
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        if (i != 0U) {
            out << ',';
        }
        out << nodes[i];
    }
    out << ']';
}

std::ofstream open_jsonl_append_stream(const std::string& path) {
    const std::filesystem::path output_path(path);
    if (output_path.empty()) {
        throw std::invalid_argument("output path cannot be empty");
    }

    const auto parent = output_path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }

    std::ofstream out(output_path, std::ios::app);
    if (!out.is_open()) {
        throw std::runtime_error("cannot open output file: " + output_path.string());
    }
    out << std::setprecision(17);
    return out;
}

}  // namespace

HybridHypergraph load_hybrid_hypergraph_from_dir(const std::string& dir) {
    const std::filesystem::path base(dir);
    return load_hybrid_hypergraph_from_files((base / "edges.txt").string(),
                                             (base / "hyperedges.txt").string(),
                                             (base / "hyperedge_thresholds.txt").string());
}

HybridHypergraph load_hybrid_hypergraph_from_files(const std::string& edge_file,
                                                   const std::string& hyperedge_file,
                                                   const std::string& hyperedge_threshold_file) {
    const std::filesystem::path edge_path(edge_file);
    const std::filesystem::path hyperedge_path(hyperedge_file);
    const std::filesystem::path threshold_path(hyperedge_threshold_file);

    if (edge_path.empty() || hyperedge_path.empty() || threshold_path.empty()) {
        throw std::invalid_argument("edge/hyperedge/threshold file paths cannot be empty");
    }

    const std::string edge_label = edge_path.filename().empty()
                                       ? edge_path.string()
                                       : edge_path.filename().string();
    const std::string hyperedge_label = hyperedge_path.filename().empty()
                                            ? hyperedge_path.string()
                                            : hyperedge_path.filename().string();
    const std::string threshold_label = threshold_path.filename().empty()
                                            ? threshold_path.string()
                                            : threshold_path.filename().string();

    const ListFileData edge_data = read_list_file(edge_path);

    std::uint32_t n = 0U;
    Count declared_edge_count = 0U;
    bool has_declared_edge_count = false;
    if (edge_data.header_comment.has_value()) {
        const EdgeFileHeader parsed =
            parse_edge_file_header(edge_data.header_comment.value(), edge_label);
        n = parsed.num_nodes;
        declared_edge_count = parsed.num_edges;
        has_declared_edge_count = true;
    } else {
        // Backward-compatible fallback: read node count from legacy nodes.txt.
        const auto node_lines = read_data_lines(edge_path.parent_path() / "nodes.txt");
        if (node_lines.empty()) {
            throw std::runtime_error(
                edge_label + " must start with '# <node_number> <edge_number>' "
                "when nodes.txt is not provided");
        }
        if (node_lines.size() != 1U) {
            throw std::runtime_error("nodes.txt should contain exactly one numeric line");
        }
        n = parse_u32(node_lines.front(), "nodes.txt");
    }

    if (n >= kInvalidNodeId) {
        throw std::runtime_error("node_number exceeds NodeId range");
    }

    HybridHypergraph graph(static_cast<NodeId>(n));
    const std::string edge_node_ctx = edge_label + " node_id";
    const std::string edge_prob_ctx = edge_label + " probability";
    struct ParsedEdgeLine {
        NodeId src = 0U;
        NodeId dst = 0U;
        std::optional<Prob> explicit_prob;
    };
    std::vector<ParsedEdgeLine> parsed_edges;
    parsed_edges.reserve(edge_data.data_lines.size());
    std::vector<Count> indegree(static_cast<std::size_t>(n), 0U);
    bool all_edges_without_explicit_prob = !edge_data.data_lines.empty();

    for (const auto& line : edge_data.data_lines) {
        const std::vector<std::string> tokens = split_tokens(line);
        if (tokens.size() != 2U && tokens.size() != 3U) {
            throw std::runtime_error(
                "edge line must be 'u v' or 'u v prob': " + line);
        }
        const NodeId src = parse_node_id(tokens[0], edge_node_ctx.c_str());
        const NodeId dst = parse_node_id(tokens[1], edge_node_ctx.c_str());
        ParsedEdgeLine parsed;
        parsed.src = src;
        parsed.dst = dst;
        if (tokens.size() == 3U) {
            parsed.explicit_prob = parse_prob(tokens[2], edge_prob_ctx.c_str());
            all_edges_without_explicit_prob = false;
        }
        parsed_edges.push_back(parsed);
        ++indegree[static_cast<std::size_t>(dst)];
    }

    for (const ParsedEdgeLine& parsed : parsed_edges) {
        Prob prob = 1.0;
        if (parsed.explicit_prob.has_value()) {
            prob = parsed.explicit_prob.value();
        } else if (all_edges_without_explicit_prob) {
            const Count in_deg = indegree[static_cast<std::size_t>(parsed.dst)];
            if (in_deg == 0U) {
                throw std::logic_error("edge destination indegree is zero during default-probability assignment");
            }
            prob = 1.0 / static_cast<double>(in_deg);
        }
        graph.add_edge(parsed.src, parsed.dst, prob);
    }
    if (has_declared_edge_count &&
        static_cast<Count>(graph.num_edges()) != declared_edge_count) {
        throw std::runtime_error(edge_label + " declared edge_number does not match data lines");
    }

    const ListFileData hyperedge_data = read_list_file(hyperedge_path);
    Count declared_hyperedge_count = 0U;
    bool has_declared_hyperedge_count = false;
    if (hyperedge_data.header_comment.has_value()) {
        declared_hyperedge_count =
            parse_hyperedge_count_header(hyperedge_data.header_comment.value(), hyperedge_label);
        has_declared_hyperedge_count = true;
    }

    if (!std::filesystem::exists(threshold_path)) {
        throw std::runtime_error(
            "missing required file: " + threshold_path.string() +
            " (new format requires separate threshold file)");
    }
    const ListFileData threshold_data = read_list_file(threshold_path);
    Count declared_threshold_count = 0U;
    bool has_declared_threshold_count = false;
    if (threshold_data.header_comment.has_value()) {
        declared_threshold_count =
            parse_hyperedge_count_header(threshold_data.header_comment.value(), threshold_label);
        has_declared_threshold_count = true;
    }

    std::vector<std::uint32_t> thresholds;
    thresholds.reserve(threshold_data.data_lines.size());
    const std::string threshold_ctx = threshold_label + " threshold";
    for (const auto& line : threshold_data.data_lines) {
        const std::vector<std::string> tokens = split_tokens(line);
        if (tokens.size() != 1U) {
            throw std::runtime_error(
                threshold_label + " line must be a single threshold integer: " + line);
        }
        thresholds.push_back(parse_u32(tokens[0], threshold_ctx.c_str()));
    }

    if (has_declared_threshold_count &&
        static_cast<Count>(thresholds.size()) != declared_threshold_count) {
        throw std::runtime_error(
            threshold_label + " declared hyperedge_number does not match data lines");
    }
    if (thresholds.size() != hyperedge_data.data_lines.size()) {
        throw std::runtime_error(
            hyperedge_label + " and " + threshold_label + " data line counts do not match");
    }
    const std::string hyperedge_node_ctx = hyperedge_label + " node_id";

    for (std::size_t line_idx = 0; line_idx < hyperedge_data.data_lines.size(); ++line_idx) {
        const std::string& line = hyperedge_data.data_lines[line_idx];
        const std::vector<std::string> tokens = split_tokens(line);
        if (tokens.size() < 2U) {
            throw std::runtime_error(
                "hyperedge line must be 'v1 v2 ...' with at least two nodes: " + line);
        }

        std::vector<NodeId> nodes;
        nodes.reserve(tokens.size());
        for (const auto& token : tokens) {
            nodes.push_back(parse_node_id(token, hyperedge_node_ctx.c_str()));
        }

        graph.add_hyperedge(std::move(nodes), thresholds[line_idx]);
    }
    if (has_declared_hyperedge_count &&
        static_cast<Count>(graph.num_hyperedges()) != declared_hyperedge_count) {
        throw std::runtime_error(
            hyperedge_label + " declared hyperedge_number does not match data lines");
    }

    graph.build_indices();
    graph.validate();
    return graph;
}

std::vector<NodeId> parse_seed_list(const std::string& text) {
    std::vector<NodeId> seeds;
    std::unordered_set<NodeId> seen;

    std::string line;
    std::istringstream lss(text);
    while (std::getline(lss, line)) {
        const auto comment_pos = line.find('#');
        if (comment_pos != std::string::npos) {
            line = line.substr(0, comment_pos);
        }
        line = trim(line);
        if (line.empty()) {
            continue;
        }
        for (char& ch : line) {
            if (ch == ',') {
                ch = ' ';
            }
        }
        std::istringstream tss(line);
        std::string token;
        while (tss >> token) {
            const NodeId node = parse_node_id(token, "seed list");
            if (seen.insert(node).second) {
                seeds.push_back(node);
            }
        }
    }

    return seeds;
}

void append_ie_result_jsonl(const std::string& path,
                            const std::string& data_dir,
                            const std::vector<NodeId>& seeds,
                            const IEConfig& config,
                            const IEResult& result) {
    std::ofstream out = open_jsonl_append_stream(path);
    out << '{';
    out << "\"task\":\"ie\"";
    out << ",\"data_dir\":";
    write_json_string(out, data_dir);
    out << ",\"method\":";
    write_json_string(out, result.method);
    out << ",\"seed_set\":";
    write_node_array(out, seeds);
    out << ",\"epsilon\":" << config.epsilon;
    out << ",\"delta\":" << config.delta;
    out << ",\"fixed_samples_config\":" << config.fixed_samples;
    out << ",\"mc_simulations_config\":" << config.mc_simulations;
    out << ",\"rr_samples_config\":" << config.rr_samples;
    out << ",\"random_seed\":" << config.seed;
    out << ",\"estimate\":" << result.estimate;
    out << ",\"samples\":" << result.samples;
    out << ",\"satisfied\":" << result.satisfied;
    out << ",\"seconds\":" << result.seconds;
    out << "}\n";
}

void append_im_result_jsonl(const std::string& path,
                            const std::string& data_dir,
                            const std::string& method,
                            const IMConfig& config,
                            const IMResult& result) {
    std::ofstream out = open_jsonl_append_stream(path);
    out << '{';
    out << "\"task\":\"im\"";
    out << ",\"data_dir\":";
    write_json_string(out, data_dir);
    out << ",\"method\":";
    write_json_string(out, method);
    out << ",\"k\":" << config.k;
    out << ",\"rtw_samples_config\":" << config.rtw_samples;
    out << ",\"rtw_training_samples_config\":" << config.rtw_training_samples;
    out << ",\"mc_simulations_config\":" << config.mc_simulations;
    out << ",\"mc_greedy_simulations_config\":" << config.mc_greedy_simulations;
    out << ",\"moea_population_size_config\":" << config.moea_population_size;
    out << ",\"moea_offspring_size_config\":" << config.moea_offspring_size;
    out << ",\"moea_generations_config\":" << config.moea_generations;
    out << ",\"moea_fitness_simulations_config\":" << config.moea_fitness_simulations;
    out << ",\"moea_fitness_simulations_cap_config\":" << config.moea_fitness_simulations_cap;
    out << ",\"moea_tournament_size_config\":" << config.moea_tournament_size;
    out << ",\"moea_mutation_rate_config\":" << config.moea_mutation_rate;
    out << ",\"moea_crossover_rate_config\":" << config.moea_crossover_rate;
    out << ",\"epsilon\":" << config.epsilon;
    out << ",\"delta\":" << config.delta;
    out << ",\"lambda\":" << config.lambda;
    out << ",\"random_seed\":" << config.seed;
    out << ",\"use_progress_tiebreak\":" << (config.use_progress_tiebreak ? "true" : "false");
    out << ",\"use_final_compare\":" << (config.use_final_compare ? "true" : "false");
    out << ",\"use_min_b_ablation\":" << (config.use_min_b_ablation ? "true" : "false");
    out << ",\"seeds\":";
    write_node_array(out, result.seeds);
    out << ",\"rtw_satisfied_samples\":" << result.rtw_satisfied_samples;
    out << ",\"rtw_total_samples\":" << result.rtw_total_samples;
    out << ",\"rtw_hit_rate\":" << result.rtw_hit_rate;
    out << ",\"rtw_sample_estimate\":" << result.rtw_sample_estimate;
    out << ",\"rtw_estimate_over_B\":" << result.rtw_estimate_over_B;
    out << ",\"empirical_over_B\":" << result.empirical_over_B;
    out << ",\"rtw_total_formula_nodes\":" << result.rtw_total_formula_nodes;
    out << ",\"rtw_total_dependencies\":" << result.rtw_total_dependencies;
    out << ",\"rtw_total_gates\":" << result.rtw_total_gates;
    out << ",\"rtw_total_gate_input_incidences\":" << result.rtw_total_gate_input_incidences;
    out << ",\"rtw_total_witness_size\":" << result.rtw_total_witness_size;
    out << ",\"rtw_avg_formula_nodes\":" << result.rtw_avg_formula_nodes;
    out << ",\"rtw_avg_dependencies\":" << result.rtw_avg_dependencies;
    out << ",\"rtw_avg_gates\":" << result.rtw_avg_gates;
    out << ",\"rtw_avg_gate_input_incidences\":" << result.rtw_avg_gate_input_incidences;
    out << ",\"rtw_avg_witness_size\":" << result.rtw_avg_witness_size;
    out << ",\"rtw_occurrence_index_entries\":" << result.rtw_occurrence_index_entries;
    out << ",\"rtw_gain_cache_initial_recomputations\":"
        << result.rtw_gain_cache_initial_recomputations;
    out << ",\"rtw_gain_cache_update_recomputations\":"
        << result.rtw_gain_cache_update_recomputations;
    out << ",\"rtw_seed_additions\":" << result.rtw_seed_additions;
    out << ",\"rtw_affected_samples\":" << result.rtw_affected_samples;
    out << ",\"rtw_affected_gates\":" << result.rtw_affected_gates;
    out << ",\"rtw_activated_nodes\":" << result.rtw_activated_nodes;
    out << ",\"rtw_dirty_candidates\":" << result.rtw_dirty_candidates;
    out << ",\"rtw_max_dirty_candidates_per_seed\":"
        << result.rtw_max_dirty_candidates_per_seed;
    out << ",\"rtw_satisfied_sample_dirty_expansions\":"
        << result.rtw_satisfied_sample_dirty_expansions;
    out << ",\"rtw_fallback_sample_dirty_expansions\":"
        << result.rtw_fallback_sample_dirty_expansions;
    out << ",\"rtw_avg_dirty_candidates_per_seed\":"
        << result.rtw_avg_dirty_candidates_per_seed;
    out << ",\"rtw_avg_affected_samples_per_seed\":"
        << result.rtw_avg_affected_samples_per_seed;
    out << ",\"rtw_gain_cache_initialization_seconds\":"
        << result.rtw_gain_cache_initialization_seconds;
    out << ",\"rtw_gain_cache_update_seconds\":" << result.rtw_gain_cache_update_seconds;
    out << ",\"rtw_validation_samples_s_prime\":"
        << result.rtw_validation_samples_s_prime;
    out << ",\"rtw_validation_samples_s_plus\":"
        << result.rtw_validation_samples_s_plus;
    out << ",\"rr_samples\":" << result.rr_samples;
    out << ",\"eval_samples\":" << result.eval_samples;
    out << ",\"upper_bound_B\":" << result.upper_bound_B;
    out << ",\"certificate_alpha\":" << result.certificate_alpha;
    out << ",\"empirical_influence\":" << result.empirical_influence;
    out << ",\"time_opimc_seconds\":" << result.time_opimc_seconds;
    out << ",\"time_sampling_seconds\":" << result.time_sampling_seconds;
    out << ",\"time_greedy_seconds\":" << result.time_greedy_seconds;
    out << ",\"time_compare_seconds\":" << result.time_compare_seconds;
    out << ",\"time_total_seconds\":" << result.time_total_seconds;
    out << ",\"peak_rss_kb\":" << result.peak_rss_kb;
    out << ",\"seconds\":" << result.seconds;
    out << "}\n";
}

}  // namespace htc
