#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "htc/baselines.hpp"
#include "htc/im.hpp"
#include "htc/random.hpp"
#include "htc/rtw.hpp"
#include "htc/synthetic.hpp"

namespace {

std::string get_arg(const std::vector<std::string>& args,
                    const std::string& key,
                    const std::string& default_value = "") {
    for (std::size_t i = 0; i + 1 < args.size(); ++i) {
        if (args[i] == key) {
            return args[i + 1];
        }
    }
    return default_value;
}

bool has_arg(const std::vector<std::string>& args, const std::string& key) {
    for (const auto& a : args) {
        if (a == key) {
            return true;
        }
    }
    return false;
}

std::uint64_t to_u64(const std::string& s, const char* name) {
    try {
        return static_cast<std::uint64_t>(std::stoull(s));
    } catch (...) {
        throw std::invalid_argument(std::string("invalid value for ") + name + ": " + s);
    }
}

std::string sanitize_token(const std::string& raw) {
    if (raw.empty()) {
        return "empty";
    }
    std::string out;
    out.reserve(raw.size());
    for (unsigned char ch : raw) {
        if (std::isalnum(ch) || ch == '-' || ch == '_' || ch == '.') {
            out.push_back(static_cast<char>(ch));
        } else {
            out.push_back('_');
        }
    }
    if (out.empty()) {
        return "empty";
    }
    return out;
}

std::string format_double_token(double value) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6) << value;
    std::string token = oss.str();
    while (token.size() > 1U && token.back() == '0') {
        token.pop_back();
    }
    if (!token.empty() && token.back() == '.') {
        token.pop_back();
    }
    return sanitize_token(token);
}

std::string bool_token(bool value) { return value ? "1" : "0"; }

std::string build_default_output_path(const htc::SyntheticConfig& gen,
                                      const htc::IMConfig& im_cfg,
                                      htc::Count bench_samples,
                                      bool skip_baselines) {
    std::filesystem::path out = std::filesystem::current_path();
    out /= "result";
    out /= "bench_m5";
    out /= "n_" + std::to_string(gen.num_nodes);
    out /= "m_" + std::to_string(gen.num_edges);
    out /= "q_" + std::to_string(gen.num_hyperedges);
    out /= "hsize_" + std::to_string(gen.min_hyperedge_size) + "_" +
           std::to_string(gen.max_hyperedge_size);
    out /= "ep_" + format_double_token(gen.min_edge_prob) + "_" +
           format_double_token(gen.max_edge_prob);
    out /= "k_" + std::to_string(im_cfg.k);
    out /= "rtw_" + std::to_string(im_cfg.rtw_samples);
    out /= "mc_" + std::to_string(im_cfg.mc_simulations);
    out /= "bench_" + std::to_string(bench_samples);
    out /= "seed_" + std::to_string(gen.seed);
    out /= "skip_baselines_" + bool_token(skip_baselines);
    out /= "results.jsonl";
    return out.string();
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

std::ofstream open_text_append_stream(const std::string& path) {
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
    out << std::setprecision(10);
    return out;
}

std::string derive_readable_output_path(const std::string& output_path) {
    const std::filesystem::path path(output_path);
    const std::filesystem::path parent = path.parent_path();
    const std::string filename = path.filename().string();
    const std::string suffix = ".jsonl";
    if (filename.size() >= suffix.size() &&
        filename.compare(filename.size() - suffix.size(), suffix.size(), suffix) == 0) {
        const std::string stem = filename.substr(0, filename.size() - suffix.size());
        return (parent / (stem + ".readable.txt")).string();
    }
    return path.string() + ".readable.txt";
}

void write_optional_method_result(std::ostream& out,
                                  const std::string& prefix,
                                  bool present,
                                  const htc::IMResult& result) {
    out << ",\"" << prefix << "_present\":" << (present ? "true" : "false");
    if (!present) {
        return;
    }
    out << ",\"" << prefix << "_seconds\":" << result.seconds;
    out << ",\"" << prefix << "_empirical_influence\":" << result.empirical_influence;
    out << ",\"" << prefix << "_seed_count\":" << result.seeds.size();
}

void append_bench_result_jsonl(const std::string& path,
                               const htc::SyntheticConfig& gen,
                               const htc::IMConfig& im_cfg,
                               htc::Count bench_samples,
                               bool skip_baselines,
                               const htc::HybridHypergraph& graph,
                               const htc::RTWStats& sampling_stats,
                               double rtw_sampling_seconds,
                               const htc::IMResult& rtw_result,
                               bool ur_present,
                               const htc::IMResult& ur_result,
                               bool hybrid_degree_present,
                               const htc::IMResult& hybrid_degree_result) {
    std::ofstream out = open_jsonl_append_stream(path);
    out << '{';
    out << "\"task\":\"bench_m5\"";
    out << ",\"generator\":\"synthetic\"";
    out << ",\"synthetic_num_nodes\":" << gen.num_nodes;
    out << ",\"synthetic_num_edges\":" << gen.num_edges;
    out << ",\"synthetic_num_hyperedges\":" << gen.num_hyperedges;
    out << ",\"synthetic_min_hyperedge_size\":" << gen.min_hyperedge_size;
    out << ",\"synthetic_max_hyperedge_size\":" << gen.max_hyperedge_size;
    out << ",\"synthetic_min_edge_prob\":" << gen.min_edge_prob;
    out << ",\"synthetic_max_edge_prob\":" << gen.max_edge_prob;
    out << ",\"random_seed\":" << gen.seed;
    out << ",\"graph_num_nodes\":" << graph.num_nodes();
    out << ",\"graph_num_edges\":" << graph.num_edges();
    out << ",\"graph_num_hyperedges\":" << graph.num_hyperedges();
    out << ",\"k\":" << im_cfg.k;
    out << ",\"rtw_samples_config\":" << im_cfg.rtw_samples;
    out << ",\"mc_simulations_config\":" << im_cfg.mc_simulations;
    out << ",\"bench_samples\":" << bench_samples;
    out << ",\"skip_baselines\":" << (skip_baselines ? "true" : "false");
    out << ",\"rtw_sampling_seconds\":" << rtw_sampling_seconds;
    out << ",\"rtw_sampling_samples_per_sec\":"
        << (static_cast<double>(bench_samples) / std::max(1e-12, rtw_sampling_seconds));
    out << ",\"rtw_sampling_avg_formula_nodes\":" << sampling_stats.avg_formula_nodes();
    out << ",\"rtw_sampling_avg_dependencies\":" << sampling_stats.avg_dependencies();
    out << ",\"rtw_sampling_avg_gates\":" << sampling_stats.avg_gates();
    out << ",\"rtw_sampling_avg_gate_input_incidences\":"
        << sampling_stats.avg_gate_input_incidences();
    out << ",\"rtw_sampling_avg_witness_size\":" << sampling_stats.avg_witness_size();
    out << ",\"rtw_greedy_seconds\":" << rtw_result.seconds;
    out << ",\"rtw_greedy_empirical_influence\":" << rtw_result.empirical_influence;
    out << ",\"rtw_greedy_seed_count\":" << rtw_result.seeds.size();
    out << ",\"rtw_greedy_dirty_candidates\":" << rtw_result.rtw_dirty_candidates;
    out << ",\"rtw_greedy_avg_dirty_candidates_per_seed\":"
        << rtw_result.rtw_avg_dirty_candidates_per_seed;
    out << ",\"rtw_greedy_max_dirty_candidates_per_seed\":"
        << rtw_result.rtw_max_dirty_candidates_per_seed;
    out << ",\"rtw_greedy_occurrence_index_entries\":"
        << rtw_result.rtw_occurrence_index_entries;
    out << ",\"rtw_greedy_gain_cache_initial_recomputations\":"
        << rtw_result.rtw_gain_cache_initial_recomputations;
    out << ",\"rtw_greedy_gain_cache_update_recomputations\":"
        << rtw_result.rtw_gain_cache_update_recomputations;
    out << ",\"rtw_greedy_gain_cache_initialization_seconds\":"
        << rtw_result.rtw_gain_cache_initialization_seconds;
    out << ",\"rtw_greedy_gain_cache_update_seconds\":"
        << rtw_result.rtw_gain_cache_update_seconds;
    write_optional_method_result(out, "ur_im", ur_present, ur_result);
    write_optional_method_result(out, "hybrid_degree", hybrid_degree_present, hybrid_degree_result);
    out << "}\n";
}

void append_bench_result_readable(const std::string& path,
                                  const htc::SyntheticConfig& gen,
                                  const htc::IMConfig& im_cfg,
                                  htc::Count bench_samples,
                                  bool skip_baselines,
                                  const htc::HybridHypergraph& graph,
                                  const htc::RTWStats& sampling_stats,
                                  double rtw_sampling_seconds,
                                  const htc::IMResult& rtw_result,
                                  bool ur_present,
                                  const htc::IMResult& ur_result,
                                  bool hybrid_degree_present,
                                  const htc::IMResult& hybrid_degree_result) {
    std::ofstream out = open_text_append_stream(path);
    out << "task=bench_m5\n";
    out << "generator=synthetic\n";
    out << "synthetic_num_nodes=" << gen.num_nodes << "\n";
    out << "synthetic_num_edges=" << gen.num_edges << "\n";
    out << "synthetic_num_hyperedges=" << gen.num_hyperedges << "\n";
    out << "synthetic_min_hyperedge_size=" << gen.min_hyperedge_size << "\n";
    out << "synthetic_max_hyperedge_size=" << gen.max_hyperedge_size << "\n";
    out << "synthetic_min_edge_prob=" << gen.min_edge_prob << "\n";
    out << "synthetic_max_edge_prob=" << gen.max_edge_prob << "\n";
    out << "random_seed=" << gen.seed << "\n";
    out << "graph_num_nodes=" << graph.num_nodes() << "\n";
    out << "graph_num_edges=" << graph.num_edges() << "\n";
    out << "graph_num_hyperedges=" << graph.num_hyperedges() << "\n";
    out << "k=" << im_cfg.k << "\n";
    out << "rtw_samples_config=" << im_cfg.rtw_samples << "\n";
    out << "mc_simulations_config=" << im_cfg.mc_simulations << "\n";
    out << "bench_samples=" << bench_samples << "\n";
    out << "skip_baselines=" << (skip_baselines ? 1 : 0) << "\n";
    out << "rtw_sampling_seconds=" << rtw_sampling_seconds << "\n";
    out << "rtw_sampling_samples_per_sec="
        << (static_cast<double>(bench_samples) / std::max(1e-12, rtw_sampling_seconds)) << "\n";
    out << "rtw_sampling_avg_formula_nodes=" << sampling_stats.avg_formula_nodes() << "\n";
    out << "rtw_sampling_avg_dependencies=" << sampling_stats.avg_dependencies() << "\n";
    out << "rtw_sampling_avg_gates=" << sampling_stats.avg_gates() << "\n";
    out << "rtw_sampling_avg_gate_input_incidences=" << sampling_stats.avg_gate_input_incidences()
        << "\n";
    out << "rtw_sampling_avg_witness_size=" << sampling_stats.avg_witness_size() << "\n";
    out << "rtw_greedy_seconds=" << rtw_result.seconds << "\n";
    out << "rtw_greedy_empirical_influence=" << rtw_result.empirical_influence << "\n";
    out << "rtw_greedy_seed_count=" << rtw_result.seeds.size() << "\n";
    out << "rtw_greedy_dirty_candidates=" << rtw_result.rtw_dirty_candidates << "\n";
    out << "rtw_greedy_avg_dirty_candidates_per_seed="
        << rtw_result.rtw_avg_dirty_candidates_per_seed << "\n";
    out << "rtw_greedy_max_dirty_candidates_per_seed="
        << rtw_result.rtw_max_dirty_candidates_per_seed << "\n";
    out << "rtw_greedy_occurrence_index_entries=" << rtw_result.rtw_occurrence_index_entries
        << "\n";
    out << "rtw_greedy_gain_cache_initial_recomputations="
        << rtw_result.rtw_gain_cache_initial_recomputations << "\n";
    out << "rtw_greedy_gain_cache_update_recomputations="
        << rtw_result.rtw_gain_cache_update_recomputations << "\n";
    out << "rtw_greedy_gain_cache_initialization_seconds="
        << rtw_result.rtw_gain_cache_initialization_seconds << "\n";
    out << "rtw_greedy_gain_cache_update_seconds=" << rtw_result.rtw_gain_cache_update_seconds
        << "\n";
    out << "ur_im_present=" << (ur_present ? 1 : 0) << "\n";
    if (ur_present) {
        out << "ur_im_seconds=" << ur_result.seconds << "\n";
        out << "ur_im_empirical_influence=" << ur_result.empirical_influence << "\n";
        out << "ur_im_seed_count=" << ur_result.seeds.size() << "\n";
    }
    out << "hybrid_degree_present=" << (hybrid_degree_present ? 1 : 0) << "\n";
    if (hybrid_degree_present) {
        out << "hybrid_degree_seconds=" << hybrid_degree_result.seconds << "\n";
        out << "hybrid_degree_empirical_influence=" << hybrid_degree_result.empirical_influence
            << "\n";
        out << "hybrid_degree_seed_count=" << hybrid_degree_result.seeds.size() << "\n";
    }
    out << "---\n";
}

void print_usage() {
    std::cout
        << "Usage:\n"
        << "  bench_m5 [options]\n"
        << "Options:\n"
        << "  --n <num_nodes>\n"
        << "  --m <num_edges>\n"
        << "  --q <num_hyperedges>\n"
        << "  --min-hsize <int>\n"
        << "  --max-hsize <int>\n"
        << "  --k <budget>\n"
        << "  --rtw-samples <int>\n"
        << "  --bench-samples <int>\n"
        << "  --mc-simulations <int>\n"
        << "  --seed <int>\n"
        << "  --skip-baselines\n"
        << "  --output <path> (default: auto path under ./result)\n"
        << "  (human-readable metrics will also be appended to *.readable.txt)\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::vector<std::string> args;
        args.reserve(static_cast<std::size_t>(argc));
        for (int i = 1; i < argc; ++i) {
            args.emplace_back(argv[i]);
        }

        if (has_arg(args, "--help") || has_arg(args, "-h")) {
            print_usage();
            return EXIT_SUCCESS;
        }

        htc::SyntheticConfig gen;
        if (const auto v = get_arg(args, "--n"); !v.empty()) {
            gen.num_nodes = static_cast<htc::NodeId>(to_u64(v, "--n"));
        }
        if (const auto v = get_arg(args, "--m"); !v.empty()) {
            gen.num_edges = static_cast<htc::Count>(to_u64(v, "--m"));
        }
        if (const auto v = get_arg(args, "--q"); !v.empty()) {
            gen.num_hyperedges = static_cast<htc::Count>(to_u64(v, "--q"));
        }
        if (const auto v = get_arg(args, "--min-hsize"); !v.empty()) {
            gen.min_hyperedge_size = static_cast<std::uint32_t>(to_u64(v, "--min-hsize"));
        }
        if (const auto v = get_arg(args, "--max-hsize"); !v.empty()) {
            gen.max_hyperedge_size = static_cast<std::uint32_t>(to_u64(v, "--max-hsize"));
        }
        if (const auto v = get_arg(args, "--seed"); !v.empty()) {
            gen.seed = to_u64(v, "--seed");
        }

        htc::IMConfig im_cfg;
        if (const auto v = get_arg(args, "--k"); !v.empty()) {
            im_cfg.k = static_cast<htc::Budget>(to_u64(v, "--k"));
        }
        if (const auto v = get_arg(args, "--rtw-samples"); !v.empty()) {
            im_cfg.rtw_samples = static_cast<htc::Count>(to_u64(v, "--rtw-samples"));
        }
        if (const auto v = get_arg(args, "--mc-simulations"); !v.empty()) {
            im_cfg.mc_simulations = static_cast<htc::Count>(to_u64(v, "--mc-simulations"));
        }
        im_cfg.seed = gen.seed;

        htc::Count bench_samples = 500;
        if (const auto v = get_arg(args, "--bench-samples"); !v.empty()) {
            bench_samples = static_cast<htc::Count>(to_u64(v, "--bench-samples"));
        }

        const bool skip_baselines = has_arg(args, "--skip-baselines");
        std::string output_path = get_arg(args, "--output");
        if (output_path.empty()) {
            output_path = build_default_output_path(gen, im_cfg, bench_samples, skip_baselines);
        }

        const htc::HybridHypergraph graph = htc::generate_synthetic_hybrid_hypergraph(gen);

        std::cout << "graph.num_nodes=" << graph.num_nodes() << "\n";
        std::cout << "graph.num_edges=" << graph.num_edges() << "\n";
        std::cout << "graph.num_hyperedges=" << graph.num_hyperedges() << "\n";

        htc::Random random(gen.seed + 1U);
        htc::RTWSampler sampler(graph);
        htc::RTWStats rtw_stats;

        const auto t0 = std::chrono::steady_clock::now();
        for (htc::Count i = 0; i < bench_samples; ++i) {
            const htc::RTWSample w = sampler.sample(random);
            rtw_stats.add_sample(w);
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double rtw_sampling_seconds = std::chrono::duration<double>(t1 - t0).count();

        std::cout << "rtw.sampling.samples=" << bench_samples << "\n";
        std::cout << "rtw.sampling.seconds=" << rtw_sampling_seconds << "\n";
        std::cout << "rtw.sampling.samples_per_sec="
                  << (bench_samples / std::max(1e-12, rtw_sampling_seconds)) << "\n";
        std::cout << "rtw.avg_formula_nodes=" << rtw_stats.avg_formula_nodes() << "\n";
        std::cout << "rtw.avg_deps=" << rtw_stats.avg_dependencies() << "\n";
        std::cout << "rtw.avg_gates=" << rtw_stats.avg_gates() << "\n";
        std::cout << "rtw.avg_gate_input_incidences="
                  << rtw_stats.avg_gate_input_incidences() << "\n";
        std::cout << "rtw.avg_witness_size=" << rtw_stats.avg_witness_size() << "\n";

        htc::RTWGreedy rtw_greedy(graph);
        const htc::IMResult rtw_result = rtw_greedy.run(im_cfg);
        std::cout << "rtw_greedy.seconds=" << rtw_result.seconds << "\n";
        std::cout << "rtw_greedy.empirical_influence=" << rtw_result.empirical_influence << "\n";
        std::cout << "rtw_greedy.seed_count=" << rtw_result.seeds.size() << "\n";
        std::cout << "rtw_greedy.dirty_candidates=" << rtw_result.rtw_dirty_candidates << "\n";
        std::cout << "rtw_greedy.avg_dirty_candidates_per_seed="
                  << rtw_result.rtw_avg_dirty_candidates_per_seed << "\n";
        std::cout << "rtw_greedy.max_dirty_candidates_per_seed="
                  << rtw_result.rtw_max_dirty_candidates_per_seed << "\n";

        bool ur_present = false;
        htc::IMResult ur_result;
        bool hybrid_degree_present = false;
        htc::IMResult hd_result;
        if (!skip_baselines) {
            htc::URIMBaseline ur(graph);
            ur_result = ur.run(im_cfg);
            ur_present = true;
            std::cout << "ur_im.seconds=" << ur_result.seconds << "\n";
            std::cout << "ur_im.empirical_influence=" << ur_result.empirical_influence << "\n";

            htc::HybridDegreeBaseline hd(graph);
            hd_result = hd.run(im_cfg);
            hybrid_degree_present = true;
            std::cout << "hybrid_degree.seconds=" << hd_result.seconds << "\n";
            std::cout << "hybrid_degree.empirical_influence=" << hd_result.empirical_influence << "\n";
        }

        append_bench_result_jsonl(output_path,
                                  gen,
                                  im_cfg,
                                  bench_samples,
                                  skip_baselines,
                                  graph,
                                  rtw_stats,
                                  rtw_sampling_seconds,
                                  rtw_result,
                                  ur_present,
                                  ur_result,
                                  hybrid_degree_present,
                                  hd_result);
        const std::string readable_output_path = derive_readable_output_path(output_path);
        append_bench_result_readable(readable_output_path,
                                     gen,
                                     im_cfg,
                                     bench_samples,
                                     skip_baselines,
                                     graph,
                                     rtw_stats,
                                     rtw_sampling_seconds,
                                     rtw_result,
                                     ur_present,
                                     ur_result,
                                     hybrid_degree_present,
                                     hd_result);
        std::cout << "output_jsonl=" << output_path << "\n";
        std::cout << "output_readable=" << readable_output_path << "\n";

        return EXIT_SUCCESS;
    } catch (const std::exception& ex) {
        std::cerr << "bench_m5 failed: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }
}
