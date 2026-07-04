#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "htc/estimator.hpp"
#include "htc/io.hpp"
#include "htc/runtime_clock.hpp"
#include "htc/types.hpp"

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

std::string read_text_file(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("cannot open seeds file: " + path);
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

htc::Count to_count(const std::string& s, const char* name) {
    try {
        return static_cast<htc::Count>(std::stoull(s));
    } catch (...) {
        throw std::invalid_argument(std::string("invalid value for ") + name + ": " + s);
    }
}

double to_double(const std::string& s, const char* name) {
    try {
        return std::stod(s);
    } catch (...) {
        throw std::invalid_argument(std::string("invalid value for ") + name + ": " + s);
    }
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

std::string dataset_token(const std::string& data_dir) {
    const std::filesystem::path path(data_dir);
    std::string token = path.filename().string();
    if (token.empty()) {
        token = path.lexically_normal().string();
    }
    return sanitize_token(token);
}

std::string seedset_token(const std::vector<htc::NodeId>& seeds) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (htc::NodeId node : seeds) {
        std::uint64_t x = static_cast<std::uint64_t>(node);
        for (int i = 0; i < 4; ++i) {
            const unsigned char byte = static_cast<unsigned char>((x >> (8 * i)) & 0xFFU);
            hash ^= static_cast<std::uint64_t>(byte);
            hash *= 1099511628211ULL;
        }
    }
    std::ostringstream oss;
    oss << "n" << seeds.size() << "_h" << std::hex << hash;
    return oss.str();
}

std::string build_default_output_path(const std::string& data_dir,
                                      const std::string& method,
                                      const std::vector<htc::NodeId>& seeds,
                                      const htc::IEConfig& config) {
    std::filesystem::path out = std::filesystem::current_path();
    out /= "result";
    out /= "ie";
    out /= "method_" + sanitize_token(method);
    out /= "data_" + dataset_token(data_dir);
    out /= "seedset_" + seedset_token(seeds);
    out /= "eps_" + format_double_token(config.epsilon);
    out /= "delta_" + format_double_token(config.delta);
    out /= "fixed_" + std::to_string(config.fixed_samples);
    out /= "mc_" + std::to_string(config.mc_simulations);
    out /= "rr_" + std::to_string(config.rr_samples);
    out /= "seed_" + std::to_string(config.seed);
    out /= "results.jsonl";
    return out.string();
}

void print_usage() {
    std::cout << "Usage:\n"
              << "  htc_ie --data <dir> (--seeds <csv> | --seeds-file <path>) --method <rtw-est|rtw-fixed|mc-htc|ur-ie> [options]\n"
              << "Options:\n"
              << "  --epsilon <double>\n"
              << "  --delta <double>\n"
              << "  --fixed-samples <int> (rtw-fixed: 0 auto-computes from epsilon/delta)\n"
              << "  --mc-simulations <int> (ignored by mc-htc; mc-htc auto-computes from epsilon/delta)\n"
              << "  --seed <int>\n"
              << "  --seeds-file <path>\n"
              << "  --edges-file <path> (default: <data>/edges.txt)\n"
              << "  --hyperedges-file <path> (default: <data>/hyperedges.txt)\n"
              << "  --hyperedge-thresholds-file <path> (default: <data>/hyperedge_thresholds.txt)\n"
              << "  --output <path> (default: auto path under ./result)\n"
              << "  (human-readable metrics will also be appended to *.readable.txt)\n";
}

void print_result(std::ostream& out, const htc::IEResult& result) {
    out << "method=" << result.method << "\n";
    out << "estimate=" << result.estimate << "\n";
    out << "samples=" << result.samples << "\n";
    out << "satisfied=" << result.satisfied << "\n";
    out << "seconds=" << result.seconds << "\n";
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

void append_readable_result(const std::string& readable_output_path, const htc::IEResult& result) {
    const std::filesystem::path path(readable_output_path);
    const auto parent = path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
    std::ofstream out(path, std::ios::app);
    if (!out.is_open()) {
        throw std::runtime_error("cannot open readable output file: " + path.string());
    }
    out << "task=ie\n";
    print_result(out, result);
    out << "---\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        htc::reset_runtime_clock();
        std::vector<std::string> args;
        args.reserve(static_cast<std::size_t>(argc));
        for (int i = 1; i < argc; ++i) {
            args.emplace_back(argv[i]);
        }

        if (has_arg(args, "--help") || has_arg(args, "-h")) {
            print_usage();
            return EXIT_SUCCESS;
        }

        const std::string data_dir = get_arg(args, "--data");
        if (data_dir.empty()) {
            throw std::invalid_argument("--data is required");
        }

        const std::string seeds_csv = get_arg(args, "--seeds");
        const std::string seeds_file_path = get_arg(args, "--seeds-file");
        if (seeds_csv.empty() && seeds_file_path.empty()) {
            throw std::invalid_argument("either --seeds or --seeds-file is required");
        }
        if (!seeds_csv.empty() && !seeds_file_path.empty()) {
            throw std::invalid_argument("cannot pass both --seeds and --seeds-file");
        }

        const std::string method = get_arg(args, "--method", "rtw-est");
        std::string output_path = get_arg(args, "--output");
        const std::filesystem::path data_path(data_dir);
        std::string edge_file_path = get_arg(args, "--edges-file");
        std::string hyperedge_file_path = get_arg(args, "--hyperedges-file");
        std::string hyperedge_threshold_file_path = get_arg(args, "--hyperedge-thresholds-file");
        if (edge_file_path.empty()) {
            edge_file_path = (data_path / "edges.txt").string();
        }
        if (hyperedge_file_path.empty()) {
            hyperedge_file_path = (data_path / "hyperedges.txt").string();
        }
        if (hyperedge_threshold_file_path.empty()) {
            hyperedge_threshold_file_path =
                (data_path / "hyperedge_thresholds.txt").string();
        }

        htc::IEConfig config;
        if (const std::string v = get_arg(args, "--epsilon"); !v.empty()) {
            config.epsilon = to_double(v, "--epsilon");
        }
        if (const std::string v = get_arg(args, "--delta"); !v.empty()) {
            config.delta = to_double(v, "--delta");
        }
        if (const std::string v = get_arg(args, "--fixed-samples"); !v.empty()) {
            config.fixed_samples = to_count(v, "--fixed-samples");
        }
        if (const std::string v = get_arg(args, "--mc-simulations"); !v.empty()) {
            config.mc_simulations = to_count(v, "--mc-simulations");
        }
        if (const std::string v = get_arg(args, "--seed"); !v.empty()) {
            config.seed = to_u64(v, "--seed");
        }
        if (has_arg(args, "--rr-samples")) {
            throw std::invalid_argument(
                "--rr-samples is no longer supported; ur-ie now uses epsilon/delta adaptive stopping");
        }

        const htc::HybridHypergraph graph = htc::load_hybrid_hypergraph_from_files(
            edge_file_path, hyperedge_file_path, hyperedge_threshold_file_path);
        std::vector<htc::NodeId> seeds;
        if (!seeds_file_path.empty()) {
            seeds = htc::parse_seed_list(read_text_file(seeds_file_path));
        } else {
            seeds = htc::parse_seed_list(seeds_csv);
        }
        if (seeds.empty()) {
            throw std::invalid_argument("seed set cannot be empty");
        }
        for (htc::NodeId v : seeds) {
            if (v >= graph.num_nodes()) {
                throw std::out_of_range("seed id out of graph range");
            }
        }
        htc::IEResult result;
        if (method == "rtw-est") {
            htc::RTWEstimator est(graph);
            result = est.estimate_adaptive(seeds, config);
        } else if (method == "rtw-fixed") {
            htc::RTWEstimator est(graph);
            if (config.fixed_samples == 0U) {
                config.fixed_samples = htc::compute_forward_mc_samples(
                    graph.num_nodes(), static_cast<htc::Count>(seeds.size()), config.epsilon,
                    config.delta);
                std::cout << "[ie:rtw-fixed] sample-sizing"
                          << ": mode=auto"
                          << ", n=" << graph.num_nodes()
                          << ", |S|=" << seeds.size()
                          << ", epsilon=" << config.epsilon
                          << ", delta=" << config.delta
                          << ", theta=" << config.fixed_samples
                          << ", total_elapsed_s=" << htc::runtime_elapsed_seconds()
                          << "\n";
            } else {
                std::cout << "[ie:rtw-fixed] sample-sizing"
                          << ": mode=manual"
                          << ", theta=" << config.fixed_samples
                          << ", total_elapsed_s=" << htc::runtime_elapsed_seconds()
                          << "\n";
            }
            const htc::Count theta = config.fixed_samples;
            result = est.estimate_fixed(seeds, theta, config.seed);
        } else if (method == "mc-htc") {
            config.mc_simulations = htc::compute_forward_mc_samples(
                graph.num_nodes(), static_cast<htc::Count>(seeds.size()), config.epsilon, config.delta);
            std::cout << "[ie:mc-htc] sample-sizing"
                      << ": n=" << graph.num_nodes()
                      << ", |S|=" << seeds.size()
                      << ", epsilon=" << config.epsilon
                      << ", delta=" << config.delta
                      << ", simulations=" << config.mc_simulations
                      << ", total_elapsed_s=" << htc::runtime_elapsed_seconds()
                      << "\n";
            htc::MCEstimator est(graph);
            result = est.estimate(seeds, config.mc_simulations, config.seed);
        } else if (method == "ur-ie") {
            htc::URIEstimator est(graph);
            result = est.estimate_adaptive(seeds, config);
            config.rr_samples = result.samples;
        } else {
            throw std::invalid_argument("unknown --method: " + method);
        }
        if (output_path.empty()) {
            output_path = build_default_output_path(data_dir, method, seeds, config);
        }

        print_result(std::cout, result);
        htc::append_ie_result_jsonl(output_path, data_dir, seeds, config, result);
        const std::string readable_output_path = derive_readable_output_path(output_path);
        append_readable_result(readable_output_path, result);
        std::cout << "output_jsonl=" << output_path << "\n";
        std::cout << "output_readable=" << readable_output_path << "\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& ex) {
        std::cerr << "htc_ie failed: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }
}
