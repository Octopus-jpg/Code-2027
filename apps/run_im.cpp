#include <cmath>
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

#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

#include "htc/baselines.hpp"
#include "htc/im.hpp"
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

htc::Count to_count(const std::string& s, const char* name) {
    try {
        return static_cast<htc::Count>(std::stoull(s));
    } catch (...) {
        throw std::invalid_argument(std::string("invalid value for ") + name + ": " + s);
    }
}

htc::Budget to_budget(const std::string& s, const char* name) {
    try {
        const auto x = std::stoull(s);
        if (x > static_cast<unsigned long long>(std::numeric_limits<htc::Budget>::max())) {
            throw std::invalid_argument("overflow");
        }
        return static_cast<htc::Budget>(x);
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

htc::Count current_peak_rss_kb() {
#if defined(__unix__) || defined(__APPLE__)
    struct rusage usage {};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return 0U;
    }
    if (usage.ru_maxrss <= 0) {
        return 0U;
    }
#if defined(__APPLE__)
    return static_cast<htc::Count>(usage.ru_maxrss / 1024);
#else
    return static_cast<htc::Count>(usage.ru_maxrss);
#endif
#else
    return 0U;
#endif
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

std::string bool_token(bool value) { return value ? "1" : "0"; }

std::string build_default_output_path(const std::string& data_dir,
                                      const std::string& method,
                                      const htc::IMConfig& config) {
    std::filesystem::path out = std::filesystem::current_path();
    out /= "result";
    out /= "im";
    out /= "method_" + sanitize_token(method);
    out /= "data_" + dataset_token(data_dir);
    out /= "k_" + std::to_string(config.k);
    out /= "rtw_" + std::to_string(config.rtw_samples);
    out /= "theta0_" + std::to_string(config.rtw_training_samples);
    out /= "mc_eval_" + std::to_string(config.mc_simulations);
    out /= "mc_greedy_" + std::to_string(config.mc_greedy_simulations);
    out /= "eps_" + format_double_token(config.epsilon);
    out /= "delta_" + format_double_token(config.delta);
    out /= "lambda_" + format_double_token(config.lambda);
    out /= "seed_" + std::to_string(config.seed);
    out /= "progress_" + bool_token(config.use_progress_tiebreak);
    out /= "finalcmp_" + bool_token(config.use_final_compare);
    out /= "trivialub_" + bool_token(config.use_trivial_upper_bound);
    out /= "minb_" + bool_token(config.use_min_b_ablation);
    out /= "results.jsonl";
    return out.string();
}

void print_usage() {
    std::cout << "Usage:\n"
              << "  htc_im --data <dir> --method <rtw-greedy|rtw-greedy-p|mc-greedy|ur-im|ur-im-hist|edge-only|edge-only-hist|htc-ce|htc-ce-hist|hci1-tm|hci2-tm|hn-moea-htc|hybrid-degree> [options]\n"
              << "Options:\n"
              << "  --k <int>\n"
              << "  --rtw-samples <int>\n"
              << "  --theta0 <int> (training RTW sample size for rtw-greedy-p)\n"
              << "  --mc-simulations <int>\n"
              << "  --mc-greedy-simulations <int>\n"
              << "  --moea-population-size <int>\n"
              << "  --moea-offspring-size <int>\n"
              << "  --moea-generations <int>\n"
              << "  --moea-fitness-simulations <int>\n"
              << "  --moea-fitness-simulations-cap <int>\n"
              << "  --moea-tournament-size <int>\n"
              << "  --moea-mutation-rate <double>\n"
              << "  --moea-crossover-rate <double>\n"
              << "  --epsilon <double>\n"
              << "  --delta <double>\n"
              << "  --lambda <double|auto>\n"
              << "  --seed <int>\n"
              << "  --edges-file <path> (default: <data>/edges.txt)\n"
              << "  --hyperedges-file <path> (default: <data>/hyperedges.txt)\n"
              << "  --hyperedge-thresholds-file <path> (default: <data>/hyperedge_thresholds.txt)\n"
              << "  --no-progress-tiebreak\n"
              << "  --use-final-compare\n"
              << "  --no-final-compare\n"
              << "  --use-trivial-upper-bound\n"
              << "  --use-minb-ablation\n"
              << "  --output <path> (default: auto path under ./result)\n"
              << "  (human-readable metrics will also be appended to *.readable.txt)\n";
}

void print_result(std::ostream& out, const std::string& method, const htc::IMResult& result) {
    out << "method=" << method << "\n";
    out << "seeds=";
    for (std::size_t i = 0; i < result.seeds.size(); ++i) {
        out << result.seeds[i];
        if (i + 1 < result.seeds.size()) {
            out << ",";
        }
    }
    out << "\n";
    out << "rtw_satisfied_samples=" << result.rtw_satisfied_samples << "\n";
    out << "rtw_total_samples=" << result.rtw_total_samples << "\n";
    out << "rtw_hit_rate=" << result.rtw_hit_rate << "\n";
    out << "rtw_sample_estimate=" << result.rtw_sample_estimate << "\n";
    out << "rtw_estimate_over_B=" << result.rtw_estimate_over_B << "\n";
    out << "empirical_over_B=" << result.empirical_over_B << "\n";
    out << "rtw_total_formula_nodes=" << result.rtw_total_formula_nodes << "\n";
    out << "rtw_total_dependencies=" << result.rtw_total_dependencies << "\n";
    out << "rtw_total_gates=" << result.rtw_total_gates << "\n";
    out << "rtw_total_gate_input_incidences=" << result.rtw_total_gate_input_incidences << "\n";
    out << "rtw_total_witness_size=" << result.rtw_total_witness_size << "\n";
    out << "rtw_avg_formula_nodes=" << result.rtw_avg_formula_nodes << "\n";
    out << "rtw_avg_dependencies=" << result.rtw_avg_dependencies << "\n";
    out << "rtw_avg_gates=" << result.rtw_avg_gates << "\n";
    out << "rtw_avg_gate_input_incidences=" << result.rtw_avg_gate_input_incidences << "\n";
    out << "rtw_avg_witness_size=" << result.rtw_avg_witness_size << "\n";
    out << "rtw_occurrence_index_entries=" << result.rtw_occurrence_index_entries << "\n";
    out << "rtw_gain_cache_initial_recomputations="
              << result.rtw_gain_cache_initial_recomputations << "\n";
    out << "rtw_gain_cache_update_recomputations="
              << result.rtw_gain_cache_update_recomputations << "\n";
    out << "rtw_seed_additions=" << result.rtw_seed_additions << "\n";
    out << "rtw_affected_samples=" << result.rtw_affected_samples << "\n";
    out << "rtw_affected_gates=" << result.rtw_affected_gates << "\n";
    out << "rtw_activated_nodes=" << result.rtw_activated_nodes << "\n";
    out << "rtw_dirty_candidates=" << result.rtw_dirty_candidates << "\n";
    out << "rtw_max_dirty_candidates_per_seed="
              << result.rtw_max_dirty_candidates_per_seed << "\n";
    out << "rtw_satisfied_sample_dirty_expansions="
              << result.rtw_satisfied_sample_dirty_expansions << "\n";
    out << "rtw_fallback_sample_dirty_expansions="
              << result.rtw_fallback_sample_dirty_expansions << "\n";
    out << "rtw_avg_dirty_candidates_per_seed="
              << result.rtw_avg_dirty_candidates_per_seed << "\n";
    out << "rtw_avg_affected_samples_per_seed="
              << result.rtw_avg_affected_samples_per_seed << "\n";
    out << "rtw_gain_cache_initialization_seconds="
              << result.rtw_gain_cache_initialization_seconds << "\n";
    out << "rtw_gain_cache_update_seconds=" << result.rtw_gain_cache_update_seconds << "\n";
    out << "rtw_validation_samples_s_prime=" << result.rtw_validation_samples_s_prime << "\n";
    out << "rtw_validation_samples_s_plus=" << result.rtw_validation_samples_s_plus << "\n";
    out << "rr_samples=" << result.rr_samples << "\n";
    out << "eval_samples=" << result.eval_samples << "\n";
    out << "upper_bound_B=" << result.upper_bound_B << "\n";
    out << "certificate_alpha=" << result.certificate_alpha << "\n";
    out << "empirical_influence=" << result.empirical_influence << "\n";
    out << "time_opimc_seconds=" << result.time_opimc_seconds << "\n";
    out << "time_sampling_seconds=" << result.time_sampling_seconds << "\n";
    out << "time_greedy_seconds=" << result.time_greedy_seconds << "\n";
    out << "time_compare_seconds=" << result.time_compare_seconds << "\n";
    out << "time_total_seconds=" << result.time_total_seconds << "\n";
    out << "peak_rss_kb=" << result.peak_rss_kb << "\n";
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

void print_config(std::ostream& out, const htc::IMConfig& config) {
    out << "k=" << config.k << "\n";
    out << "rtw_samples_config=" << config.rtw_samples << "\n";
    out << "rtw_training_samples_config=" << config.rtw_training_samples << "\n";
    out << "mc_simulations_config=" << config.mc_simulations << "\n";
    out << "mc_greedy_simulations_config=" << config.mc_greedy_simulations << "\n";
    out << "moea_population_size_config=" << config.moea_population_size << "\n";
    out << "moea_offspring_size_config=" << config.moea_offspring_size << "\n";
    out << "moea_generations_config=" << config.moea_generations << "\n";
    out << "moea_fitness_simulations_config=" << config.moea_fitness_simulations << "\n";
    out << "moea_fitness_simulations_cap_config=" << config.moea_fitness_simulations_cap << "\n";
    out << "moea_tournament_size_config=" << config.moea_tournament_size << "\n";
    out << "moea_mutation_rate_config=" << config.moea_mutation_rate << "\n";
    out << "moea_crossover_rate_config=" << config.moea_crossover_rate << "\n";
    out << "epsilon=" << config.epsilon << "\n";
    out << "delta=" << config.delta << "\n";
    out << "lambda=" << config.lambda << "\n";
    out << "random_seed=" << config.seed << "\n";
    out << "use_progress_tiebreak=" << (config.use_progress_tiebreak ? "true" : "false")
        << "\n";
    out << "use_final_compare=" << (config.use_final_compare ? "true" : "false") << "\n";
    out << "use_trivial_upper_bound="
        << (config.use_trivial_upper_bound ? "true" : "false") << "\n";
    out << "use_min_b_ablation=" << (config.use_min_b_ablation ? "true" : "false") << "\n";
}

void append_readable_result(const std::string& readable_output_path,
                            const std::string& method,
                            const htc::IMConfig& config,
                            const htc::IMResult& result) {
    const std::filesystem::path path(readable_output_path);
    const auto parent = path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
    std::ofstream out(path, std::ios::app);
    if (!out.is_open()) {
        throw std::runtime_error("cannot open readable output file: " + path.string());
    }
    out << "task=im\n";
    print_config(out, config);
    print_result(out, method, result);
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

        const std::string method = get_arg(args, "--method", "rtw-greedy");
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

        htc::IMConfig config;
        if (const std::string v = get_arg(args, "--k"); !v.empty()) {
            config.k = to_budget(v, "--k");
        }
        if (const std::string v = get_arg(args, "--rtw-samples"); !v.empty()) {
            config.rtw_samples = to_count(v, "--rtw-samples");
        }
        if (const std::string v = get_arg(args, "--theta0"); !v.empty()) {
            config.rtw_training_samples = to_count(v, "--theta0");
        }
        if (const std::string v = get_arg(args, "--mc-simulations"); !v.empty()) {
            config.mc_simulations = to_count(v, "--mc-simulations");
        }
        if (const std::string v = get_arg(args, "--mc-greedy-simulations"); !v.empty()) {
            config.mc_greedy_simulations = to_count(v, "--mc-greedy-simulations");
        }
        if (const std::string v = get_arg(args, "--moea-population-size"); !v.empty()) {
            config.moea_population_size = to_count(v, "--moea-population-size");
        }
        if (const std::string v = get_arg(args, "--moea-offspring-size"); !v.empty()) {
            config.moea_offspring_size = to_count(v, "--moea-offspring-size");
        }
        if (const std::string v = get_arg(args, "--moea-generations"); !v.empty()) {
            config.moea_generations = to_count(v, "--moea-generations");
        }
        if (const std::string v = get_arg(args, "--moea-fitness-simulations"); !v.empty()) {
            config.moea_fitness_simulations = to_count(v, "--moea-fitness-simulations");
        }
        if (const std::string v = get_arg(args, "--moea-fitness-simulations-cap"); !v.empty()) {
            config.moea_fitness_simulations_cap = to_count(v, "--moea-fitness-simulations-cap");
        }
        if (const std::string v = get_arg(args, "--moea-tournament-size"); !v.empty()) {
            config.moea_tournament_size = to_count(v, "--moea-tournament-size");
        }
        if (const std::string v = get_arg(args, "--moea-mutation-rate"); !v.empty()) {
            config.moea_mutation_rate = to_double(v, "--moea-mutation-rate");
        }
        if (const std::string v = get_arg(args, "--moea-crossover-rate"); !v.empty()) {
            config.moea_crossover_rate = to_double(v, "--moea-crossover-rate");
        }
        if (const std::string v = get_arg(args, "--epsilon"); !v.empty()) {
            config.epsilon = to_double(v, "--epsilon");
        }
        if (const std::string v = get_arg(args, "--delta"); !v.empty()) {
            config.delta = to_double(v, "--delta");
        }
        if (const std::string v = get_arg(args, "--lambda"); !v.empty()) {
            if (v == "auto") {
                if (config.k == 0U) {
                    throw std::invalid_argument("--lambda auto requires --k > 0");
                }
                const double k_cuberoot = std::cbrt(static_cast<double>(config.k));
                config.lambda = k_cuberoot / (1.0 + k_cuberoot);
                if (!(config.lambda > 0.0) || !(config.lambda < 1.0) ||
                    !std::isfinite(config.lambda)) {
                    throw std::invalid_argument("computed --lambda auto is invalid");
                }
            } else {
                config.lambda = to_double(v, "--lambda");
            }
        }
        if (const std::string v = get_arg(args, "--seed"); !v.empty()) {
            config.seed = to_u64(v, "--seed");
        }
        config.use_progress_tiebreak = !has_arg(args, "--no-progress-tiebreak");
        const bool force_final_compare = has_arg(args, "--use-final-compare");
        const bool disable_final_compare = has_arg(args, "--no-final-compare");
        if (force_final_compare && disable_final_compare) {
            throw std::invalid_argument("cannot pass both --use-final-compare and --no-final-compare");
        }
        config.use_final_compare = !disable_final_compare;
        if (force_final_compare) {
            config.use_final_compare = true;
        }
        config.use_trivial_upper_bound = has_arg(args, "--use-trivial-upper-bound");
        config.use_min_b_ablation = has_arg(args, "--use-minb-ablation");
        if (config.use_trivial_upper_bound && config.use_min_b_ablation) {
            throw std::invalid_argument(
                "cannot pass both --use-trivial-upper-bound and --use-minb-ablation");
        }
        if (output_path.empty()) {
            output_path = build_default_output_path(data_dir, method, config);
        }

        const htc::HybridHypergraph graph = htc::load_hybrid_hypergraph_from_files(
            edge_file_path, hyperedge_file_path, hyperedge_threshold_file_path);

        htc::IMResult result;
        if (method == "rtw-greedy") {
            htc::RTWGreedy algo(graph);
            result = algo.run(config);
        } else if (method == "rtw-greedy-p") {
            htc::RTWGreedy algo(graph);
            result = algo.run_practical(config);
        } else if (method == "mc-greedy") {
            htc::MCGreedyBaseline algo(graph);
            result = algo.run(config);
        } else if (method == "ur-im") {
            htc::URIMBaseline algo(graph);
            result = algo.run(config);
        } else if (method == "ur-im-hist") {
            htc::URIMHistBaseline algo(graph);
            result = algo.run(config);
        } else if (method == "edge-only") {
            htc::EdgeOnlyBaseline algo(graph);
            result = algo.run(config);
        } else if (method == "edge-only-hist") {
            htc::EdgeOnlyHistBaseline algo(graph);
            result = algo.run(config);
        } else if (method == "htc-ce") {
            htc::CliqueExpansionBaseline algo(graph);
            result = algo.run(config);
        } else if (method == "htc-ce-hist") {
            htc::CliqueExpansionHistBaseline algo(graph);
            result = algo.run(config);
        } else if (method == "hci1-tm") {
            htc::HCI1TMBaseline algo(graph);
            result = algo.run(config);
        } else if (method == "hci2-tm") {
            htc::HCI2TMBaseline algo(graph);
            result = algo.run(config);
        } else if (method == "hn-moea-htc") {
            htc::HNMOEAHTCBaseline algo(graph);
            result = algo.run(config);
        } else if (method == "hybrid-degree") {
            htc::HybridDegreeBaseline algo(graph);
            result = algo.run(config);
        } else {
            throw std::invalid_argument("unknown --method: " + method);
        }
        if (result.time_total_seconds <= 0.0) {
            result.time_total_seconds = result.seconds;
        }
        if (result.peak_rss_kb == 0U) {
            result.peak_rss_kb = current_peak_rss_kb();
        }
        if (result.seconds <= 0.0) {
            result.seconds = result.time_total_seconds;
        }

        print_result(std::cout, method, result);
        htc::append_im_result_jsonl(output_path, data_dir, method, config, result);
        const std::string readable_output_path = derive_readable_output_path(output_path);
        append_readable_result(readable_output_path, method, config, result);
        std::cout << "output_jsonl=" << output_path << "\n";
        std::cout << "output_readable=" << readable_output_path << "\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& ex) {
        std::cerr << "htc_im failed: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }
}
