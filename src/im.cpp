#include "htc/im.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

#include "htc/cascade.hpp"
#include "htc/estimator.hpp"
#include "htc/ic.hpp"
#include "htc/runtime_clock.hpp"

namespace htc {
namespace {

using SteadyClock = std::chrono::steady_clock;
constexpr Count kFinalMCProgressPercentStep = 1U;

double elapsed_seconds(SteadyClock::time_point begin, SteadyClock::time_point end) {
    return std::chrono::duration<double>(end - begin).count();
}

Count current_peak_rss_kb() {
#if defined(__unix__) || defined(__APPLE__)
    struct rusage usage {};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return 0U;
    }
    if (usage.ru_maxrss <= 0) {
        return 0U;
    }
#if defined(__APPLE__)
    return static_cast<Count>(usage.ru_maxrss / 1024);
#else
    return static_cast<Count>(usage.ru_maxrss);
#endif
#else
    return 0U;
#endif
}

double logcnk(Count n, Count k) {
    if (k > n) {
        throw std::invalid_argument("logcnk requires k <= n");
    }
    k = std::min(k, n - k);
    long double result = 0.0L;
    for (Count i = 1U; i <= k; ++i) {
        const long double numer = static_cast<long double>(n - k + i);
        const long double denom = static_cast<long double>(i);
        result += std::log(numer / denom);
    }
    return static_cast<double>(result);
}

Count ceil_to_count(long double value) {
    if (!(value > 0.0L)) {
        return 1U;
    }
    const long double ceil_value = std::ceil(value);
    const long double count_max = static_cast<long double>(std::numeric_limits<Count>::max());
    if (!std::isfinite(static_cast<double>(ceil_value)) || ceil_value >= count_max) {
        return std::numeric_limits<Count>::max();
    }
    return static_cast<Count>(ceil_value);
}

Count compute_auto_theta(NodeId n,
                         Budget k,
                         double effective_epsilon,
                         double effective_delta,
                         double upper_bound) {
    if (n == 0U || k == 0U) {
        throw std::invalid_argument("compute_auto_theta requires n > 0 and k > 0");
    }
    if (effective_epsilon <= 0.0 || effective_epsilon >= 1.0) {
        throw std::invalid_argument("effective epsilon must be in (0,1)");
    }
    if (effective_delta <= 0.0 || effective_delta >= 1.0) {
        throw std::invalid_argument("effective delta must be in (0,1)");
    }
    if (!(upper_bound > 0.0) || !std::isfinite(upper_bound)) {
        throw std::invalid_argument("RTW-Greedy upper bound B must be finite and > 0");
    }

    const long double n_ld = static_cast<long double>(n);
    const long double eps_ld = static_cast<long double>(effective_epsilon);
    const long double delta_ld = static_cast<long double>(effective_delta);
    const long double b_ld = static_cast<long double>(upper_bound);

    const long double numerator = 2.0L * n_ld * n_ld;
    const long double denominator = eps_ld * eps_ld * b_ld * b_ld;
    const long double log_term =
        static_cast<long double>(logcnk(n, k)) + std::log(2.0L / delta_ld);
    const long double theta = (numerator / denominator) * log_term;
    return ceil_to_count(theta);
}

std::vector<Count> build_sampling_progress_checkpoints(Count total_samples,
                                                       Count percent_step) {
    std::vector<Count> checkpoints;
    if (total_samples == 0U || percent_step == 0U) {
        return checkpoints;
    }

    const Count step = std::min<Count>(percent_step, 100U);
    for (Count pct = step; pct <= 100U; pct += step) {
        const long double raw = (static_cast<long double>(total_samples) *
                                 static_cast<long double>(pct)) /
                                100.0L;
        Count checkpoint = ceil_to_count(raw);
        if (checkpoint == 0U) {
            checkpoint = 1U;
        }
        if (checkpoint > total_samples) {
            checkpoint = total_samples;
        }
        if (checkpoints.empty() || checkpoints.back() != checkpoint) {
            checkpoints.push_back(checkpoint);
        }
    }
    if (checkpoints.empty() || checkpoints.back() != total_samples) {
        checkpoints.push_back(total_samples);
    }
    return checkpoints;
}

std::vector<NodeId> run_rtw_greedy_selection(const IMConfig& config,
                                             Budget steps,
                                             RTWCollection* collection) {
    if (collection == nullptr) {
        throw std::invalid_argument("collection cannot be null");
    }

    std::vector<NodeId> seeds;
    seeds.reserve(steps);
    const Budget log_interval = config.rtw_greedy_progress_interval;

    for (Budget step = 0; step < steps; ++step) {
        const NodeId best_node = collection->best_candidate(config.use_progress_tiebreak);
        if (best_node == kInvalidNodeId) {
            break;
        }
        collection->add_seed(best_node);
        seeds.push_back(best_node);
        if (log_interval > 0U && (seeds.size() % log_interval == 0U)) {
            const Count satisfied = collection->current_satisfied_count();
        std::cout << "[rtw-greedy] greedy progress: selected="
                  << seeds.size() << "/" << steps
                  << ", latest_seed=" << best_node
                  << ", satisfied_samples=" << satisfied
                  << ", total_elapsed_s=" << runtime_elapsed_seconds() << std::endl;
        }
    }
    if (log_interval > 0U && !seeds.empty() && (seeds.size() % log_interval != 0U)) {
        const Count satisfied = collection->current_satisfied_count();
        std::cout << "[rtw-greedy] greedy progress: selected="
                  << seeds.size() << "/" << steps
                  << ", latest_seed=" << seeds.back()
                  << ", satisfied_samples=" << satisfied
                  << ", total_elapsed_s=" << runtime_elapsed_seconds() << std::endl;
    }

    return seeds;
}

double estimate_htc_influence(const HybridHypergraph& graph,
                              const std::vector<NodeId>& seeds,
                              Count simulations,
                              std::uint64_t seed,
                              const char* method_tag) {
    if (simulations == 0U) {
        throw std::invalid_argument("IMConfig.mc_simulations must be > 0");
    }

    const char* tag = (method_tag == nullptr) ? "rtw-greedy" : method_tag;
    const auto t0 = SteadyClock::now();
    const std::vector<Count> checkpoints =
        build_sampling_progress_checkpoints(simulations, kFinalMCProgressPercentStep);
    std::size_t next_checkpoint = 0U;
    std::cout << "[" << tag << "] final MC started: simulations=" << simulations
              << ", checkpoint_step=" << kFinalMCProgressPercentStep
              << "%"
              << ", total_elapsed_s=" << runtime_elapsed_seconds()
              << std::endl;

    Random random(seed);
    double total = 0.0;
    for (Count i = 0U; i < simulations; ++i) {
        total += static_cast<double>(
            HTCCascadeSimulator::simulate_spread(graph, seeds, &random, "RTWGreedy final MC"));
        const Count completed = i + 1U;
        while (next_checkpoint < checkpoints.size() &&
               completed >= checkpoints[next_checkpoint]) {
            const Count checkpoint = checkpoints[next_checkpoint];
            const double elapsed = elapsed_seconds(t0, SteadyClock::now());
            const double rate =
                (elapsed > 1e-12) ? (static_cast<double>(checkpoint) / elapsed) : 0.0;
            const double eta = (rate > 1e-12)
                                   ? (static_cast<double>(simulations - checkpoint) / rate)
                                   : 0.0;
            const double estimate =
                static_cast<double>(checkpoint > 0U ? total : 0.0) /
                static_cast<double>(checkpoint);
            std::cout << "[" << tag << "] final MC progress: " << checkpoint
                      << "/" << simulations
                      << ", estimate=" << estimate
                      << ", elapsed_s=" << elapsed
                      << ", eta_s=" << eta
                      << ", rate=" << rate << " sims/s"
                      << ", total_elapsed_s=" << runtime_elapsed_seconds()
                      << std::endl;
            ++next_checkpoint;
        }
    }
    std::cout << "[" << tag << "] final MC done: simulations=" << simulations
              << ", elapsed_s=" << elapsed_seconds(t0, SteadyClock::now())
              << ", total_elapsed_s=" << runtime_elapsed_seconds()
              << std::endl;
    return total / static_cast<double>(simulations);
}

}  // namespace

RTWGreedy::RTWGreedy(const HybridHypergraph& graph) : graph_(graph) {}

IMResult RTWGreedy::run(const IMConfig& config) const {
    const NodeId n = graph_.num_nodes();
    if (n == 0U) {
        throw std::invalid_argument("cannot run RTWGreedy on an empty graph");
    }
    if (config.k == 0U) {
        throw std::invalid_argument("IMConfig.k must be > 0");
    }
    if (config.epsilon <= 0.0 || config.epsilon >= 1.0) {
        throw std::invalid_argument("IMConfig epsilon must be in (0,1)");
    }
    if (config.delta <= 0.0 || config.delta >= 1.0) {
        throw std::invalid_argument("IMConfig delta must be in (0,1)");
    }
    if (config.use_min_b_ablation && config.use_trivial_upper_bound) {
        throw std::invalid_argument(
            "RTW-Greedy cannot combine --use-minb-ablation and --use-trivial-upper-bound");
    }
    if (!config.use_trivial_upper_bound && (config.lambda <= 0.0 || config.lambda >= 1.0)) {
        throw std::invalid_argument("IMConfig lambda must be in (0,1)");
    }

    const Budget steps = (config.k > n) ? n : config.k;
    if (steps == 0U) {
        throw std::invalid_argument("effective IM budget is zero");
    }

    const auto t_total_begin = SteadyClock::now();

    OPIMProxyResult proxy_result;
    double upper_bound_b = static_cast<double>(steps);
    double time_opimc_seconds = 0.0;
    double theta_effective_epsilon = config.lambda * config.epsilon;
    double theta_effective_delta = config.delta / 2.0;
    if (config.use_trivial_upper_bound) {
        upper_bound_b = static_cast<double>(n);
        theta_effective_epsilon = config.epsilon;
        theta_effective_delta = config.delta;
        std::cout << "[rtw-greedy] opim skipped: use_trivial_upper_bound=true"
                  << ", upper_bound_B=n=" << upper_bound_b
                  << ", theta_epsilon=" << theta_effective_epsilon
                  << ", theta_delta=" << theta_effective_delta
                  << ", total_elapsed_s=" << runtime_elapsed_seconds() << std::endl;
    } else if (!config.use_min_b_ablation) {
        const auto t_opim_begin = SteadyClock::now();
        const ICGraph upper_graph = build_upper_relaxed_aux_ic_graph(graph_);
        const double opim_epsilon = (1.0 - config.lambda) * config.epsilon;
        const double opim_delta = config.delta / 2.0;
        std::cout << "[rtw-greedy] opim started: model=upper-relaxed-aux"
                  << ", k=" << steps
                  << ", epsilon=" << opim_epsilon
                  << ", delta=" << opim_delta
                  << ", total_elapsed_s=" << runtime_elapsed_seconds() << std::endl;

        OPIMProxy proxy;
        proxy_result = proxy.run(upper_graph,
                                 steps,
                                 opim_epsilon,
                                 opim_delta,
                                 config.seed,
                                 graph_.num_nodes());
        upper_bound_b = proxy_result.upper_bound;
        const auto t_opim_end = SteadyClock::now();
        time_opimc_seconds = elapsed_seconds(t_opim_begin, t_opim_end);
        std::cout << "[rtw-greedy] opim done: rr_samples=" << proxy_result.rr_samples
                  << ", upper_bound_B=" << upper_bound_b
                  << ", seconds=" << time_opimc_seconds
                  << ", total_elapsed_s=" << runtime_elapsed_seconds() << std::endl;
    } else {
        std::cout << "[rtw-greedy] opim skipped: use_minb_ablation=true"
                  << ", upper_bound_B=" << upper_bound_b
                  << ", total_elapsed_s=" << runtime_elapsed_seconds() << std::endl;
    }

    const auto t_sampling_begin = SteadyClock::now();
    Count theta = config.rtw_samples;
    if (theta == 0U) {
        theta = compute_auto_theta(
            n, steps, theta_effective_epsilon, theta_effective_delta, upper_bound_b);
    }
    if (theta == 0U) {
        throw std::logic_error("theta unexpectedly becomes zero");
    }

    Random random(config.seed + 1U);
    RTWSampler sampler(graph_);
    RTWCollection collection;
    RTWStats rtw_stats;
    const std::vector<Count> sampling_checkpoints =
        build_sampling_progress_checkpoints(theta, config.rtw_sampling_progress_percent);
    std::size_t next_sampling_checkpoint = 0U;
    if (!sampling_checkpoints.empty()) {
        std::cout << "[rtw-greedy] sampling started: theta=" << theta
                  << ", upper_bound_B=" << upper_bound_b
                  << ", checkpoint_step=" << config.rtw_sampling_progress_percent
                  << "%"
                  << ", total_elapsed_s=" << runtime_elapsed_seconds() << std::endl;
    }
    for (Count i = 0U; i < theta; ++i) {
        RTWSample sample = sampler.sample(random);
        rtw_stats.add_sample(sample);
        collection.add_sample(std::move(sample));
        const Count completed = i + 1U;
        while (next_sampling_checkpoint < sampling_checkpoints.size() &&
               completed >= sampling_checkpoints[next_sampling_checkpoint]) {
            const Count checkpoint = sampling_checkpoints[next_sampling_checkpoint];
            const Count pct = static_cast<Count>(
                (static_cast<long double>(checkpoint) * 100.0L) /
                static_cast<long double>(theta));
            std::cout << "[rtw-greedy] sampling progress: " << checkpoint
                      << "/" << theta << " (" << pct << "%)"
                      << ", total_elapsed_s=" << runtime_elapsed_seconds() << std::endl;
            ++next_sampling_checkpoint;
        }
    }
    collection.build_global_occurrence_index(n);
    collection.initialize_incremental_state();
    collection.initialize_gain_cache();
    const auto t_sampling_end = SteadyClock::now();
    const double time_sampling_seconds = elapsed_seconds(t_sampling_begin, t_sampling_end);

    const auto t_greedy_begin = SteadyClock::now();
    const std::vector<NodeId> s_star = run_rtw_greedy_selection(config, steps, &collection);
    const auto t_greedy_end = SteadyClock::now();
    const double time_greedy_seconds = elapsed_seconds(t_greedy_begin, t_greedy_end);

    Count satisfied_star = collection.current_satisfied_count();
    const RTWCollectionStats& collection_stats = collection.stats();

    const auto t_compare_begin = SteadyClock::now();
    std::vector<NodeId> selected = s_star;
    Count selected_satisfied = satisfied_star;
    if (config.use_final_compare && !proxy_result.seeds.empty()) {
        const Count satisfied_plus = collection.evaluate_seed_set_with_index(proxy_result.seeds);
        if (satisfied_plus > selected_satisfied) {
            selected = proxy_result.seeds;
            selected_satisfied = satisfied_plus;
        }
    }
    const auto t_compare_end = SteadyClock::now();
    const double time_compare_seconds = elapsed_seconds(t_compare_begin, t_compare_end);
    const double rtw_hit_rate =
        static_cast<double>(selected_satisfied) / static_cast<double>(theta);
    const double rtw_sample_estimate = static_cast<double>(n) *
                                       rtw_hit_rate;

    IMResult result;
    result.seeds = std::move(selected);
    result.rtw_satisfied_samples = selected_satisfied;
    result.rtw_total_samples = theta;
    result.rtw_hit_rate = rtw_hit_rate;
    result.rtw_sample_estimate = rtw_sample_estimate;
    result.rtw_estimate_over_B =
        (upper_bound_b > 0.0) ? (rtw_sample_estimate / upper_bound_b) : 0.0;
    result.rtw_total_formula_nodes = rtw_stats.total_formula_nodes;
    result.rtw_total_dependencies = rtw_stats.total_dependencies;
    result.rtw_total_gates = rtw_stats.total_gates;
    result.rtw_total_gate_input_incidences = rtw_stats.total_gate_input_incidences;
    result.rtw_total_witness_size = rtw_stats.total_witness_size;
    result.rtw_avg_formula_nodes = rtw_stats.avg_formula_nodes();
    result.rtw_avg_dependencies = rtw_stats.avg_dependencies();
    result.rtw_avg_gates = rtw_stats.avg_gates();
    result.rtw_avg_gate_input_incidences = rtw_stats.avg_gate_input_incidences();
    result.rtw_avg_witness_size = rtw_stats.avg_witness_size();
    result.rtw_occurrence_index_entries = collection_stats.occurrence_index_entries;
    result.rtw_gain_cache_initial_recomputations = collection_stats.gain_cache_initial_recomputations;
    result.rtw_gain_cache_update_recomputations = collection_stats.gain_cache_update_recomputations;
    result.rtw_seed_additions = collection_stats.seed_additions;
    result.rtw_affected_samples = collection_stats.affected_samples;
    result.rtw_affected_gates = collection_stats.affected_gates;
    result.rtw_activated_nodes = collection_stats.activated_nodes;
    result.rtw_dirty_candidates = collection_stats.dirty_candidates;
    result.rtw_max_dirty_candidates_per_seed = collection_stats.max_dirty_candidates_per_seed;
    result.rtw_satisfied_sample_dirty_expansions = collection_stats.satisfied_sample_dirty_expansions;
    result.rtw_fallback_sample_dirty_expansions = collection_stats.fallback_sample_dirty_expansions;
    result.rtw_avg_dirty_candidates_per_seed = collection_stats.avg_dirty_candidates_per_seed();
    result.rtw_avg_affected_samples_per_seed = collection_stats.avg_affected_samples_per_seed();
    result.rtw_gain_cache_initialization_seconds =
        collection_stats.gain_cache_initialization_seconds;
    result.rtw_gain_cache_update_seconds = collection_stats.gain_cache_update_seconds;
    result.rr_samples = proxy_result.rr_samples;
    result.eval_samples = config.mc_simulations;
    result.upper_bound_B = upper_bound_b;
    result.time_opimc_seconds = time_opimc_seconds;
    result.time_sampling_seconds = time_sampling_seconds;
    result.time_greedy_seconds = time_greedy_seconds;
    result.time_compare_seconds = time_compare_seconds;
    if (!config.use_min_b_ablation) {
        const double certificate_epsilon =
            config.use_trivial_upper_bound ? config.epsilon : (config.lambda * config.epsilon);
        const double alpha = rtw_sample_estimate / upper_bound_b -
                             (certificate_epsilon * 0.5);
        result.certificate_alpha = std::max(0.0, alpha);
    } else {
        result.certificate_alpha = 0.0;
    }

    const auto t_total_end = SteadyClock::now();
    result.time_total_seconds = elapsed_seconds(t_total_begin, t_total_end);
    result.seconds = result.time_total_seconds;
    result.peak_rss_kb = current_peak_rss_kb();
    result.empirical_influence =
        estimate_htc_influence(graph_,
                               result.seeds,
                               config.mc_simulations,
                               config.seed + 1001U,
                               "rtw-greedy");
    result.empirical_over_B =
        (upper_bound_b > 0.0) ? (result.empirical_influence / upper_bound_b) : 0.0;
    return result;
}

IMResult RTWGreedy::run_practical(const IMConfig& config) const {
    const NodeId n = graph_.num_nodes();
    if (n == 0U) {
        throw std::invalid_argument("cannot run RTWGreedyP on an empty graph");
    }
    if (config.k == 0U) {
        throw std::invalid_argument("IMConfig.k must be > 0");
    }
    if (config.epsilon <= 0.0 || config.epsilon >= 1.0) {
        throw std::invalid_argument("IMConfig epsilon must be in (0,1)");
    }
    if (config.delta <= 0.0 || config.delta >= 1.0) {
        throw std::invalid_argument("IMConfig delta must be in (0,1)");
    }
    if (config.lambda <= 0.0 || config.lambda >= 1.0) {
        throw std::invalid_argument("IMConfig lambda must be in (0,1)");
    }
    if (config.use_min_b_ablation || config.use_trivial_upper_bound) {
        throw std::invalid_argument(
            "RTW-Greedy-P requires OPIM-C and does not support "
            "--use-minb-ablation/--use-trivial-upper-bound");
    }

    const Budget steps = (config.k > n) ? n : config.k;
    if (steps == 0U) {
        throw std::invalid_argument("effective IM budget is zero");
    }

    Count theta0 = config.rtw_training_samples;
    if (theta0 == 0U && config.rtw_samples > 0U) {
        theta0 = config.rtw_samples;
    }
    if (theta0 == 0U) {
        throw std::invalid_argument("RTW-Greedy-P requires --theta0 > 0");
    }

    const auto t_total_begin = SteadyClock::now();

    const auto t_opim_begin = SteadyClock::now();
    const ICGraph upper_graph = build_upper_relaxed_aux_ic_graph(graph_);
    const double opim_epsilon = (1.0 - config.lambda) * config.epsilon;
    const double opim_delta = config.delta / 2.0;
    std::cout << "[rtw-greedy-p] opim started: model=upper-relaxed-aux"
              << ", k=" << steps
              << ", epsilon=" << opim_epsilon
              << ", delta=" << opim_delta
              << ", total_elapsed_s=" << runtime_elapsed_seconds() << std::endl;

    OPIMProxy proxy;
    OPIMProxyResult proxy_result = proxy.run(upper_graph,
                                             steps,
                                             opim_epsilon,
                                             opim_delta,
                                             config.seed,
                                             graph_.num_nodes());
    const double upper_bound_b = proxy_result.upper_bound;
    const auto t_opim_end = SteadyClock::now();
    const double time_opimc_seconds = elapsed_seconds(t_opim_begin, t_opim_end);
    std::cout << "[rtw-greedy-p] opim done: rr_samples=" << proxy_result.rr_samples
              << ", upper_bound_B=" << upper_bound_b
              << ", seconds=" << time_opimc_seconds
              << ", total_elapsed_s=" << runtime_elapsed_seconds() << std::endl;

    const auto t_sampling_begin = SteadyClock::now();
    Random random(config.seed + 1U);
    RTWSampler sampler(graph_);
    RTWCollection collection;
    RTWStats rtw_stats;
    const std::vector<Count> sampling_checkpoints =
        build_sampling_progress_checkpoints(theta0, config.rtw_sampling_progress_percent);
    std::size_t next_sampling_checkpoint = 0U;
    if (!sampling_checkpoints.empty()) {
        std::cout << "[rtw-greedy-p] sampling started: theta0=" << theta0
                  << ", checkpoint_step=" << config.rtw_sampling_progress_percent
                  << "%"
                  << ", total_elapsed_s=" << runtime_elapsed_seconds() << std::endl;
    }
    for (Count i = 0U; i < theta0; ++i) {
        RTWSample sample = sampler.sample(random);
        rtw_stats.add_sample(sample);
        collection.add_sample(std::move(sample));
        const Count completed = i + 1U;
        while (next_sampling_checkpoint < sampling_checkpoints.size() &&
               completed >= sampling_checkpoints[next_sampling_checkpoint]) {
            const Count checkpoint = sampling_checkpoints[next_sampling_checkpoint];
            const Count pct = static_cast<Count>(
                (static_cast<long double>(checkpoint) * 100.0L) /
                static_cast<long double>(theta0));
            std::cout << "[rtw-greedy-p] sampling progress: " << checkpoint
                      << "/" << theta0 << " (" << pct << "%)"
                      << ", total_elapsed_s=" << runtime_elapsed_seconds() << std::endl;
            ++next_sampling_checkpoint;
        }
    }
    collection.build_global_occurrence_index(n);
    collection.initialize_incremental_state();
    collection.initialize_gain_cache();
    const auto t_sampling_end = SteadyClock::now();
    const double time_sampling_seconds = elapsed_seconds(t_sampling_begin, t_sampling_end);

    const auto t_greedy_begin = SteadyClock::now();
    const std::vector<NodeId> s_prime = run_rtw_greedy_selection(config, steps, &collection);
    const Count satisfied_prime_train = collection.current_satisfied_count();
    const auto t_greedy_end = SteadyClock::now();
    const double time_greedy_seconds = elapsed_seconds(t_greedy_begin, t_greedy_end);

    const RTWCollectionStats& collection_stats = collection.stats();

    const auto t_compare_begin = SteadyClock::now();
    const double eta = config.lambda * config.epsilon * 0.5;
    const double validation_delta = config.delta / 4.0;
    if (validation_delta <= 0.0 || validation_delta >= 1.0) {
        throw std::logic_error("RTW-Greedy-P validation delta is out of range");
    }
    std::cout << "[rtw-greedy-p] validation started: epsilon=" << eta
              << ", delta_each=" << validation_delta
              << ", total_elapsed_s=" << runtime_elapsed_seconds() << std::endl;

    IEConfig validation_config;
    validation_config.epsilon = eta;
    validation_config.delta = validation_delta;
    validation_config.seed = config.seed + 2001U;
    RTWEstimator validation_estimator(graph_);
    const IEResult val_prime = validation_estimator.estimate_adaptive(s_prime, validation_config);

    validation_config.seed = config.seed + 3001U;
    const IEResult val_plus =
        validation_estimator.estimate_adaptive(proxy_result.seeds, validation_config);

    std::vector<NodeId> selected = s_prime;
    Count selected_satisfied_train = satisfied_prime_train;
    double selected_validation_estimate = val_prime.estimate;
    if (val_plus.estimate > val_prime.estimate) {
        selected = proxy_result.seeds;
        selected_satisfied_train = collection.evaluate_seed_set_with_index(selected);
        selected_validation_estimate = val_plus.estimate;
    }
    const auto t_compare_end = SteadyClock::now();
    const double time_compare_seconds = elapsed_seconds(t_compare_begin, t_compare_end);
    const double rtw_hit_rate =
        static_cast<double>(selected_satisfied_train) / static_cast<double>(theta0);
    const double rtw_sample_estimate = static_cast<double>(n) * rtw_hit_rate;
    std::cout << "[rtw-greedy-p] validation done: est(S')=" << val_prime.estimate
              << ", samples(S')=" << val_prime.samples
              << ", est(S+)=" << val_plus.estimate
              << ", samples(S+)=" << val_plus.samples
              << ", selected_estimate=" << selected_validation_estimate
              << ", selected="
              << ((val_plus.estimate > val_prime.estimate) ? "S+" : "S'")
              << ", seconds=" << time_compare_seconds
              << ", total_elapsed_s=" << runtime_elapsed_seconds() << std::endl;

    IMResult result;
    result.seeds = std::move(selected);
    result.rtw_satisfied_samples = selected_satisfied_train;
    result.rtw_total_samples = theta0;
    result.rtw_hit_rate = rtw_hit_rate;
    result.rtw_sample_estimate = rtw_sample_estimate;
    result.rtw_estimate_over_B =
        (upper_bound_b > 0.0) ? (rtw_sample_estimate / upper_bound_b) : 0.0;
    result.rtw_total_formula_nodes = rtw_stats.total_formula_nodes;
    result.rtw_total_dependencies = rtw_stats.total_dependencies;
    result.rtw_total_gates = rtw_stats.total_gates;
    result.rtw_total_gate_input_incidences = rtw_stats.total_gate_input_incidences;
    result.rtw_total_witness_size = rtw_stats.total_witness_size;
    result.rtw_avg_formula_nodes = rtw_stats.avg_formula_nodes();
    result.rtw_avg_dependencies = rtw_stats.avg_dependencies();
    result.rtw_avg_gates = rtw_stats.avg_gates();
    result.rtw_avg_gate_input_incidences = rtw_stats.avg_gate_input_incidences();
    result.rtw_avg_witness_size = rtw_stats.avg_witness_size();
    result.rtw_occurrence_index_entries = collection_stats.occurrence_index_entries;
    result.rtw_gain_cache_initial_recomputations = collection_stats.gain_cache_initial_recomputations;
    result.rtw_gain_cache_update_recomputations = collection_stats.gain_cache_update_recomputations;
    result.rtw_seed_additions = collection_stats.seed_additions;
    result.rtw_affected_samples = collection_stats.affected_samples;
    result.rtw_affected_gates = collection_stats.affected_gates;
    result.rtw_activated_nodes = collection_stats.activated_nodes;
    result.rtw_dirty_candidates = collection_stats.dirty_candidates;
    result.rtw_max_dirty_candidates_per_seed = collection_stats.max_dirty_candidates_per_seed;
    result.rtw_satisfied_sample_dirty_expansions = collection_stats.satisfied_sample_dirty_expansions;
    result.rtw_fallback_sample_dirty_expansions = collection_stats.fallback_sample_dirty_expansions;
    result.rtw_avg_dirty_candidates_per_seed = collection_stats.avg_dirty_candidates_per_seed();
    result.rtw_avg_affected_samples_per_seed = collection_stats.avg_affected_samples_per_seed();
    result.rtw_gain_cache_initialization_seconds =
        collection_stats.gain_cache_initialization_seconds;
    result.rtw_gain_cache_update_seconds = collection_stats.gain_cache_update_seconds;
    result.rtw_validation_samples_s_prime = val_prime.samples;
    result.rtw_validation_samples_s_plus = val_plus.samples;
    result.rr_samples = proxy_result.rr_samples;
    result.eval_samples = config.mc_simulations;
    result.upper_bound_B = upper_bound_b;
    result.certificate_alpha = 0.0;
    result.time_opimc_seconds = time_opimc_seconds;
    result.time_sampling_seconds = time_sampling_seconds;
    result.time_greedy_seconds = time_greedy_seconds;
    result.time_compare_seconds = time_compare_seconds;

    const auto t_total_end = SteadyClock::now();
    result.time_total_seconds = elapsed_seconds(t_total_begin, t_total_end);
    result.seconds = result.time_total_seconds;
    result.peak_rss_kb = current_peak_rss_kb();
    result.empirical_influence =
        estimate_htc_influence(graph_,
                               result.seeds,
                               config.mc_simulations,
                               config.seed + 1001U,
                               "rtw-greedy-p");
    result.empirical_over_B =
        (upper_bound_b > 0.0) ? (result.empirical_influence / upper_bound_b) : 0.0;
    return result;
}

}  // namespace htc
