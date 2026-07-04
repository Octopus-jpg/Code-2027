#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace htc {

using NodeId = std::uint32_t;
using EdgeId = std::uint32_t;
using HyperedgeId = std::uint32_t;
using GateId = std::uint32_t;
using SampleId = std::uint64_t;
using Count = std::uint64_t;
using Budget = std::uint32_t;
using Prob = double;

inline constexpr NodeId kInvalidNodeId = std::numeric_limits<NodeId>::max();
inline constexpr EdgeId kInvalidEdgeId = std::numeric_limits<EdgeId>::max();
inline constexpr HyperedgeId kInvalidHyperedgeId = std::numeric_limits<HyperedgeId>::max();
inline constexpr GateId kInvalidGateId = std::numeric_limits<GateId>::max();
inline constexpr SampleId kInvalidSampleId = std::numeric_limits<SampleId>::max();

struct IEConfig {
    double epsilon = 0.1;
    double delta = 0.01;
    Count fixed_samples = 0;
    Count mc_simulations = 1000;
    Count rr_samples = 1000;
    std::uint64_t seed = 1;
};

struct IEResult {
    std::string method;
    double estimate = 0.0;
    Count samples = 0;
    Count satisfied = 0;
    double seconds = 0.0;
};

struct IMConfig {
    Budget k = 10;
    // 0 means auto-compute from (B, effective epsilon, effective delta).
    Count rtw_samples = 0;
    // Training RTW sample size theta_0 used by RTW-Greedy-P.
    // 0 means unset (RTW-Greedy-P requires theta_0 > 0).
    Count rtw_training_samples = 0;
    // Independent MC simulations for final influence evaluation.
    Count mc_simulations = 10000;
    // Internal MC simulations for candidate scoring in mc-greedy selection.
    Count mc_greedy_simulations = 500;
    double epsilon = 0.1;
    double delta = 0.01;
    // Error-budget allocation parameter in (0,1).
    double lambda = 0.5;
    std::uint64_t seed = 1;
    // RTW-Greedy runtime progress logging controls.
    // Sampling: print once every N% completion (0 disables).
    Count rtw_sampling_progress_percent = 1;
    // Greedy selection: print once every N selected seeds (0 disables).
    Budget rtw_greedy_progress_interval = 1;
    bool use_progress_tiebreak = true;
    bool use_final_compare = true;
    bool use_min_b_ablation = false;
    bool use_trivial_upper_bound = false;
    // HN-MOEA-HTC search hyper-parameters.
    Count moea_population_size = 100;
    Count moea_offspring_size = 100;
    Count moea_generations = 100;
    Count moea_fitness_simulations = 100;
    Count moea_fitness_simulations_cap = 10000;
    Count moea_tournament_size = 5;
    double moea_mutation_rate = 0.1;
    double moea_crossover_rate = 1.0;
};

struct IMResult {
    std::vector<NodeId> seeds;
    // RTW-specific counters.
    Count rtw_satisfied_samples = 0;
    Count rtw_total_samples = 0;
    double rtw_hit_rate = 0.0;
    double rtw_sample_estimate = 0.0;
    double rtw_estimate_over_B = 0.0;
    double empirical_over_B = 0.0;
    Count rtw_total_formula_nodes = 0;
    Count rtw_total_dependencies = 0;
    Count rtw_total_gates = 0;
    Count rtw_total_gate_input_incidences = 0;
    Count rtw_total_witness_size = 0;
    double rtw_avg_formula_nodes = 0.0;
    double rtw_avg_dependencies = 0.0;
    double rtw_avg_gates = 0.0;
    double rtw_avg_gate_input_incidences = 0.0;
    double rtw_avg_witness_size = 0.0;
    Count rtw_occurrence_index_entries = 0;
    Count rtw_gain_cache_initial_recomputations = 0;
    Count rtw_gain_cache_update_recomputations = 0;
    Count rtw_seed_additions = 0;
    Count rtw_affected_samples = 0;
    Count rtw_affected_gates = 0;
    Count rtw_activated_nodes = 0;
    Count rtw_dirty_candidates = 0;
    Count rtw_max_dirty_candidates_per_seed = 0;
    Count rtw_satisfied_sample_dirty_expansions = 0;
    Count rtw_fallback_sample_dirty_expansions = 0;
    double rtw_avg_dirty_candidates_per_seed = 0.0;
    double rtw_avg_affected_samples_per_seed = 0.0;
    double rtw_gain_cache_initialization_seconds = 0.0;
    double rtw_gain_cache_update_seconds = 0.0;
    // RTW-Greedy-P validation stage sample counts (0 for methods that do not run this stage).
    Count rtw_validation_samples_s_prime = 0;
    Count rtw_validation_samples_s_plus = 0;
    // OPIM-C RR sample count used by UR/EdgeOnly/HTC-CE style methods.
    Count rr_samples = 0;
    // Number of Monte Carlo simulations used for final influence evaluation.
    Count eval_samples = 0;
    // Upper bound B used in RTW-Greedy sample-size calibration.
    double upper_bound_B = 0.0;
    // Certificate value (0 if unavailable / not applicable).
    double certificate_alpha = 0.0;
    double empirical_influence = 0.0;
    double time_opimc_seconds = 0.0;
    double time_sampling_seconds = 0.0;
    double time_greedy_seconds = 0.0;
    double time_compare_seconds = 0.0;
    double time_total_seconds = 0.0;
    Count peak_rss_kb = 0;
    double seconds = 0.0;
};

}  // namespace htc
