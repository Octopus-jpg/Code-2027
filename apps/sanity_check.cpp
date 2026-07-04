#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "htc/graph.hpp"
#include "htc/random.hpp"
#include "htc/rtw.hpp"

int main() {
    try {
        htc::HybridHypergraph graph(5);

        graph.add_edge(0, 1, 0.2);
        graph.add_edge(1, 2, 0.7);
        graph.add_edge(2, 3, 1.0);

        graph.add_hyperedge({0, 2, 4}, 2);
        graph.add_hyperedge({1, 3}, 1);

        graph.build_indices();
        graph.validate();

        if (graph.num_nodes() != 5U) {
            throw std::runtime_error("unexpected node count");
        }
        if (graph.num_edges() != 3U) {
            throw std::runtime_error("unexpected edge count");
        }
        if (graph.num_hyperedges() != 2U) {
            throw std::runtime_error("unexpected hyperedge count");
        }

        if (graph.out_edges(1).size() != 1U) {
            throw std::runtime_error("out-edge index check failed for node 1");
        }
        if (graph.in_edges(2).size() != 1U) {
            throw std::runtime_error("in-edge index check failed for node 2");
        }
        if (graph.incident_hyperedges(2).size() != 1U) {
            throw std::runtime_error("incident hyperedge index check failed for node 2");
        }

        htc::Random random(1);
        htc::RTWSampler sampler(graph);
        htc::RTWEvaluator evaluator;
        auto witness = sampler.sample_with_root(3U, random);

        if (evaluator.is_satisfied(witness, {}) != false) {
            throw std::runtime_error("RTW evaluator empty-seed check failed");
        }
        if (evaluator.is_satisfied(witness, {2U}) != true) {
            throw std::runtime_error("RTW evaluator seed-satisfaction check failed");
        }

        htc::RTWCollection collection;
        collection.add_sample(witness);
        collection.build_global_occurrence_index(graph.num_nodes());
        collection.initialize_incremental_state();

        if (collection.current_satisfied_count() != 0U) {
            throw std::runtime_error("RTW collection initial satisfied count should be 0");
        }
        if (collection.marginal_gain(2U) < 1U) {
            throw std::runtime_error("RTW collection marginal gain check failed");
        }

        collection.add_seed(2U);
        if (collection.current_satisfied_count() != 1U) {
            throw std::runtime_error("RTW collection satisfied count after add_seed check failed");
        }

        std::cout << "Sanity check passed.\n";
        std::cout << "nodes=" << graph.num_nodes()
                  << " edges=" << graph.num_edges()
                  << " hyperedges=" << graph.num_hyperedges() << "\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& ex) {
        std::cerr << "Sanity check failed: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }
}
