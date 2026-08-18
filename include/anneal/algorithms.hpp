// The four metaheuristics. Each one owns its own state and speaks only to the
// Problem contract, so adding a fifth problem needs no change here and adding
// a fifth algorithm needs no change to the problems.
#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include "anneal/core.hpp"

namespace anneal {

// ---------------------------------------------------------------------------
// Simulated annealing.
// ---------------------------------------------------------------------------
enum class Cooling { Geometric, Linear, Logarithmic };

struct AnnealingConfig {
    Cooling schedule = Cooling::Geometric;
    double alpha = 0.995;        // geometric factor, per cooling step
    double t_end = 1e-4;         // temperature floor, then the search reheats
    int chain_length = 50;       // moves at one temperature
    double target_acceptance = 0.8;  // used to calibrate the initial temperature
    bool reheat = true;
};

// The Metropolis rule, pulled out so it can be tested without running a
// search. u is a draw from [0, 1).
inline bool metropolis_accepts(double delta, double temperature, double u) {
    if (delta <= 0.0) return true;
    if (temperature <= 0.0) return false;
    return u < std::exp(-delta / temperature);
}

// Temperature after `steps` cooling steps, given the schedule.
double cooled(const AnnealingConfig& cfg, double t0, std::uint64_t steps);

MetaheuristicPtr make_annealing(const Problem& problem, std::uint64_t seed,
                                AnnealingConfig cfg = {});

// ---------------------------------------------------------------------------
// Tabu search.
// ---------------------------------------------------------------------------

// Direct mapped table of move attributes. A real tabu list would be a queue
// plus a set; this is a fixed array indexed by the low bits of the attribute,
// which makes lookup and insertion branch-free and allocation-free. The price
// is that two attributes sharing a slot make each other tabu, which costs a
// little diversification and never costs correctness, because the aspiration
// criterion can always override a tabu verdict.
class TabuTable {
public:
    explicit TabuTable(std::size_t slots = 4096) : expiry_(slots, 0), key_(slots, 0) {}

    void forbid(std::uint64_t key, std::uint64_t iteration, std::uint64_t tenure) {
        const std::size_t slot = index(key);
        key_[slot] = key;
        expiry_[slot] = iteration + tenure;
    }

    bool is_tabu(std::uint64_t key, std::uint64_t iteration) const {
        const std::size_t slot = index(key);
        return key_[slot] == key && expiry_[slot] > iteration;
    }

private:
    std::size_t index(std::uint64_t key) const {
        // Mix before masking: the low bits of a move attribute are the second
        // index, so masking the raw value would put every move on a vertex
        // into a handful of slots.
        key ^= key >> 33;
        key *= 0xFF51AFD7ED558CCDULL;
        key ^= key >> 29;
        return static_cast<std::size_t>(key) & (expiry_.size() - 1);
    }
    std::vector<std::uint64_t> expiry_;
    std::vector<std::uint64_t> key_;
};

// A tabu move is taken anyway when it would beat the best solution seen so
// far. Separated out so the rule itself is testable.
inline bool tabu_admits(bool is_tabu, double candidate_cost, double best_cost) {
    return !is_tabu || candidate_cost < best_cost;
}

struct TabuConfig {
    int candidate_list = 40;   // moves sampled per iteration
    std::uint64_t tenure = 25;
    std::size_t slots = 4096;
};

MetaheuristicPtr make_tabu(const Problem& problem, std::uint64_t seed, TabuConfig cfg = {});

// ---------------------------------------------------------------------------
// Genetic algorithm.
// ---------------------------------------------------------------------------
struct GeneticConfig {
    int population = 40;
    int tournament = 3;
    double crossover_rate = 0.9;
    double mutation_rate = 0.3;
    int mutation_moves = 2;
    int elite = 2;
};

MetaheuristicPtr make_genetic(const Problem& problem, std::uint64_t seed,
                              GeneticConfig cfg = {});

// ---------------------------------------------------------------------------
// Ant colony optimisation.
// ---------------------------------------------------------------------------
struct AcoConfig {
    int ants = 16;
    double alpha = 1.0;       // pheromone exponent
    double beta = 3.0;        // heuristic exponent
    double rho = 0.1;         // evaporation rate per iteration
    double q = 1.0;           // deposit scale
    double tau0 = 1.0;        // initial trail
    double tau_min = 1e-6;    // floor, so a trail can always recover
    int local_search_moves = 0;  // descent steps applied to each ant, 0 to disable
};

// One evaporation step, exposed for testing: tau <- max(tau_min, (1-rho) tau).
void evaporate(std::vector<double>& tau, double rho, double tau_min);

MetaheuristicPtr make_aco(const Problem& problem, std::uint64_t seed, AcoConfig cfg = {});

// ---------------------------------------------------------------------------
// Uniform construction by name, used by the demo and the benchmark harness so
// the algorithm can be a command line string.
// ---------------------------------------------------------------------------
MetaheuristicPtr make_algorithm(const std::string& name, const Problem& problem,
                                std::uint64_t seed);

}  // namespace anneal
