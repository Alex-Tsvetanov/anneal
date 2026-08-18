// The measurement harness.
//
// One repetition of a metaheuristic is an anecdote, so nothing in this project
// runs a configuration once. An Experiment is a configuration plus a
// repetition count plus a seed, and it returns every individual trial rather
// than a summary, so that the summary can be recomputed and the raw values can
// be printed in an appendix.
#pragma once

#include <string>
#include <vector>

#include "anneal/core.hpp"
#include "anneal/parallel.hpp"
#include "anneal/stats.hpp"

namespace anneal {

struct Trial {
    double cost = kInfinity;          // internal, minimised units
    double display = 0.0;             // the problem's natural units
    double gap = 0.0;                 // relative to the known value, NaN if none
    double wall_ms = 0.0;
    std::uint64_t evaluations = 0;
    std::uint64_t sync_wait_ns = 0;
    std::uint64_t cas_failures = 0;
    std::uint64_t adopted = 0;
    bool reached_target = false;
    double time_to_target_ms = -1.0;
};

struct Experiment {
    const Problem* problem = nullptr;
    std::string algorithm = "annealing";
    std::string scheme = "multistart";
    int threads = 1;
    int repetitions = 7;
    Budget budget;
    // Each repetition uses base_seed + repetition index, so a repetition can
    // be replayed on its own without replaying the ones before it.
    std::uint64_t base_seed = 1;
};

std::vector<Trial> run_experiment(const Experiment& experiment);

// Field extractors, so a caller can summarise any column without writing the
// same loop again.
std::vector<double> column_gap(const std::vector<Trial>& trials);
std::vector<double> column_display(const std::vector<Trial>& trials);
std::vector<double> column_wall_ms(const std::vector<Trial>& trials);
std::vector<double> column_time_to_target(const std::vector<Trial>& trials);

// Fraction of repetitions that reached the target. A time to target median
// computed over runs that mostly missed the target is meaningless, so this is
// reported next to it every time.
double hit_rate(const std::vector<Trial>& trials);

}  // namespace anneal
