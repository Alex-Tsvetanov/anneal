#include "anneal/experiment.hpp"

#include <cmath>
#include <stdexcept>

#include "anneal/algorithms.hpp"

namespace anneal {

std::vector<Trial> run_experiment(const Experiment& experiment) {
    if (experiment.problem == nullptr) throw std::runtime_error("experiment without a problem");
    const Problem& problem = *experiment.problem;
    const std::string algorithm = experiment.algorithm;
    const Factory factory = [&problem, algorithm](std::uint64_t seed, int) {
        return make_algorithm(algorithm, problem, seed);
    };

    std::vector<Trial> trials;
    trials.reserve(static_cast<std::size_t>(experiment.repetitions));
    for (int r = 0; r < experiment.repetitions; ++r) {
        const RunResult run = run_scheme(experiment.scheme, factory, experiment.threads,
                                         experiment.budget,
                                         experiment.base_seed + static_cast<std::uint64_t>(r));
        Trial t;
        t.cost = run.cost;
        t.display = problem.display(run.cost);
        const double reference = problem.best_known();
        t.gap = std::isnan(reference) ? std::numeric_limits<double>::quiet_NaN()
                                      : relative_gap(run.cost, reference);
        t.wall_ms = run.wall_ms;
        t.evaluations = run.evaluations;
        t.sync_wait_ns = run.total_sync_wait_ns();
        t.cas_failures = run.total_cas_failures();
        for (const WorkerMetrics& m : run.workers) t.adopted += m.adopted;
        t.reached_target = run.reached_target;
        t.time_to_target_ms = run.time_to_target_ms;
        trials.push_back(t);
    }
    return trials;
}

std::vector<double> column_gap(const std::vector<Trial>& trials) {
    std::vector<double> out;
    out.reserve(trials.size());
    for (const Trial& t : trials) out.push_back(t.gap);
    return out;
}

std::vector<double> column_display(const std::vector<Trial>& trials) {
    std::vector<double> out;
    out.reserve(trials.size());
    for (const Trial& t : trials) out.push_back(t.display);
    return out;
}

std::vector<double> column_wall_ms(const std::vector<Trial>& trials) {
    std::vector<double> out;
    out.reserve(trials.size());
    for (const Trial& t : trials) out.push_back(t.wall_ms);
    return out;
}

std::vector<double> column_time_to_target(const std::vector<Trial>& trials) {
    // Only the repetitions that actually reached the target contribute. A run
    // that timed out has no time to target, and substituting the timeout would
    // quietly turn a failure into a measurement.
    std::vector<double> out;
    for (const Trial& t : trials) {
        if (t.reached_target) out.push_back(t.time_to_target_ms);
    }
    return out;
}

double hit_rate(const std::vector<Trial>& trials) {
    if (trials.empty()) return 0.0;
    std::size_t hits = 0;
    for (const Trial& t : trials) {
        if (t.reached_target) ++hits;
    }
    return static_cast<double>(hits) / static_cast<double>(trials.size());
}

}  // namespace anneal
