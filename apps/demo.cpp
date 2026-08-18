// One command, one picture of the whole system: four metaheuristics on one
// instance under one time budget, then the best of them across every thread
// count the machine has.
//
//   cmake --build build --target demo
//   ./build/anneal_demo
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include "anneal/algorithms.hpp"
#include "anneal/experiment.hpp"
#include "anneal/problems.hpp"
#include "anneal/stats.hpp"

using namespace anneal;

namespace {

void rule(std::size_t width = 82) { std::printf("%s\n", std::string(width, '-').c_str()); }

void heading(const std::string& text) {
    std::printf("\n");
    rule();
    std::printf("%s\n", text.c_str());
    rule();
}

void row(const std::vector<std::string>& cells, const std::vector<int>& widths) {
    std::string line;
    for (std::size_t i = 0; i < cells.size(); ++i) {
        const std::size_t w = static_cast<std::size_t>(widths[i] < 0 ? -widths[i] : widths[i]);
        line += widths[i] < 0 ? pad_right(cells[i], w) : pad_left(cells[i], w);
        line += "  ";
    }
    std::printf("%s\n", line.c_str());
}

std::string compiler_string() {
#if defined(__clang__)
    return "clang++ " + std::to_string(__clang_major__) + "." + std::to_string(__clang_minor__);
#elif defined(__GNUC__)
    return "g++ " + std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__) + "." +
           std::to_string(__GNUC_PATCHLEVEL__);
#elif defined(_MSC_VER)
    return "msvc " + std::to_string(_MSC_VER);
#else
    return "unknown compiler";
#endif
}

// A relative gap is meaningless when the reference is zero, which is exactly
// the case for graph colouring, where the optimum is zero conflicts. Rather
// than print forty thousand per cent, the distance is reported in the
// problem's own units whenever the reference is smaller than one.
std::string gap_string(double gap, double reference) {
    if (std::isnan(gap)) return "not known";
    if (std::fabs(reference) < 1.0) return format_number(gap, 0) + " absolute";
    return format_number(gap * 100.0, 2) + " %";
}

}  // namespace

int main() {
    const unsigned hardware = std::max(1u, std::thread::hardware_concurrency());

    heading("Anneal: parallel metaheuristics, measured");
    std::printf("compiler        : %s\n", compiler_string().c_str());
    std::printf("logical CPUs    : %u\n", hardware);
    std::printf("clock           : std::chrono::steady_clock\n");

    // The search instance for parts 1 to 3: uniformly random Euclidean points.
    // Its optimum is not known and the program does not pretend otherwise, so
    // quality is reported as a tour length. The instances with known values
    // are in part 4, where the gap is a real gap.
    TspProblem tsp = make_uniform_tsp(1000, 20260819);
    std::printf("instance        : %s, %zu cities, uniform in a square\n", tsp.name().c_str(),
                tsp.size());
    std::printf("optimum         : not known for this instance, so quality is a tour length\n");

    const std::vector<std::string> algorithms{"annealing", "tabu", "genetic", "aco"};
    const int repetitions = 7;
    const std::chrono::milliseconds phase1_budget{250};

    // ------------------------------------------------------------------
    // Part 1. Four metaheuristics, same instance, same wall clock budget,
    // one thread each. Seven seeds, reported as best, median and worst.
    // ------------------------------------------------------------------
    heading("Part 1. Four metaheuristics, 1 thread, " +
            std::to_string(phase1_budget.count()) + " ms budget, " +
            std::to_string(repetitions) + " seeds");
    const std::vector<int> widths1{-11, 12, 12, 12, 12, 14};
    row({"algorithm", "best", "median", "worst", "IQR", "median evals"}, widths1);
    rule();

    std::string champion;
    double champion_median = kInfinity;
    for (const std::string& algorithm : algorithms) {
        Experiment e;
        e.problem = &tsp;
        e.algorithm = algorithm;
        e.scheme = "multistart";
        e.threads = 1;
        e.repetitions = repetitions;
        e.budget.wall = phase1_budget;
        e.base_seed = 1000;
        const std::vector<Trial> trials = run_experiment(e);
        const Summary quality = summarise(column_display(trials));
        std::vector<double> evaluations;
        for (const Trial& t : trials) evaluations.push_back(static_cast<double>(t.evaluations));
        row({algorithm, format_number(quality.min, 1), format_number(quality.median, 1),
             format_number(quality.max, 1), format_number(quality.iqr(), 1),
             format_number(summarise(evaluations).median, 0)},
            widths1);
        if (quality.median < champion_median) {
            champion_median = quality.median;
            champion = algorithm;
        }
    }
    rule();
    std::printf("Lower is better. Best, median and worst are over the %d seeds, and the\n",
                repetitions);
    std::printf("interquartile range is the spread a single run would have hidden.\n");
    std::printf("Best median quality: %s.\n", champion.c_str());

    // ------------------------------------------------------------------
    // Part 2. Speedup, measured as time to a fixed target quality. That is
    // the only comparison across thread counts that keeps the work being
    // compared the same: at a fixed time budget more threads simply do more
    // work, which is not a speedup, it is a larger machine.
    // ------------------------------------------------------------------
    heading("Part 2. Speedup and efficiency of " + champion + ", time to a fixed target");

    Experiment pilot;
    pilot.problem = &tsp;
    pilot.algorithm = champion;
    pilot.scheme = "multistart";
    pilot.threads = 1;
    pilot.repetitions = 5;
    pilot.budget.wall = std::chrono::milliseconds{400};
    pilot.base_seed = 77;
    const Summary pilot_quality = summarise(column_display(run_experiment(pilot)));
    // The target is the worst quality the pilot reached, so a single thread
    // reaches it in every repetition and the baseline of the speedup is a
    // measurement rather than a timeout.
    const double target = pilot_quality.max;
    std::printf("pilot, 1 thread, 400 ms, 5 seeds : best %s, worst %s\n",
                format_number(pilot_quality.min, 1).c_str(),
                format_number(pilot_quality.max, 1).c_str());
    std::printf("target tour length               : %s\n", format_number(target, 1).c_str());
    std::printf("cap per repetition               : 3000 ms, 7 seeds per thread count\n\n");

    std::vector<int> thread_counts;
    for (int t : {1, 2, 3, 4, 6, 8, 10, 12}) {
        if (t <= static_cast<int>(hardware)) thread_counts.push_back(t);
    }
    if (thread_counts.empty()) thread_counts.push_back(1);

    const std::vector<int> widths2{-8, 15, 12, 10, 12, 8};
    row({"threads", "median t2t ms", "IQR ms", "speedup", "efficiency", "hits"}, widths2);
    rule();

    double baseline = 0.0;
    for (int threads : thread_counts) {
        Experiment e;
        e.problem = &tsp;
        e.algorithm = champion;
        e.scheme = "multistart";
        e.threads = threads;
        e.repetitions = 7;
        e.budget.wall = std::chrono::milliseconds{3000};
        e.budget.target = target;
        e.base_seed = 5000;
        const std::vector<Trial> trials = run_experiment(e);
        const Summary t2t = summarise(column_time_to_target(trials));
        if (threads == thread_counts.front()) baseline = t2t.median;
        const double s = speedup(baseline, t2t.median);
        row({std::to_string(threads), format_number(t2t.median, 1), format_number(t2t.iqr(), 1),
             format_number(s, 2), format_number(efficiency(s, threads), 2),
             format_number(hit_rate(trials) * 100.0, 0) + "%"},
            widths2);
    }
    rule();
    std::printf("Speedup is the median time of one thread divided by the median time of p\n");
    std::printf("threads. Efficiency is that speedup divided by p. Both come from medians of\n");
    std::printf("seven runs, and hits is the share of runs that reached the target at all.\n");

    // ------------------------------------------------------------------
    // Part 3. What communication buys. Same algorithm, same budget, same
    // thread count, three different exchange policies.
    // ------------------------------------------------------------------
    const int scheme_threads = static_cast<int>(std::min(8u, hardware));
    heading("Part 3. Three parallel schemes, " + champion + ", " +
            std::to_string(scheme_threads) + " threads, 300 ms budget");
    const std::vector<int> widths3{-13, 12, 12, 12, 14, 12, 10};
    row({"scheme", "best", "median", "worst", "sync ms total", "CAS retries", "adopted"},
        widths3);
    rule();
    for (const std::string scheme : {"multistart", "cooperative", "island"}) {
        Experiment e;
        e.problem = &tsp;
        e.algorithm = champion;
        e.scheme = scheme;
        e.threads = scheme_threads;
        e.repetitions = 7;
        e.budget.wall = std::chrono::milliseconds{300};
        e.base_seed = 31337;
        const std::vector<Trial> trials = run_experiment(e);
        const Summary quality = summarise(column_display(trials));
        std::vector<double> sync;
        std::vector<double> cas;
        std::vector<double> adopted;
        for (const Trial& t : trials) {
            sync.push_back(static_cast<double>(t.sync_wait_ns) / 1e6);
            cas.push_back(static_cast<double>(t.cas_failures));
            adopted.push_back(static_cast<double>(t.adopted));
        }
        row({scheme, format_number(quality.min, 1), format_number(quality.median, 1),
             format_number(quality.max, 1), format_number(summarise(sync).median, 3),
             format_number(summarise(cas).median, 0),
             format_number(summarise(adopted).median, 0)},
            widths3);
    }
    rule();
    std::printf("The sync column is the time all workers together spent inside the shared\n");
    std::printf("best or a mailbox, against 300 ms of budget per worker.\n");

    // ------------------------------------------------------------------
    // Part 4. The instances whose optimum is known, so the gap is a real
    // gap. One generator per problem, nothing downloaded.
    // ------------------------------------------------------------------
    heading("Part 4. Instances with a known optimum, " + std::to_string(scheme_threads) +
            " threads, 300 ms budget, cooperative");
    TspProblem grid = make_grid_tsp(24, 10.0);
    KnapsackProblem knapsack = make_random_knapsack(400, 20260819);
    ColoringProblem coloring = make_planted_coloring(150, 8, 0.15, 20260819);
    const Problem* problems[] = {&grid, &knapsack, &coloring};
    const std::vector<int> widths4{-18, -18, 14, 14, 14};
    row({"instance", "unit", "known value", "median found", "median gap"}, widths4);
    rule();
    for (const Problem* p : problems) {
        Experiment e;
        e.problem = p;
        e.algorithm = champion;
        e.scheme = "cooperative";
        e.threads = scheme_threads;
        e.repetitions = 7;
        e.budget.wall = std::chrono::milliseconds{300};
        e.base_seed = 4242;
        const std::vector<Trial> trials = run_experiment(e);
        const Summary quality = summarise(column_display(trials));
        const Summary gap = summarise(column_gap(trials));
        row({p->name(), p->unit(), format_number(p->display(p->best_known()), 2),
             format_number(quality.median, 2), gap_string(gap.median, p->best_known())},
            widths4);
    }
    rule();
    std::printf("The grid optimum is exact: no tour edge can be shorter than the spacing, and\n");
    std::printf("a tour attaining that bound exists for an even side. The knapsack optimum is\n");
    std::printf("exact by dynamic programming over the generated instance. The colouring\n");
    std::printf("optimum is zero conflicts, reachable because the instance is generated from\n");
    std::printf("a planted proper colouring, so its gap is reported as an absolute count.\n\n");
    return 0;
}
