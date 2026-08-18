// The measurement programme. Every number that appears in the report comes
// from one of these sub-commands, and each one prints the command that
// produced it so a table cell can be traced back to a run.
//
//   ./build/anneal_bench                 all sub-commands
//   ./build/anneal_bench objective       cost of one objective evaluation
//   ./build/anneal_bench neighbourhood   how the 2-opt move is chosen
//   ./build/anneal_bench falsesharing    padded against packed counters
//   ./build/anneal_bench profile         time attribution by ablation
//   ./build/anneal_bench quality         four algorithms on four instances
//   ./build/anneal_bench schemes         the three parallel schemes
//   ./build/anneal_bench speedup         thread scan, time to target
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include <cmath>

#include "anneal/algorithms.hpp"
#include "anneal/experiment.hpp"
#include "anneal/problems.hpp"
#include "anneal/stats.hpp"

using namespace anneal;

namespace {

using Clock = std::chrono::steady_clock;

void rule(std::size_t width = 88) { std::printf("%s\n", std::string(width, '-').c_str()); }

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

std::string gap_cell(double gap, double reference) {
    if (std::isnan(gap)) return "n/a";
    if (std::fabs(reference) < 1.0) return format_number(gap, 1) + " abs";
    return format_number(gap * 100.0, 3) + " %";
}

double sink = 0.0;  // written to, so the timed loops cannot be optimised away

// Times a callable, repeating until at least `min_ms` have passed, and reports
// nanoseconds per call. Repeating to a floor rather than a fixed count keeps a
// fast operation from being measured entirely as clock resolution.
template <typename F>
double ns_per_call(F&& body, double min_ms = 250.0) {
    std::uint64_t calls = 0;
    const Clock::time_point start = Clock::now();
    double elapsed_ms = 0.0;
    do {
        for (int i = 0; i < 64; ++i) body();
        calls += 64;
        elapsed_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    } while (elapsed_ms < min_ms);
    return elapsed_ms * 1e6 / static_cast<double>(calls);
}

// ---------------------------------------------------------------------------
void benchmark_objective() {
    heading("Objective evaluation: vectorisation, and where the distances come from");
    std::printf("Command: anneal_bench objective\n");
    std::printf("Nanoseconds per full evaluation, from a loop timed for at least 250 ms.\n\n");

    const std::vector<int> widths{-36, 14, 14, 10};
    row({"objective", "ns per call", "ns per item", "ratio"}, widths);
    rule();
    for (int n : {256, 1024, 4096}) {
        KnapsackProblem knapsack = make_random_knapsack(n, 12345);
        Rng rng(7);
        const Solution x = knapsack.random_solution(rng);
        const double branchless = ns_per_call([&] { sink += knapsack.evaluate(x); });
        const double branchy = ns_per_call([&] { sink += knapsack.evaluate_branchy(x); });
        row({"knapsack branchless, n=" + std::to_string(n), format_number(branchless, 1),
             format_number(branchless / n, 3), "1.00"},
            widths);
        row({"knapsack branchy, n=" + std::to_string(n), format_number(branchy, 1),
             format_number(branchy / n, 3), format_number(branchy / branchless, 2)},
            widths);
    }
    rule();
    std::printf("The knapsack objective is a contiguous multiply and accumulate with no branch\n");
    std::printf("and no gather, so it is the only objective here a compiler can vectorise. The\n");
    std::printf("branchy form is what most people write first and is the same computation.\n\n");

    row({"objective", "ns per call", "ns per item", "ratio"}, widths);
    rule();
    for (int n : {200, 1000, 2000, 4000}) {
        TspProblem tsp = make_uniform_tsp(n, 999);
        Rng rng(7);
        const Solution x = tsp.random_solution(rng);
        tsp.set_distance_mode(true);
        const double matrix = ns_per_call([&] { sink += tsp.evaluate(x); });
        tsp.set_distance_mode(false);
        const double coords = ns_per_call([&] { sink += tsp.evaluate(x); });
        row({"tour length via matrix, n=" + std::to_string(n), format_number(matrix, 1),
             format_number(matrix / n, 3), "1.00"},
            widths);
        row({"tour length via coordinates, n=" + std::to_string(n), format_number(coords, 1),
             format_number(coords / n, 3), format_number(coords / matrix, 2)},
            widths);
    }
    rule();
    std::printf("The tour length is a gather through a permutation, so neither form vectorises.\n");
    std::printf("The matrix is n^2 doubles and leaves cache; the coordinates are 2n and do not.\n");
}

// ---------------------------------------------------------------------------
void benchmark_neighbourhood() {
    heading("How the 2-opt move is chosen: the largest single effect measured here");
    std::printf("Command: anneal_bench neighbourhood\n");
    std::printf("Pure descent, two million proposals, from the same nearest neighbour tour.\n");
    std::printf("Descent accepts only improving moves, so the column is the quality the move\n");
    std::printf("generator can reach on its own, with no metaheuristic on top of it.\n\n");

    const std::vector<int> widths{-30, 10, 14, 12, 14, 12};
    row({"move generator", "span", "tour length", "improved", "ms", "ns per move"}, widths);
    rule();
    const int proposals = 2000000;
    for (int n : {1000}) {
        TspProblem reference = make_uniform_tsp(n, 20260819);
        Rng seed_rng(1);
        const Solution start = reference.construct(seed_rng);
        const double start_cost = reference.evaluate(start);
        row({"nearest neighbour start", "-", format_number(start_cost, 1), "-", "-", "-"},
            widths);

        struct Setting {
            const char* label;
            bool candidate;
            std::size_t span;
        };
        const Setting settings[] = {
            {"uniform 2-opt", false, static_cast<std::size_t>(n - 1)},
            {"uniform 2-opt, bounded", false, 50},
            {"neighbour list, bounded", true, 50},
            {"neighbour list, bounded", true, 200},
            {"neighbour list", true, static_cast<std::size_t>(n - 1)},
        };
        for (const Setting& setting : settings) {
            TspProblem p = make_uniform_tsp(n, 20260819);
            p.set_candidate_moves(setting.candidate);
            p.set_span(setting.span);
            Rng rng(1);
            Candidate c = p.make(start);
            int improved = 0;
            const Clock::time_point t0 = Clock::now();
            for (int i = 0; i < proposals; ++i) {
                const Move m = p.random_move(c, rng);
                const double d = p.delta(c, m);
                if (d < 0.0) {
                    p.apply(c, m, d);
                    ++improved;
                }
            }
            const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            row({setting.label, std::to_string(setting.span), format_number(c.cost, 1),
                 std::to_string(improved), format_number(ms, 0),
                 format_number(ms * 1e6 / proposals, 1)},
                widths);
        }
    }
    rule();
    std::printf("The neighbour list move names a city and one of its sixteen nearest, then\n");
    std::printf("looks that neighbour up in the inverse permutation carried by the candidate.\n");
    std::printf("It always installs a short edge, so most proposals are repairs.\n");
}

// ---------------------------------------------------------------------------
void benchmark_false_sharing() {
    heading("False sharing: per worker counters, packed against padded");
    std::printf("Command: anneal_bench falsesharing\n");
    std::printf("Every thread increments its own counter through a volatile pointer, so the\n");
    std::printf("store reaches memory on every iteration. Stride 1 puts eight counters on one\n");
    std::printf("64 byte line; stride 8 gives each counter a line of its own.\n\n");

    const unsigned hardware = std::max(1u, std::thread::hardware_concurrency());
    const std::vector<int> widths{-10, 16, 16, 10};
    row({"threads", "packed ms", "padded ms", "ratio"}, widths);
    rule();

    const std::uint64_t iterations = 20000000;
    for (int threads : {2, 4, 8, 12}) {
        if (threads > static_cast<int>(hardware)) continue;
        double timings[2] = {0.0, 0.0};
        int index = 0;
        for (std::size_t stride : {std::size_t{1}, std::size_t{8}}) {
            std::vector<std::uint64_t> counters(static_cast<std::size_t>(threads) * stride + 8, 0);
            const Clock::time_point start = Clock::now();
            {
                std::vector<std::jthread> pool;
                pool.reserve(static_cast<std::size_t>(threads));
                for (int t = 0; t < threads; ++t) {
                    pool.emplace_back([&counters, stride, t] {
                        volatile std::uint64_t* slot =
                            &counters[static_cast<std::size_t>(t) * stride];
                        for (std::uint64_t i = 0; i < iterations; ++i) *slot = *slot + 1;
                    });
                }
            }
            timings[index++] =
                std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        }
        row({std::to_string(threads), format_number(timings[0], 1), format_number(timings[1], 1),
             format_number(timings[0] / timings[1], 2)},
            widths);
    }
    rule();
    std::printf("A ratio above one is the cost of sharing a cache line. The per worker state in\n");
    std::printf("src/parallel.cpp is aligned to 64 bytes for exactly this reason.\n");
}

// ---------------------------------------------------------------------------
void benchmark_profile() {
    heading("Time attribution inside one annealing iteration, by ablation");
    std::printf("Command: anneal_bench profile\n");
    std::printf("Method: no external profiler was run, and none is claimed. Each row times a\n");
    std::printf("loop containing a prefix of the work an iteration does, so the difference\n");
    std::printf("between two rows is the cost of the part that was added. This attributes time\n");
    std::printf("without the per-call clock reads that instrumentation would need, which on\n");
    std::printf("this workload would cost more than the work being measured.\n\n");

    const std::vector<int> widths{-40, 14, 14, 10};
    row({"loop body", "ns per pass", "difference", "share"}, widths);
    rule();

    for (int n : {200, 1000}) {
        TspProblem tsp = make_uniform_tsp(n, 4242);
        Rng rng(11);
        Candidate c = tsp.make(tsp.construct(rng));

        const double draw = ns_per_call([&] { sink += rng.uniform(); });
        const double with_move = ns_per_call([&] {
            const Move m = tsp.random_move(c, rng);
            sink += m.i + m.j;
        });
        const double with_delta = ns_per_call([&] {
            const Move m = tsp.random_move(c, rng);
            sink += tsp.delta(c, m);
        });
        const double full_step = ns_per_call([&] {
            const Move m = tsp.random_move(c, rng);
            const double d = tsp.delta(c, m);
            if (metropolis_accepts(d, 1.0, rng.uniform())) tsp.apply(c, m, d);
        });
        std::printf("instance %s\n", tsp.name().c_str());
        auto share = [&](double part) {
            return format_number(100.0 * part / full_step, 1) + " %";
        };
        row({"  random draw only", format_number(draw, 1), "-", share(draw)}, widths);
        row({"  + move generation", format_number(with_move, 1),
             format_number(with_move - draw, 1), share(with_move - draw)},
            widths);
        row({"  + incremental evaluation", format_number(with_delta, 1),
             format_number(with_delta - with_move, 1), share(with_delta - with_move)},
            widths);
        row({"  + acceptance and apply", format_number(full_step, 1),
             format_number(full_step - with_delta, 1), share(full_step - with_delta)},
            widths);
        const double full_eval = ns_per_call([&] { sink += tsp.evaluate(c.x); });
        row({"  full re-evaluation, for scale", format_number(full_eval, 1), "-",
             format_number(full_eval / full_step, 0) + "x"},
            widths);
        rule();
    }
    std::printf("The last row is the ratio a full re-evaluation would cost per iteration, which\n");
    std::printf("is what the incremental evaluation exists to avoid.\n");
}

// ---------------------------------------------------------------------------
struct Instances {
    TspProblem uniform = make_uniform_tsp(1000, 20260819);
    TspProblem grid = make_grid_tsp(24, 10.0);
    KnapsackProblem knapsack = make_random_knapsack(400, 20260819);
    ColoringProblem coloring = make_planted_coloring(150, 8, 0.15, 20260819);
};

void benchmark_quality(int repetitions, int budget_ms) {
    heading("Solution quality at a fixed wall clock budget, one thread");
    std::printf("Command: anneal_bench quality\n");
    std::printf("Budget %d ms per repetition, %d seeds, independent multi-start with one\n",
                budget_ms, repetitions);
    std::printf("thread, so this measures the algorithms and not the schemes. The gap is\n");
    std::printf("relative to the known value where one exists; for graph colouring the known\n");
    std::printf("value is zero conflicts, so the gap is an absolute count of conflicts.\n\n");

    Instances instances;
    const Problem* problems[] = {&instances.uniform, &instances.grid, &instances.knapsack,
                                 &instances.coloring};
    const std::vector<int> widths{-18, -11, 14, 14, 14, 13};
    row({"instance", "algorithm", "median", "IQR", "median gap", "median evals"}, widths);
    rule();
    for (const Problem* p : problems) {
        for (const std::string algorithm : {"annealing", "tabu", "genetic", "aco"}) {
            Experiment e;
            e.problem = p;
            e.algorithm = algorithm;
            e.scheme = "multistart";
            e.threads = 1;
            e.repetitions = repetitions;
            e.budget.wall = std::chrono::milliseconds{budget_ms};
            e.base_seed = 900;
            const std::vector<Trial> trials = run_experiment(e);
            const Summary quality = summarise(column_display(trials));
            const Summary gap = summarise(column_gap(trials));
            std::vector<double> evaluations;
            for (const Trial& t : trials) {
                evaluations.push_back(static_cast<double>(t.evaluations));
            }
            row({p->name(), algorithm, format_number(quality.median, 2),
                 format_number(quality.iqr(), 2), gap_cell(gap.median, p->best_known()),
                 format_number(summarise(evaluations).median, 0)},
                widths);
        }
        rule();
    }
}

// ---------------------------------------------------------------------------
void benchmark_schemes(int repetitions, int budget_ms) {
    const unsigned hardware = std::max(1u, std::thread::hardware_concurrency());
    const int threads = static_cast<int>(std::min(8u, hardware));
    heading("The three parallel schemes at equal wall clock budget");
    std::printf("Command: anneal_bench schemes\n");
    std::printf("%d threads, %d ms per repetition, %d seeds, tabu search throughout.\n", threads,
                budget_ms, repetitions);
    std::printf("sync ms is the total across workers, against %d ms of budget each.\n\n",
                budget_ms);

    Instances instances;
    const Problem* problems[] = {&instances.uniform, &instances.knapsack, &instances.coloring};

    const std::vector<int> widths{-18, -13, 13, 13, 13, 11, 11, 10};
    row({"instance", "scheme", "median", "IQR", "median gap", "sync ms", "CAS", "adopted"},
        widths);
    rule();
    for (const Problem* p : problems) {
        for (const std::string scheme : {"multistart", "cooperative", "island"}) {
            Experiment e;
            e.problem = p;
            e.algorithm = "tabu";
            e.scheme = scheme;
            e.threads = threads;
            e.repetitions = repetitions;
            e.budget.wall = std::chrono::milliseconds{budget_ms};
            e.base_seed = 6100;
            const std::vector<Trial> trials = run_experiment(e);
            const Summary quality = summarise(column_display(trials));
            const Summary gap = summarise(column_gap(trials));
            std::vector<double> sync;
            std::vector<double> cas;
            std::vector<double> adopted;
            for (const Trial& t : trials) {
                sync.push_back(static_cast<double>(t.sync_wait_ns) / 1e6);
                cas.push_back(static_cast<double>(t.cas_failures));
                adopted.push_back(static_cast<double>(t.adopted));
            }
            row({p->name(), scheme, format_number(quality.median, 2),
                 format_number(quality.iqr(), 2), gap_cell(gap.median, p->best_known()),
                 format_number(summarise(sync).median, 3),
                 format_number(summarise(cas).median, 0),
                 format_number(summarise(adopted).median, 0)},
                widths);
        }
        rule();
    }
}

// ---------------------------------------------------------------------------
void benchmark_speedup(int repetitions) {
    const unsigned hardware = std::max(1u, std::thread::hardware_concurrency());
    heading("Speedup and efficiency against thread count, time to target");
    std::printf("Command: anneal_bench speedup\n");
    std::printf("The target is the worst quality a single thread reached in a pilot of five\n");
    std::printf("400 ms runs, so one thread reaches it reliably and the baseline is measured\n");
    std::printf("rather than assumed. Cap 3000 ms per repetition, %d seeds per cell, tabu.\n",
                repetitions);

    TspProblem tsp = make_uniform_tsp(1000, 20260819);
    Experiment pilot;
    pilot.problem = &tsp;
    pilot.algorithm = "tabu";
    pilot.scheme = "multistart";
    pilot.threads = 1;
    pilot.repetitions = 5;
    pilot.budget.wall = std::chrono::milliseconds{400};
    pilot.base_seed = 77;
    const Summary pilot_quality = summarise(column_display(run_experiment(pilot)));
    const double target = pilot_quality.max;
    std::printf("instance %s, pilot best %s, pilot worst %s, target %s\n\n", tsp.name().c_str(),
                format_number(pilot_quality.min, 1).c_str(),
                format_number(pilot_quality.max, 1).c_str(), format_number(target, 1).c_str());

    const std::vector<int> widths{-13, 9, 14, 12, 10, 12, 8};
    row({"scheme", "threads", "median t2t ms", "IQR ms", "speedup", "efficiency", "hits"},
        widths);
    rule();
    for (const std::string scheme : {"multistart", "cooperative", "island"}) {
        double baseline = 0.0;
        for (int threads : {1, 2, 3, 4, 6, 8, 10, 12}) {
            if (threads > static_cast<int>(hardware)) continue;
            Experiment e;
            e.problem = &tsp;
            e.algorithm = "tabu";
            e.scheme = scheme;
            e.threads = threads;
            e.repetitions = repetitions;
            e.budget.wall = std::chrono::milliseconds{3000};
            e.budget.target = target;
            e.base_seed = 5000;
            const std::vector<Trial> trials = run_experiment(e);
            const Summary t2t = summarise(column_time_to_target(trials));
            if (threads == 1) baseline = t2t.median;
            const double s = speedup(baseline, t2t.median);
            row({scheme, std::to_string(threads), format_number(t2t.median, 2),
                 format_number(t2t.iqr(), 2), format_number(s, 2),
                 format_number(efficiency(s, threads), 2),
                 format_number(hit_rate(trials) * 100.0, 0) + "%"},
                widths);
        }
        rule();
    }
    std::printf("A speedup above the thread count is not an error. The time to target of a\n");
    std::printf("single run is heavily skewed, so p independent runs finish in the minimum of p\n");
    std::printf("draws from that distribution, which falls faster than 1/p while the tail is\n");
    std::printf("long. The interquartile range column is what makes this visible.\n");
}

}  // namespace

int main(int argc, char** argv) {
    const std::string what = argc > 1 ? argv[1] : "all";
    const unsigned hardware = std::max(1u, std::thread::hardware_concurrency());
    std::printf("anneal_bench: %u logical CPUs, std::chrono::steady_clock\n", hardware);

    const std::vector<std::string> known{"objective", "neighbourhood", "falsesharing",
                                         "profile",   "quality",       "schemes",
                                         "speedup"};
    const bool all = what == "all";
    if (!all && std::find(known.begin(), known.end(), what) == known.end()) {
        std::printf("unknown sub-command '%s'\ntry: all", what.c_str());
        for (const std::string& k : known) std::printf(" %s", k.c_str());
        std::printf("\n");
        return 2;
    }

    if (all || what == "objective") benchmark_objective();
    if (all || what == "neighbourhood") benchmark_neighbourhood();
    if (all || what == "falsesharing") benchmark_false_sharing();
    if (all || what == "profile") benchmark_profile();
    if (all || what == "quality") benchmark_quality(11, 250);
    if (all || what == "schemes") benchmark_schemes(11, 300);
    if (all || what == "speedup") benchmark_speedup(11);

    std::printf("\nsink = %g (printed so the timed loops cannot be optimised away)\n", sink);
    return 0;
}
