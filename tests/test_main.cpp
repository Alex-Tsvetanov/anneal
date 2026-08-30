// Correctness tests. Timing tests live in benchmarks/, because a correctness
// test that fails when the machine is busy teaches everyone to ignore it.
#include "check.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <set>
#include <thread>
#include <vector>

#include "anneal/algorithms.hpp"
#include "anneal/parallel.hpp"
#include "anneal/problems.hpp"
#include "anneal/stats.hpp"
#include "anneal/tsplib.hpp"

using namespace anneal;

namespace {

constexpr double kPi = 3.14159265358979323846;

// Checks the incremental evaluation against a full re-evaluation over many
// random moves. This is the single most valuable test in the project: a wrong
// delta does not crash and does not look wrong, it just quietly reports a cost
// the solution does not have.
void check_delta_against_full(const Problem& p, std::uint64_t seed, int trials) {
    Rng rng(seed);
    Candidate c = p.make(p.construct(rng));
    CHECK_NEAR(c.cost, p.evaluate(c.x), 1e-9);
    int applied = 0;
    for (int i = 0; i < trials; ++i) {
        const Move m = p.random_move(c, rng);
        const double d = p.delta(c, m);
        if (!std::isfinite(d)) continue;  // infeasible move, correctly refused
        const double expected = p.evaluate(c.x) + d;
        p.apply(c, m, d);
        ++applied;
        CHECK_NEAR(c.cost, expected, 1e-6);
        CHECK_NEAR(c.cost, p.evaluate(c.x), 1e-6);
        CHECK_NEAR(c.aux, p.aux_of(c.x), 1e-6);
        CHECK(p.feasible(c.x));
    }
    CHECK_MSG(applied > trials / 10, "almost every move was refused, so the test proved nothing");
}

}  // namespace

// ---------------------------------------------------------------------------
// Generator and statistics.
// ---------------------------------------------------------------------------

TEST(rng_reproducible) {
    Rng a(12345), b(12345), c(12346);
    for (int i = 0; i < 100; ++i) {
        const std::uint64_t x = a.next();
        CHECK_EQ(x, b.next());
    }
    // Two seeds must not produce the same stream; splitmix64 seeding is what
    // makes a nearby seed diverge rather than lag by one.
    Rng d(12345);
    bool differs = false;
    for (int i = 0; i < 20; ++i) {
        if (c.next() != d.next()) differs = true;
    }
    CHECK(differs);
    // Derived worker streams must differ from each other.
    CHECK(Rng::derive(7, 0) != Rng::derive(7, 1));
    CHECK(Rng::derive(7, 0) != Rng::derive(8, 0));
}

TEST(rng_bounded_range) {
    Rng rng(99);
    for (int i = 0; i < 10000; ++i) {
        CHECK(rng.below(7) < 7);
        const int r = rng.range(-3, 4);
        CHECK(r >= -3 && r <= 4);
        const double u = rng.uniform();
        CHECK(u >= 0.0 && u < 1.0);
    }
    CHECK_EQ(rng.below(0), 0u);
    CHECK_EQ(rng.below(1), 0u);
    // Rough uniformity over four buckets. A modulo-biased generator fails this
    // only for large bounds, so the bound is chosen near 2^63 where the bias
    // of `next() % n` would be a factor of two.
    const std::uint64_t bound = (1ULL << 63) + (1ULL << 62);
    std::vector<int> buckets(4, 0);
    const int draws = 40000;
    for (int i = 0; i < draws; ++i) {
        const std::uint64_t v = rng.below(bound);
        CHECK(v < bound);
        buckets[static_cast<std::size_t>(v / (bound / 4 + 1))] += 1;
    }
    for (int count : buckets) {
        CHECK_MSG(count > draws / 4 - draws / 20 && count < draws / 4 + draws / 20,
                  "bucket " + std::to_string(count) + " is off by more than 5 per cent");
    }
}

TEST(stats_median_quartiles) {
    std::vector<double> odd{9, 1, 8, 2, 7, 3, 6, 4, 5};
    const Summary s = summarise(odd);
    CHECK_EQ(s.n, 9u);
    CHECK_NEAR(s.median, 5.0, 1e-12);
    CHECK_NEAR(s.q1, 3.0, 1e-12);
    CHECK_NEAR(s.q3, 7.0, 1e-12);
    CHECK_NEAR(s.min, 1.0, 1e-12);
    CHECK_NEAR(s.max, 9.0, 1e-12);
    CHECK_NEAR(s.mean, 5.0, 1e-12);
    CHECK_NEAR(s.iqr(), 4.0, 1e-12);

    std::vector<double> even{4, 1, 3, 2};
    const Summary t = summarise(even);
    CHECK_NEAR(t.median, 2.5, 1e-12);
    CHECK_NEAR(t.q1, 1.75, 1e-12);
    CHECK_NEAR(t.q3, 3.25, 1e-12);

    CHECK_NEAR(speedup(100.0, 25.0), 4.0, 1e-12);
    CHECK_NEAR(efficiency(4.0, 8), 0.5, 1e-12);
    CHECK_NEAR(relative_gap(110.0, 100.0), 0.1, 1e-12);
    // An optimum of zero would make a ratio undefined, so the denominator is
    // clamped and the gap degrades to an absolute difference.
    CHECK_NEAR(relative_gap(3.0, 0.0), 3.0, 1e-12);
    CHECK_EQ(summarise({}).n, 0u);
}

// ---------------------------------------------------------------------------
// Travelling salesman problem.
// ---------------------------------------------------------------------------

TEST(tsp_circle_optimum_is_known) {
    const int n = 14;
    const double radius = 100.0;
    TspProblem p = make_circle_tsp(n, radius);
    const double closed_form = n * 2.0 * radius * std::sin(kPi / n);
    CHECK_NEAR(p.best_known(), closed_form, 1e-9);

    // The optimal tour of points in convex position is the hull order, so
    // sorting the cities by angle must reproduce the closed form exactly.
    Solution by_angle(static_cast<std::size_t>(n));
    std::iota(by_angle.begin(), by_angle.end(), 0);
    std::sort(by_angle.begin(), by_angle.end(), [&](int a, int b) {
        return std::atan2(p.ys()[static_cast<std::size_t>(a)], p.xs()[static_cast<std::size_t>(a)]) <
               std::atan2(p.ys()[static_cast<std::size_t>(b)], p.xs()[static_cast<std::size_t>(b)]);
    });
    CHECK_NEAR(p.evaluate(by_angle), closed_form, 1e-6);

    // And nothing beats it. If a random tour did, the closed form would be
    // wrong and every gap computed against it would be wrong too.
    Rng rng(4);
    for (int trial = 0; trial < 2000; ++trial) {
        const double cost = p.evaluate(p.random_solution(rng));
        CHECK_MSG(cost >= closed_form - 1e-6, "a random tour beat the stated optimum");
    }
    // The identity permutation must not already be optimal, otherwise the
    // instance measures nothing.
    Solution identity(static_cast<std::size_t>(n));
    std::iota(identity.begin(), identity.end(), 0);
    CHECK(p.evaluate(identity) > closed_form + 1.0);
}

TEST(tsp_grid_optimum_is_known) {
    const int k = 8;
    const double spacing = 10.0;
    TspProblem p = make_grid_tsp(k, spacing);
    const int n = k * k;
    CHECK_EQ(p.size(), static_cast<std::size_t>(n));
    CHECK_NEAR(p.best_known(), n * spacing, 1e-9);

    // The lower bound is n * spacing because no two distinct grid points are
    // closer than the spacing. The bound is attained, and this is the tour
    // that attains it: along the bottom row, then a boustrophedon over the
    // remaining columns, then back up the reserved first column.
    auto at = [k](int x, int y) { return y * k + x; };
    Solution tour;
    tour.reserve(static_cast<std::size_t>(n));
    for (int x = 0; x < k; ++x) tour.push_back(at(x, 0));
    for (int y = 1; y < k; ++y) {
        if (y % 2 == 1) {
            for (int x = k - 1; x >= 1; --x) tour.push_back(at(x, y));
        } else {
            for (int x = 1; x < k; ++x) tour.push_back(at(x, y));
        }
    }
    for (int y = k - 1; y >= 1; --y) tour.push_back(at(0, y));
    CHECK_EQ(tour.size(), static_cast<std::size_t>(n));
    CHECK_MSG(p.feasible(tour), "the constructed optimal tour is not a permutation");
    CHECK_NEAR(p.evaluate(tour), n * spacing, 1e-6);

    // An odd grid has no such tour, so the generator refuses rather than
    // reporting an optimum that cannot be reached.
    bool threw = false;
    try {
        make_grid_tsp(7);
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);

    // And, unlike the circle, the nearest neighbour construction does not
    // already solve it. That is why this is the instance the demo uses.
    Rng rng(3);
    bool nn_is_suboptimal = false;
    for (int i = 0; i < 20; ++i) {
        if (p.evaluate(p.construct(rng)) > n * spacing + 1e-6) nn_is_suboptimal = true;
    }
    CHECK_MSG(nn_is_suboptimal, "nearest neighbour already solves the grid, so it measures nothing");
}

TEST(tsp_delta_matches_full_evaluation) {
    TspProblem p = make_uniform_tsp(60, 11);
    check_delta_against_full(p, 7, 400);
    // The two objective forms must agree; they are measured against each
    // other in the benchmark, and a disagreement would make that comparison
    // meaningless.
    Rng rng(3);
    for (int i = 0; i < 50; ++i) {
        const Solution x = p.random_solution(rng);
        CHECK_NEAR(p.evaluate(x), p.evaluate_from_coords(x), 1e-6);
    }
}

TEST(tsp_move_preserves_permutation) {
    TspProblem p = make_uniform_tsp(40, 5);
    Rng rng(6);
    Candidate c = p.make(p.construct(rng));
    CHECK(p.feasible(c.x));
    for (int i = 0; i < 500; ++i) {
        const Move m = p.random_move(c, rng);
        CHECK(m.i >= 0);
        CHECK(m.j > m.i);
        CHECK(m.j < 40);
        p.apply(c, m, p.delta(c, m));
        CHECK(p.feasible(c.x));
    }
}

TEST(tsp_crossover_preserves_permutation) {
    TspProblem p = make_uniform_tsp(30, 8);
    Rng rng(9);
    for (int i = 0; i < 200; ++i) {
        const Solution a = p.random_solution(rng);
        const Solution b = p.random_solution(rng);
        const Solution child = p.crossover(a, b, rng);
        CHECK_MSG(p.feasible(child), "order crossover produced a non-permutation");
        CHECK_EQ(child.size(), a.size());
    }
    // A child of two identical parents is that parent.
    const Solution a = p.random_solution(rng);
    CHECK(p.crossover(a, a, rng) == a);
}

TEST(tsplib_parse_reads_coordinates) {
    const std::string text =
        "NAME : toy4\n"
        "TYPE : TSP\n"
        "COMMENT : four corners\n"
        "DIMENSION : 4\n"
        "EDGE_WEIGHT_TYPE : EUC_2D\n"
        "NODE_COORD_SECTION\n"
        "1 0.0 0.0\n"
        "2 3.0 0.0\n"
        "3 3.0 4.0\n"
        "4 0.0 4.0\n"
        "EOF\n";
    const TsplibInstance instance = parse_tsplib(text);
    CHECK_EQ(instance.name, std::string("toy4"));
    CHECK_EQ(instance.xs.size(), 4u);
    CHECK_NEAR(instance.xs[1], 3.0, 1e-12);
    CHECK_NEAR(instance.ys[2], 4.0, 1e-12);

    TspProblem p = tsplib_to_problem(instance);
    CHECK_NEAR(p.distance(0, 1), 3.0, 1e-12);
    CHECK_NEAR(p.distance(0, 2), 5.0, 1e-12);
    Solution rectangle{0, 1, 2, 3};
    CHECK_NEAR(p.evaluate(rectangle), 14.0, 1e-12);
    CHECK(std::isnan(p.best_known()));

    // A dimension that disagrees with the section must be refused rather than
    // silently trusted.
    bool threw = false;
    try {
        parse_tsplib("DIMENSION : 9\nNODE_COORD_SECTION\n1 0 0\nEOF\n");
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK_MSG(threw, "a DIMENSION mismatch was accepted");
}

TEST(tsplib_roundtrip_preserves_distances) {
    TspProblem original = make_circle_tsp(17, 50.0);
    TsplibInstance instance;
    instance.name = "circle17";
    instance.xs = original.xs();
    instance.ys = original.ys();
    const TsplibInstance reparsed = parse_tsplib(write_tsplib(instance));
    CHECK_EQ(reparsed.xs.size(), instance.xs.size());
    // Continuous mode: the round trip is about the coordinates, not about the
    // TSPLIB nint convention used for published optima.
    TspProblem rebuilt = tsplib_to_problem(reparsed, original.best_known(),
                                           Euc2dMode::Continuous);
    Rng rng(2);
    for (int i = 0; i < 40; ++i) {
        const Solution x = original.random_solution(rng);
        // The writer emits six decimals, so agreement is to that precision and
        // not to the bit. Claiming more would be claiming a round trip the
        // format does not provide.
        CHECK_NEAR(original.evaluate(x), rebuilt.evaluate(x), 1e-3);
    }
}

TEST(berlin52_published_optimum_is_attained) {
    // The known value is the published TSPLIB optimum, not a number invented
    // in this repository. The tour in data/berlin52.opt.tour is the published
    // tour; under EUC_2D nint it must evaluate to exactly 7542.
    TspProblem p = make_berlin52_tsp();
    CHECK_EQ(p.name(), std::string("berlin52"));
    CHECK_EQ(p.size(), 52u);
    CHECK_EQ(p.euc2d_mode(), Euc2dMode::TsplibNint);
    CHECK_NEAR(p.best_known(), 7542.0, 1e-12);

    // The tour in data/berlin52.opt.tour is the published tour; under EUC_2D
    // nint it must evaluate to exactly 7542. Read it here rather than invent
    // a tour that happens to hit the number.
    std::ifstream tour_file(std::string(ANNEAL_DATA_DIR) + "/berlin52.opt.tour");
    CHECK_MSG(static_cast<bool>(tour_file), "missing data/berlin52.opt.tour");
    std::string line;
    Solution tour;
    bool in_tour = false;
    while (std::getline(tour_file, line)) {
        if (line == "TOUR_SECTION") {
            in_tour = true;
            continue;
        }
        if (!in_tour) continue;
        if (line == "-1" || line == "EOF") break;
        const int city = std::stoi(line);
        CHECK_MSG(city >= 1 && city <= 52, "tour city out of range");
        tour.push_back(city - 1);  // TSPLIB is 1-based
    }
    CHECK_EQ(tour.size(), 52u);
    CHECK(p.feasible(tour));
    CHECK_NEAR(p.evaluate(tour), 7542.0, 1e-9);
    CHECK_NEAR(p.evaluate_from_coords(tour), 7542.0, 1e-9);

    // Uniform random stays unknown: never invent a number for it.
    TspProblem uniform = make_uniform_tsp(40, 5);
    CHECK(std::isnan(uniform.best_known()));
}

// ---------------------------------------------------------------------------
// Knapsack.
// ---------------------------------------------------------------------------

TEST(knapsack_delta_matches_full_evaluation) {
    KnapsackProblem p = make_random_knapsack(80, 21);
    check_delta_against_full(p, 13, 500);
}

TEST(knapsack_dp_optimum_matches_brute_force) {
    // Small enough to enumerate every subset, which is the only way to be sure
    // the dynamic programme is right; every reported knapsack gap is measured
    // against its answer.
    const int n = 16;
    KnapsackProblem p = make_random_knapsack(n, 33, 20);
    const std::vector<double>& w = p.weights();
    const std::vector<double>& v = p.values();
    double brute = 0.0;
    for (std::uint32_t mask = 0; mask < (1u << n); ++mask) {
        double weight = 0.0;
        double value = 0.0;
        for (int i = 0; i < n; ++i) {
            if (mask & (1u << i)) {
                weight += w[static_cast<std::size_t>(i)];
                value += v[static_cast<std::size_t>(i)];
            }
        }
        if (weight <= p.capacity() && value > brute) brute = value;
    }
    CHECK_NEAR(KnapsackProblem::dp_optimum(w, v, p.capacity()), brute, 1e-9);
    // best_known() is stored in internal minimised units, so it is the
    // negated optimum and display() undoes the sign.
    CHECK_NEAR(p.display(p.best_known()), brute, 1e-9);
}

TEST(knapsack_solution_stays_feasible) {
    KnapsackProblem p = make_random_knapsack(50, 44);
    Rng rng(17);
    for (int i = 0; i < 100; ++i) {
        CHECK(p.feasible(p.construct(rng)));
        CHECK(p.feasible(p.random_solution(rng)));
        CHECK(p.feasible(p.crossover(p.construct(rng), p.random_solution(rng), rng)));
        CHECK(p.feasible(p.construct_aco(std::vector<double>(p.size(), 1.0), 1.0, 2.0, rng)));
    }
    // A move that would overflow the sack has to be refused, not penalised.
    Candidate full = p.make(Solution(p.size(), 0));
    for (std::size_t i = 0; i < p.size(); ++i) {
        Move m{static_cast<int>(i), -1};
        const double d = p.delta(full, m);
        if (std::isfinite(d)) p.apply(full, m, d);
    }
    CHECK(p.feasible(full.x));
    CHECK(full.aux <= p.capacity() + 1e-9);
}

TEST(knapsack_vectorised_matches_scalar) {
    // The branchless form is the one that can vectorise; the branchy form is
    // what most people write. The benchmark reports which is faster here, and
    // that report is only worth something if the two agree exactly.
    KnapsackProblem p = make_random_knapsack(256, 55);
    Rng rng(1234);
    for (int i = 0; i < 200; ++i) {
        const Solution x = p.random_solution(rng);
        CHECK_NEAR(p.evaluate(x), p.evaluate_branchy(x), 1e-9);
    }
    CHECK_NEAR(p.evaluate(Solution(p.size(), 0)), 0.0, 1e-12);
}

// ---------------------------------------------------------------------------
// Graph colouring.
// ---------------------------------------------------------------------------

TEST(coloring_delta_matches_full_evaluation) {
    ColoringProblem p = make_planted_coloring(80, 5, 0.35, 77);
    check_delta_against_full(p, 19, 500);
}

TEST(coloring_planted_partition_is_conflict_free) {
    Solution planted;
    ColoringProblem p = make_planted_coloring(120, 6, 0.3, 88, &planted);
    CHECK(p.edge_count() > 100u);
    CHECK_EQ(planted.size(), p.size());
    CHECK(p.feasible(planted));
    // The stated optimum is zero conflicts. If the planted colouring did not
    // achieve it, the instance would be unsolvable and every gap against zero
    // would be a lie.
    CHECK_NEAR(p.evaluate(planted), 0.0, 1e-12);
    CHECK_NEAR(p.best_known(), 0.0, 1e-12);
    // A random colouring is far from it, so the instance is not trivial.
    Rng rng(23);
    CHECK(p.evaluate(p.random_solution(rng)) > 10.0);
}

TEST(coloring_crossover_stays_in_palette) {
    ColoringProblem p = make_planted_coloring(60, 4, 0.4, 91);
    Rng rng(31);
    for (int i = 0; i < 200; ++i) {
        const Solution child = p.crossover(p.random_solution(rng), p.construct(rng), rng);
        CHECK_EQ(child.size(), p.size());
        CHECK(p.feasible(child));
        for (int colour : child) CHECK(colour >= 0 && colour < p.colours());
    }
}

// ---------------------------------------------------------------------------
// Metaheuristics.
// ---------------------------------------------------------------------------

TEST(annealing_accepts_worse_only_when_warm) {
    // Downhill is always taken, at any temperature including zero.
    CHECK(metropolis_accepts(-1.0, 0.0, 0.999));
    CHECK(metropolis_accepts(0.0, 0.0, 0.999));
    // Uphill is never taken at zero temperature.
    CHECK(!metropolis_accepts(1e-9, 0.0, 0.0));
    // At temperature one an uphill step of one is taken with probability
    // exp(-1), which is 0.3679, so the two draws either side of it decide.
    CHECK(metropolis_accepts(1.0, 1.0, 0.36));
    CHECK(!metropolis_accepts(1.0, 1.0, 0.37));
    // Hot enough and almost everything is accepted.
    CHECK(metropolis_accepts(1.0, 1e6, 0.99));

    AnnealingConfig cfg;
    cfg.alpha = 0.9;
    double previous = cooled(cfg, 100.0, 0);
    CHECK_NEAR(previous, 100.0, 1e-12);
    for (std::uint64_t k = 1; k < 50; ++k) {
        const double t = cooled(cfg, 100.0, k);
        CHECK_MSG(t < previous, "the geometric schedule failed to decrease");
        previous = t;
    }
    CHECK_NEAR(cooled(cfg, 100.0, 1), 90.0, 1e-9);
    cfg.schedule = Cooling::Linear;
    CHECK_NEAR(cooled(cfg, 100.0, 1), 90.0, 1e-9);   // same first step by design
    CHECK_NEAR(cooled(cfg, 100.0, 2), 80.0, 1e-9);   // then a constant decrement
    cfg.schedule = Cooling::Logarithmic;
    CHECK_NEAR(cooled(cfg, 100.0, 0), 100.0, 1e-9);
    CHECK(cooled(cfg, 100.0, 30) > cooled(cfg, 100.0, 31));
}

TEST(annealing_never_reports_worse_than_start) {
    // The grid instance, not the circle one: on points in convex position the
    // nearest neighbour construction already returns the optimal tour, so
    // annealing has nothing left to improve and the test would fail for a
    // reason that is a property of the instance rather than a defect.
    TspProblem p = make_grid_tsp(8, 10.0);
    MetaheuristicPtr sa = make_annealing(p, 1);
    const double start = sa->best_cost();
    double previous = start;
    // The step count is not arbitrary. The initial temperature is calibrated
    // for an acceptance rate of 0.8, and the geometric schedule needs about
    // 2500 cooling steps of 50 moves each to reach the floor from there. A
    // shorter run leaves the chain hot, and a hot chain is a random walk whose
    // best is still the construction it started from. That is a property of
    // the schedule, not a defect, and it is why the demo budget is measured in
    // millions of moves rather than thousands.
    const int steps = 400000;
    for (int i = 0; i < steps; ++i) {
        sa->step();
        CHECK_MSG(sa->best_cost() <= previous + 1e-9, "the reported best got worse");
        previous = sa->best_cost();
    }
    CHECK(p.feasible(sa->best()));
    CHECK_NEAR(sa->best_cost(), p.evaluate(sa->best()), 1e-6);
    CHECK_MSG(sa->best_cost() < start, "annealing failed to improve on its construction");
    CHECK(sa->best_cost() >= p.best_known() - 1e-6);
    CHECK(sa->evaluations() > static_cast<std::uint64_t>(steps));
}

TEST(tabu_forbids_recent_move) {
    TabuTable table(1024);
    const std::uint64_t key = 4242;
    CHECK(!table.is_tabu(key, 0));
    table.forbid(key, 0, 5);
    for (std::uint64_t i = 0; i < 5; ++i) CHECK(table.is_tabu(key, i));
    CHECK_MSG(!table.is_tabu(key, 5), "the tenure did not expire");
    CHECK(!table.is_tabu(key, 100));
    // Distinct attributes are independent unless they collide in the table.
    // Finding one that does not collide is part of the claim: if every key
    // collided, the list would forbid everything and the search would stall.
    int independent = 0;
    table.forbid(key, 10, 50);
    for (std::uint64_t other = key + 1; other < key + 40; ++other) {
        if (!table.is_tabu(other, 12)) ++independent;
    }
    CHECK_MSG(independent > 30, "far too many attributes share a slot");
}

TEST(tabu_aspiration_overrides_tabu) {
    // A non-tabu move is always admissible.
    CHECK(tabu_admits(false, 100.0, 50.0));
    // A tabu move that would not beat the incumbent is refused.
    CHECK(!tabu_admits(true, 60.0, 50.0));
    CHECK(!tabu_admits(true, 50.0, 50.0));
    // A tabu move that would beat the incumbent is taken anyway. That is the
    // aspiration criterion, and without it the tabu list can forbid the only
    // route to the best solution in the neighbourhood.
    CHECK(tabu_admits(true, 49.999, 50.0));

    ColoringProblem p = make_planted_coloring(70, 5, 0.3, 101);
    MetaheuristicPtr ts = make_tabu(p, 2);
    const double start = ts->best_cost();
    for (int i = 0; i < 3000; ++i) ts->step();
    CHECK(p.feasible(ts->best()));
    CHECK_NEAR(ts->best_cost(), p.evaluate(ts->best()), 1e-9);
    CHECK_MSG(ts->best_cost() < start, "tabu search failed to improve on its construction");
}

TEST(genetic_population_improves) {
    KnapsackProblem p = make_random_knapsack(120, 61);
    MetaheuristicPtr ga = make_genetic(p, 3);
    const double start = ga->best_cost();
    for (int i = 0; i < 300; ++i) ga->step();
    CHECK(p.feasible(ga->best()));
    CHECK_NEAR(ga->best_cost(), p.evaluate(ga->best()), 1e-9);
    CHECK_MSG(ga->best_cost() < start, "the genetic algorithm never improved");
    // It must not beat the exact optimum, which would mean the optimum, the
    // objective or the feasibility check is wrong.
    CHECK_MSG(ga->best_cost() >= p.best_known() - 1e-9,
              "a solution beat the dynamic programming optimum");
}

TEST(aco_evaporation_and_deposit) {
    std::vector<double> tau{1.0, 2.0, 0.5};
    evaporate(tau, 0.1, 1e-6);
    CHECK_NEAR(tau[0], 0.9, 1e-12);
    CHECK_NEAR(tau[1], 1.8, 1e-12);
    CHECK_NEAR(tau[2], 0.45, 1e-12);
    std::vector<double> tiny{1e-9};
    evaporate(tiny, 0.5, 1e-6);
    CHECK_MSG(tiny[0] >= 1e-6, "the trail floor was not applied, so a trail can die for good");

    TspProblem p = make_circle_tsp(8, 10.0);
    std::vector<double> trail(p.pheromone_size(), 0.0);
    Solution tour{0, 1, 2, 3, 4, 5, 6, 7};
    p.deposit(tour, 2.0, trail);
    // Symmetric instance, so the trail is laid in both directions.
    CHECK_NEAR(trail[0 * 8 + 1], 2.0, 1e-12);
    CHECK_NEAR(trail[1 * 8 + 0], 2.0, 1e-12);
    CHECK_NEAR(trail[7 * 8 + 0], 2.0, 1e-12);  // the closing edge
    CHECK_NEAR(trail[0 * 8 + 3], 0.0, 1e-12);  // an edge not on the tour

    MetaheuristicPtr aco = make_aco(p, 4);
    const double start = aco->best_cost();
    for (int i = 0; i < 60; ++i) aco->step();
    CHECK(p.feasible(aco->best()));
    CHECK(aco->best_cost() <= start + 1e-9);
    CHECK(aco->best_cost() >= p.best_known() - 1e-6);
}

TEST(metaheuristics_are_reproducible) {
    TspProblem p = make_uniform_tsp(70, 12);
    for (const std::string name : {"annealing", "tabu", "genetic", "aco"}) {
        MetaheuristicPtr a = make_algorithm(name, p, 555);
        MetaheuristicPtr b = make_algorithm(name, p, 555);
        MetaheuristicPtr c = make_algorithm(name, p, 556);
        const int steps = name == "genetic" || name == "aco" ? 30 : 2000;
        for (int i = 0; i < steps; ++i) {
            a->step();
            b->step();
            c->step();
        }
        CHECK_MSG(a->best_cost() == b->best_cost(), name + " is not reproducible from its seed");
        CHECK_MSG(a->best() == b->best(), name + " returned a different solution for the same seed");
        CHECK_MSG(a->best_cost() != c->best_cost(), name + " ignores its seed");
        CHECK_EQ(a->evaluations(), b->evaluations());
    }
}

// ---------------------------------------------------------------------------
// Parallel execution.
// ---------------------------------------------------------------------------

TEST(multistart_matches_best_of_workers) {
    TspProblem p = make_grid_tsp(8, 10.0);
    Budget budget;
    budget.evaluations = 400000;
    const Factory factory = [&p](std::uint64_t seed, int) { return make_annealing(p, seed); };
    const RunResult r = run_multistart(factory, 4, budget, 2024);
    CHECK_EQ(r.workers.size(), 4u);
    CHECK(p.feasible(r.best));
    // The reported cost must belong to the reported solution. The two travel
    // separately through the worker collection, so this is the invariant that
    // catches a mismatched pair.
    CHECK_NEAR(r.cost, p.evaluate(r.best), 1e-6);
    CHECK(r.cost >= p.best_known() - 1e-6);
    std::uint64_t counted = 0;
    for (const WorkerMetrics& m : r.workers) {
        CHECK(m.iterations > 0u);
        counted += m.evaluations;
    }
    CHECK_EQ(counted, r.evaluations);
    // No communication means no synchronisation cost at all.
    CHECK_EQ(r.total_sync_wait_ns(), 0u);
    CHECK_EQ(r.total_cas_failures(), 0u);

    // An evaluation budget is deterministic, unlike a wall clock one, so the
    // same seed and thread count must reproduce the result exactly.
    const RunResult again = run_multistart(factory, 4, budget, 2024);
    CHECK_MSG(again.cost == r.cost, "a run under an evaluation budget was not reproducible");
    CHECK(again.best == r.best);
}

TEST(cooperative_best_is_monotone) {
    // The shared best under real contention: many threads offering random
    // costs at once. Afterwards the published cost must be the minimum of
    // everything offered, and the published solution must be the one that
    // carried it. A compare and swap on the cost alone would pass the first
    // check and fail the second.
    SharedBest shared;
    const int threads = 8;
    const int per_thread = 4000;
    std::vector<double> minima(static_cast<std::size_t>(threads), kInfinity);
    {
        std::vector<std::jthread> pool;
        for (int t = 0; t < threads; ++t) {
            pool.emplace_back([&, t] {
                Rng rng(static_cast<std::uint64_t>(t) + 1);
                WorkerMetrics metrics;
                double seen = kInfinity;
                double previous_peek = kInfinity;
                for (int i = 0; i < per_thread; ++i) {
                    const double cost = std::floor(rng.uniform() * 1e6);
                    // The solution encodes its own cost, so a mismatched pair
                    // is detectable from the outside.
                    Solution x{static_cast<int>(cost)};
                    shared.offer(x, cost, metrics);
                    const double peeked = shared.peek();
                    CHECK(peeked <= previous_peek);
                    previous_peek = peeked;
                    if (cost < seen) seen = cost;
                }
                minima[static_cast<std::size_t>(t)] = seen;
            });
        }
    }
    const double expected = *std::min_element(minima.begin(), minima.end());
    CHECK_NEAR(shared.cost(), expected, 1e-9);
    const Solution best = shared.snapshot();
    CHECK_EQ(best.size(), 1u);
    CHECK_NEAR(static_cast<double>(best[0]), expected, 1e-9);
}

TEST(island_migration_transfers_solution) {
    KnapsackProblem p = make_random_knapsack(150, 71);
    Budget budget;
    budget.evaluations = 600000;
    const Factory factory = [&p](std::uint64_t seed, int) { return make_annealing(p, seed); };
    const RunResult r = run_island(factory, 4, budget, 909, 512);
    std::uint64_t offers = 0;
    std::uint64_t adopted = 0;
    for (const WorkerMetrics& m : r.workers) {
        offers += m.offers;
        adopted += m.adopted;
    }
    CHECK_MSG(offers > 0u, "no island ever posted to its neighbour");
    CHECK_MSG(adopted > 0u, "no island ever accepted a migrant");
    CHECK_MSG(r.total_sync_wait_ns() > 0u, "migration was free, which cannot be true");
    CHECK(p.feasible(r.best));
    CHECK_NEAR(r.cost, p.evaluate(r.best), 1e-9);
}

TEST(parallel_schemes_return_feasible_solutions) {
    TspProblem tsp = make_grid_tsp(8, 10.0);
    KnapsackProblem knap = make_random_knapsack(100, 81);
    ColoringProblem col = make_planted_coloring(80, 5, 0.3, 91);
    const Problem* problems[] = {&tsp, &knap, &col};
    Budget budget;
    budget.evaluations = 60000;
    for (const Problem* p : problems) {
        for (const std::string scheme : {"multistart", "cooperative", "island"}) {
            for (const std::string algo : {"annealing", "tabu", "genetic", "aco"}) {
                const Factory factory = [p, &algo](std::uint64_t seed, int) {
                    return make_algorithm(algo, *p, seed);
                };
                const RunResult r = run_scheme(scheme, factory, 3, budget, 4242);
                const std::string label = p->name() + "/" + scheme + "/" + algo;
                CHECK_MSG(p->feasible(r.best), label + " returned an infeasible solution");
                CHECK_MSG(std::fabs(r.cost - p->evaluate(r.best)) < 1e-6,
                          label + " reported a cost that is not the cost of its solution");
                const double reference = p->best_known();
                if (!std::isnan(reference)) {
                    CHECK_MSG(r.cost >= reference - 1e-6, label + " beat the known optimum");
                }
                CHECK_MSG(r.evaluations > 0u, label + " did no work");
            }
        }
    }
}

int main(int argc, char** argv) { return check::run(argc, argv); }
