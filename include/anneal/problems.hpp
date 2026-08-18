// The three concrete problems, plus generators that build instances with a
// known or provably characterised optimum. Nothing here reads the network: the
// demo has to work on a machine that has never seen a benchmark archive.
#pragma once

#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "anneal/core.hpp"

namespace anneal {

// ---------------------------------------------------------------------------
// Travelling salesman problem, symmetric, Euclidean.
// ---------------------------------------------------------------------------
class TspProblem final : public Problem {
public:
    TspProblem(std::vector<double> xs, std::vector<double> ys, std::string label,
               double optimum_tour_length = -1.0);

    std::string name() const override { return label_; }
    std::size_t size() const override { return n_; }

    double evaluate(const Solution& x) const override;
    // Same objective computed from the coordinates instead of the precomputed
    // matrix. Kept because the two differ in memory behaviour, and the
    // difference is measured rather than assumed.
    double evaluate_from_coords(const Solution& x) const;

    bool feasible(const Solution& x) const override;
    // The cached index is the inverse permutation: index[city] is the position
    // that city occupies in the tour.
    void build_index(const Solution& x, std::vector<int>& out) const override;
    Solution construct(Rng& rng) const override;       // randomised nearest neighbour
    Solution random_solution(Rng& rng) const override;

    Move random_move(const Candidate& c, Rng& rng) const override;
    double delta(const Candidate& c, const Move& m) const override;
    void apply(Candidate& c, const Move& m, double d) const override;

    Solution crossover(const Solution& a, const Solution& b, Rng& rng) const override;

    std::size_t pheromone_size() const override { return n_ * n_; }
    Solution construct_aco(const std::vector<double>& tau, double alpha, double beta,
                           Rng& rng) const override;
    void deposit(const Solution& x, double amount, std::vector<double>& tau) const override;

    double best_known() const override { return optimum_; }
    const char* unit() const override { return "tour length"; }

    // Distance between two cities. Either a lookup in the precomputed matrix
    // or a square root over the coordinates, chosen by set_distance_mode.
    // The matrix is the obvious optimisation and it is the wrong one above a
    // few hundred cities: the matrix is n^2 doubles, so it leaves cache while
    // the two coordinate arrays stay in it, and a cache miss costs more than
    // the square root it was meant to save. The default is chosen by size and
    // the crossover point is measured, not guessed.
    double distance(int a, int b) const {
        const std::size_t i = static_cast<std::size_t>(a);
        const std::size_t j = static_cast<std::size_t>(b);
        if (use_matrix_) return dist_[i * n_ + j];
        const double dx = xs_[i] - xs_[j];
        const double dy = ys_[i] - ys_[j];
        return std::sqrt(dx * dx + dy * dy);
    }
    void set_distance_mode(bool use_matrix) { use_matrix_ = use_matrix; }
    bool uses_matrix() const { return use_matrix_; }
    const std::vector<double>& xs() const { return xs_; }
    const std::vector<double>& ys() const { return ys_; }
    // The `candidates` nearest cities to each city, nearest first. Used by the
    // ant construction, which is otherwise quadratic in the number of cities
    // and dominates every other cost in the framework.
    std::size_t candidates() const { return candidates_; }
    // Largest segment a single 2-opt move may reverse. See the comment on
    // random_move for why it is bounded and what happens when it is not.
    std::size_t span() const { return span_; }
    void set_span(std::size_t span) { span_ = std::max<std::size_t>(1, span); }
    // Turn the neighbour list move off, leaving only uniformly sampled 2-opt.
    // Exists so the benchmark can measure the difference rather than assert it.
    void set_candidate_moves(bool enabled) { candidate_moves_ = enabled; }
    bool candidate_moves() const { return candidate_moves_; }

private:
    std::size_t n_;
    std::vector<double> xs_, ys_;
    std::vector<double> dist_;
    std::vector<int> near_;      // n_ * candidates_, row major
    std::size_t candidates_ = 0;
    std::size_t span_ = 1;
    bool use_matrix_ = true;
    bool candidate_moves_ = true;
    std::string label_;
    double optimum_;
};

// n points equally spaced on a circle. The optimum is known in closed form:
// for points in convex position the optimal tour visits them in hull order,
// because any tour with crossing edges is shortened by uncrossing them
// (triangle inequality), so the optimum is n * 2 * R * sin(pi / n).
TspProblem make_circle_tsp(int n, double radius = 100.0);

// A k by k unit grid, k even. The optimum is known exactly and equals
// k * k * spacing: no edge of any tour can be shorter than the spacing, so
// that value is a lower bound, and a tour achieving it exists (the test
// constructs one). Unlike the circle instance this one is not solved by the
// nearest neighbour construction, which walks into a corner and has to jump,
// so it is the instance the demo and the benchmarks use.
TspProblem make_grid_tsp(int k, double spacing = 10.0);
// Uniform points in a square. No known optimum, and the class reports NaN
// rather than a guess.
TspProblem make_uniform_tsp(int n, std::uint64_t seed, double side = 1000.0);

// ---------------------------------------------------------------------------
// 0/1 knapsack. Maximisation, so evaluate() returns the negated value and
// display() undoes the sign.
// ---------------------------------------------------------------------------
class KnapsackProblem final : public Problem {
public:
    KnapsackProblem(std::vector<double> weights, std::vector<double> values,
                    double capacity, std::string label, double optimum_value = -1.0);

    std::string name() const override { return label_; }
    std::size_t size() const override { return n_; }

    double evaluate(const Solution& x) const override;   // branchless, vectorises
    double evaluate_branchy(const Solution& x) const;    // the obvious form, for comparison
    double aux_of(const Solution& x) const override;     // total weight
    bool feasible(const Solution& x) const override;

    Solution construct(Rng& rng) const override;         // randomised greedy by ratio
    Solution random_solution(Rng& rng) const override;

    Move random_move(const Candidate& c, Rng& rng) const override;
    double delta(const Candidate& c, const Move& m) const override;
    void apply(Candidate& c, const Move& m, double d) const override;

    Solution crossover(const Solution& a, const Solution& b, Rng& rng) const override;

    std::size_t pheromone_size() const override { return n_; }
    Solution construct_aco(const std::vector<double>& tau, double alpha, double beta,
                           Rng& rng) const override;
    void deposit(const Solution& x, double amount, std::vector<double>& tau) const override;

    double best_known() const override { return optimum_; }
    double display(double cost) const override { return -cost; }
    const char* unit() const override { return "packed value"; }

    double capacity() const { return capacity_; }
    const std::vector<double>& weights() const { return w_; }
    const std::vector<double>& values() const { return v_; }

    // Exact optimum by dynamic programming over integral weights. Only called
    // by the generator, and only while the table fits in memory.
    static double dp_optimum(const std::vector<double>& w, const std::vector<double>& v,
                             double capacity);

private:
    void repair(Solution& x) const;

    std::size_t n_;
    std::vector<double> w_, v_;
    std::vector<double> ratio_;
    std::vector<int> by_ratio_;
    double capacity_;
    std::string label_;
    double optimum_;
};

// Uncorrelated instance with integral weights, so the exact optimum can be
// computed by dynamic programming and used as the reference value.
KnapsackProblem make_random_knapsack(int n, std::uint64_t seed, int max_weight = 100,
                                     double capacity_fraction = 0.5);

// ---------------------------------------------------------------------------
// Graph colouring with a fixed palette. Minimises the number of edges whose
// endpoints share a colour, so a proper colouring has cost zero.
// ---------------------------------------------------------------------------
class ColoringProblem final : public Problem {
public:
    ColoringProblem(int vertices, int colours, std::vector<std::pair<int, int>> edges,
                    std::string label, double optimum_conflicts = -1.0);

    std::string name() const override { return label_; }
    std::size_t size() const override { return n_; }
    int colours() const { return k_; }

    double evaluate(const Solution& x) const override;
    bool feasible(const Solution& x) const override;

    Solution construct(Rng& rng) const override;   // greedy over a random vertex order
    Solution random_solution(Rng& rng) const override;

    Move random_move(const Candidate& c, Rng& rng) const override;
    double delta(const Candidate& c, const Move& m) const override;
    void apply(Candidate& c, const Move& m, double d) const override;

    Solution crossover(const Solution& a, const Solution& b, Rng& rng) const override;

    std::size_t pheromone_size() const override { return n_ * static_cast<std::size_t>(k_); }
    Solution construct_aco(const std::vector<double>& tau, double alpha, double beta,
                           Rng& rng) const override;
    void deposit(const Solution& x, double amount, std::vector<double>& tau) const override;

    double best_known() const override { return optimum_; }
    const char* unit() const override { return "conflicting edges"; }

    const std::vector<int>& neighbours_of(int v) const { return adj_[static_cast<std::size_t>(v)]; }
    std::size_t edge_count() const { return edges_.size(); }

private:
    int conflicts_at(const Solution& x, int v, int colour) const;

    std::size_t n_;
    int k_;
    std::vector<std::pair<int, int>> edges_;
    std::vector<std::vector<int>> adj_;
    std::string label_;
    double optimum_;
};

// Random k-partite graph: vertices are assigned to k classes and edges are
// drawn only between classes, so a colouring with zero conflicts exists by
// construction and the optimum is zero. A k-clique is planted so that fewer
// than k colours cannot succeed.
// `planted`, when given, receives the colouring the generator used. The tests
// need it: without the witness there is no way to check from outside that the
// instance really is k-colourable, and an instance whose stated optimum of
// zero is unreachable would make every gap in the results wrong.
ColoringProblem make_planted_coloring(int vertices, int colours, double density,
                                      std::uint64_t seed, Solution* planted = nullptr);

}  // namespace anneal
