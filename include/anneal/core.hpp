// The two contracts the whole framework is built around: a problem and a
// metaheuristic. Everything else in the library depends on these and on
// nothing else, which is what lets the problem, the algorithm and the
// execution scheme vary independently.
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "anneal/rng.hpp"

namespace anneal {

// One representation serves all three problems. For the travelling salesman
// problem the vector is a permutation of the cities, for the knapsack it is a
// 0/1 indicator over the items, and for graph colouring it is a colour index
// per vertex. Keeping a single contiguous type means a move, a crossover and a
// migration are the same memory operation everywhere, and it keeps the
// interface free of templates.
using Solution = std::vector<int>;

// Everything is minimised. A problem whose natural form is a maximisation
// negates in evaluate() and undoes the sign in display().
inline constexpr double kInfinity = std::numeric_limits<double>::infinity();

// A move is two indices; each problem decides what they mean. TSP: reverse the
// segment between positions i and j (2-opt). Knapsack: flip item i, and if
// j >= 0 flip item j too, which is a swap move and escapes the local optimum
// that single flips leave you in once the sack is full. Colouring: give
// vertex i the colour j.
struct Move {
    int i = 0;
    int j = 0;
};

// A solution together with the cached state that makes incremental evaluation
// and move generation cheap.
//
// `aux` is one scalar: without it the knapsack delta would have to re-sum the
// weights, which is linear and would defeat the point of having a delta.
// `index` is an integer array whose meaning is the problem's own: the
// travelling salesman problem keeps the inverse permutation there, which is
// what lets it propose a move by naming a city rather than a tour position.
// Both are maintained by apply() and rebuilt from scratch by Problem::make().
struct Candidate {
    Solution x;
    double cost = kInfinity;
    double aux = 0.0;
    std::vector<int> index;
};

class Problem {
public:
    virtual ~Problem() = default;

    virtual std::string name() const = 0;
    virtual std::size_t size() const = 0;

    // Full evaluation, used to seed a Candidate and to check the deltas.
    virtual double evaluate(const Solution& x) const = 0;
    // Problem specific cached scalar, recomputed from scratch.
    virtual double aux_of(const Solution&) const { return 0.0; }
    // Problem specific cached index, recomputed from scratch. A problem that
    // needs none leaves the vector empty, which costs nothing.
    virtual void build_index(const Solution&, std::vector<int>& out) const { out.clear(); }
    virtual bool feasible(const Solution& x) const = 0;

    // Construction heuristic: randomised greedy, the starting point every
    // algorithm uses unless a test asks for a purely random one.
    virtual Solution construct(Rng& rng) const = 0;
    virtual Solution random_solution(Rng& rng) const = 0;

    // Neighbourhood.
    virtual Move random_move(const Candidate& c, Rng& rng) const = 0;
    virtual double delta(const Candidate& c, const Move& m) const = 0;
    virtual void apply(Candidate& c, const Move& m, double d) const = 0;

    // Attribute of a move, for the tabu list. The default treats the index
    // pair itself as the attribute, which is the right granularity for all
    // three problems here: it forbids undoing the exact move just made.
    virtual std::uint64_t move_key(const Move& m) const {
        return static_cast<std::uint64_t>(m.i) * 1000003ULL +
               static_cast<std::uint64_t>(m.j + 1);
    }

    // Recombination for the genetic algorithm. Each problem owns this because
    // a uniform crossover would destroy a permutation, and an order crossover
    // is meaningless for a bit vector.
    virtual Solution crossover(const Solution& a, const Solution& b, Rng& rng) const = 0;

    // Ant colony support. The algorithm owns the pheromone dynamics, meaning
    // evaporation, deposit amount and the choice of which ant deposits; the
    // problem owns what a pheromone entry means and how a trail is walked.
    virtual std::size_t pheromone_size() const = 0;
    virtual Solution construct_aco(const std::vector<double>& tau, double alpha,
                                   double beta, Rng& rng) const = 0;
    virtual void deposit(const Solution& x, double amount,
                         std::vector<double>& tau) const = 0;

    // Known optimum in internal (minimised) units, or NaN when unknown.
    virtual double best_known() const { return std::numeric_limits<double>::quiet_NaN(); }
    // Internal cost converted back to the problem's natural units.
    virtual double display(double cost) const { return cost; }
    virtual const char* unit() const { return "cost"; }

    Candidate make(const Solution& x) const {
        Candidate c;
        c.x = x;
        c.cost = evaluate(c.x);
        c.aux = aux_of(c.x);
        build_index(c.x, c.index);
        return c;
    }
};

// A metaheuristic is an object with state that takes steps, not a function
// that returns an answer. The parallel schemes need to interrupt a search,
// read its best solution and hand it a better one from elsewhere, and none of
// that is expressible if the search is a single call that runs to completion.
class Metaheuristic {
public:
    virtual ~Metaheuristic() = default;

    virtual std::string name() const = 0;
    // One iteration. What an iteration is differs per algorithm: one move for
    // annealing, one generation for the genetic algorithm, one colony sweep
    // for ant colony optimisation. That is exactly why the budget is counted
    // in objective evaluations and wall clock, never in steps.
    virtual void step() = 0;
    virtual const Solution& best() const = 0;
    virtual double best_cost() const = 0;
    // Accept a solution found elsewhere. Returns true if it was adopted.
    virtual bool inject(const Solution& x, double cost) = 0;
    virtual std::uint64_t evaluations() const = 0;
};

using MetaheuristicPtr = std::unique_ptr<Metaheuristic>;

}  // namespace anneal
