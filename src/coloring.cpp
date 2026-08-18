#include "anneal/problems.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace anneal {

ColoringProblem::ColoringProblem(int vertices, int colours,
                                 std::vector<std::pair<int, int>> edges, std::string label,
                                 double optimum_conflicts)
    : n_(static_cast<std::size_t>(vertices)),
      k_(colours),
      edges_(std::move(edges)),
      label_(std::move(label)),
      optimum_(optimum_conflicts >= 0.0 ? optimum_conflicts
                                        : std::numeric_limits<double>::quiet_NaN()) {
    adj_.resize(n_);
    for (const auto& e : edges_) {
        adj_[static_cast<std::size_t>(e.first)].push_back(e.second);
        adj_[static_cast<std::size_t>(e.second)].push_back(e.first);
    }
}

double ColoringProblem::evaluate(const Solution& x) const {
    std::size_t conflicts = 0;
    for (const auto& e : edges_) {
        if (x[static_cast<std::size_t>(e.first)] == x[static_cast<std::size_t>(e.second)]) {
            ++conflicts;
        }
    }
    return static_cast<double>(conflicts);
}

bool ColoringProblem::feasible(const Solution& x) const {
    if (x.size() != n_) return false;
    for (int c : x) {
        if (c < 0 || c >= k_) return false;
    }
    return true;
}

int ColoringProblem::conflicts_at(const Solution& x, int v, int colour) const {
    int count = 0;
    for (int u : adj_[static_cast<std::size_t>(v)]) {
        if (x[static_cast<std::size_t>(u)] == colour) ++count;
    }
    return count;
}

Solution ColoringProblem::random_solution(Rng& rng) const {
    Solution x(n_);
    for (std::size_t i = 0; i < n_; ++i) {
        x[i] = static_cast<int>(rng.below(static_cast<std::uint64_t>(k_)));
    }
    return x;
}

Solution ColoringProblem::construct(Rng& rng) const {
    // Greedy over a random vertex order: give each vertex the colour that
    // conflicts least with the neighbours already coloured, breaking ties by
    // the first such colour. On a graph that is k-colourable this often lands
    // on zero conflicts straight away for sparse instances, which is exactly
    // why the demo uses a density high enough that it does not.
    std::vector<int> order(n_);
    std::iota(order.begin(), order.end(), 0);
    for (std::size_t i = n_; i > 1; --i) {
        std::swap(order[i - 1], order[static_cast<std::size_t>(rng.below(i))]);
    }
    Solution x(n_, 0);
    std::vector<char> placed(n_, 0);
    std::vector<int> tally(static_cast<std::size_t>(k_), 0);
    for (int v : order) {
        std::fill(tally.begin(), tally.end(), 0);
        for (int u : adj_[static_cast<std::size_t>(v)]) {
            if (placed[static_cast<std::size_t>(u)]) {
                ++tally[static_cast<std::size_t>(x[static_cast<std::size_t>(u)])];
            }
        }
        int best = 0;
        for (int c = 1; c < k_; ++c) {
            if (tally[static_cast<std::size_t>(c)] < tally[static_cast<std::size_t>(best)]) best = c;
        }
        x[static_cast<std::size_t>(v)] = best;
        placed[static_cast<std::size_t>(v)] = 1;
    }
    return x;
}

Move ColoringProblem::random_move(const Candidate& c, Rng& rng) const {
    // Bias towards vertices that are actually in conflict. A uniform choice
    // wastes most iterations on vertices whose colour is already fine, and on
    // a near-solved instance almost every vertex is fine.
    Move m;
    m.i = static_cast<int>(rng.below(n_));
    if (c.cost > 0.0) {
        for (int attempt = 0; attempt < 8; ++attempt) {
            const int v = static_cast<int>(rng.below(n_));
            if (conflicts_at(c.x, v, c.x[static_cast<std::size_t>(v)]) > 0) {
                m.i = v;
                break;
            }
        }
    }
    int colour = static_cast<int>(rng.below(static_cast<std::uint64_t>(k_ - 1)));
    if (colour >= c.x[static_cast<std::size_t>(m.i)]) ++colour;
    m.j = colour;
    return m;
}

double ColoringProblem::delta(const Candidate& c, const Move& m) const {
    const int current = c.x[static_cast<std::size_t>(m.i)];
    if (current == m.j) return 0.0;
    // Only the edges incident to one vertex change status, so the cost of the
    // delta is the degree of that vertex rather than the number of edges.
    return static_cast<double>(conflicts_at(c.x, m.i, m.j) -
                               conflicts_at(c.x, m.i, current));
}

void ColoringProblem::apply(Candidate& c, const Move& m, double d) const {
    c.x[static_cast<std::size_t>(m.i)] = m.j;
    c.cost += d;
}

Solution ColoringProblem::crossover(const Solution& a, const Solution& b, Rng& rng) const {
    // Every colour vector is a valid solution, so a per-vertex uniform choice
    // needs no repair. The parents are picked whole per block rather than per
    // vertex so that a locally consistent region survives recombination.
    Solution child(n_);
    const std::size_t block = std::max<std::size_t>(1, n_ / 8);
    bool from_a = rng.uniform() < 0.5;
    for (std::size_t i = 0; i < n_; ++i) {
        if (i % block == 0) from_a = rng.uniform() < 0.5;
        child[i] = from_a ? a[i] : b[i];
    }
    return child;
}

Solution ColoringProblem::construct_aco(const std::vector<double>& tau, double alpha,
                                        double beta, Rng& rng) const {
    Solution x(n_, 0);
    std::vector<char> placed(n_, 0);
    std::vector<double> weight(static_cast<std::size_t>(k_), 0.0);
    std::vector<int> order(n_);
    std::iota(order.begin(), order.end(), 0);
    for (std::size_t i = n_; i > 1; --i) {
        std::swap(order[i - 1], order[static_cast<std::size_t>(rng.below(i))]);
    }
    for (int v : order) {
        const std::size_t base = static_cast<std::size_t>(v) * static_cast<std::size_t>(k_);
        double total = 0.0;
        for (int c = 0; c < k_; ++c) {
            int clash = 0;
            for (int u : adj_[static_cast<std::size_t>(v)]) {
                if (placed[static_cast<std::size_t>(u)] && x[static_cast<std::size_t>(u)] == c) ++clash;
            }
            const double eta = 1.0 / (1.0 + static_cast<double>(clash));
            weight[static_cast<std::size_t>(c)] =
                std::pow(tau[base + static_cast<std::size_t>(c)], alpha) * std::pow(eta, beta);
            total += weight[static_cast<std::size_t>(c)];
        }
        int chosen = 0;
        if (total > 0.0) {
            double r = rng.uniform() * total;
            for (int c = 0; c < k_; ++c) {
                r -= weight[static_cast<std::size_t>(c)];
                if (r <= 0.0) { chosen = c; break; }
            }
        } else {
            chosen = static_cast<int>(rng.below(static_cast<std::uint64_t>(k_)));
        }
        x[static_cast<std::size_t>(v)] = chosen;
        placed[static_cast<std::size_t>(v)] = 1;
    }
    return x;
}

void ColoringProblem::deposit(const Solution& x, double amount,
                              std::vector<double>& tau) const {
    for (std::size_t v = 0; v < n_; ++v) {
        tau[v * static_cast<std::size_t>(k_) + static_cast<std::size_t>(x[v])] += amount;
    }
}

ColoringProblem make_planted_coloring(int vertices, int colours, double density,
                                      std::uint64_t seed, Solution* planted) {
    Rng rng(seed);
    const std::size_t n = static_cast<std::size_t>(vertices);
    std::vector<int> klass(n);
    for (std::size_t i = 0; i < n; ++i) {
        klass[i] = static_cast<int>(rng.below(static_cast<std::uint64_t>(colours)));
    }
    // The first `colours` vertices are forced into distinct classes and joined
    // into a clique, so the instance genuinely needs all k colours and is not
    // accidentally solvable with fewer.
    for (int c = 0; c < colours && static_cast<std::size_t>(c) < n; ++c) {
        klass[static_cast<std::size_t>(c)] = c;
    }
    std::vector<std::pair<int, int>> edges;
    for (int a = 0; a < colours && static_cast<std::size_t>(a) < n; ++a) {
        for (int b = a + 1; b < colours && static_cast<std::size_t>(b) < n; ++b) {
            edges.emplace_back(a, b);
        }
    }
    for (std::size_t a = 0; a < n; ++a) {
        for (std::size_t b = a + 1; b < n; ++b) {
            if (a < static_cast<std::size_t>(colours) && b < static_cast<std::size_t>(colours)) continue;
            if (klass[a] == klass[b]) continue;  // never inside a class, so k colours suffice
            if (rng.uniform() < density) edges.emplace_back(static_cast<int>(a), static_cast<int>(b));
        }
    }
    if (planted != nullptr) *planted = klass;
    return ColoringProblem(vertices, colours, std::move(edges),
                           "coloring-" + std::to_string(vertices) + "-k" +
                               std::to_string(colours),
                           0.0);
}

}  // namespace anneal
