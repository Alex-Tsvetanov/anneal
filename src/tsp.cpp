#include "anneal/problems.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace anneal {

namespace {
constexpr double kPi = 3.14159265358979323846;
}

TspProblem::TspProblem(std::vector<double> xs, std::vector<double> ys, std::string label,
                       double optimum_tour_length)
    : n_(xs.size()),
      xs_(std::move(xs)),
      ys_(std::move(ys)),
      label_(std::move(label)),
      optimum_(optimum_tour_length > 0.0 ? optimum_tour_length
                                         : std::numeric_limits<double>::quiet_NaN()) {
    // The full distance matrix costs n^2 doubles. It is always built, because
    // the candidate lists below are derived from it and the construction
    // heuristic scans it, but whether the search reads it on the hot path is a
    // separate decision taken by use_matrix_ further down.
    dist_.resize(n_ * n_);
    for (std::size_t i = 0; i < n_; ++i) {
        for (std::size_t j = 0; j < n_; ++j) {
            const double dx = xs_[i] - xs_[j];
            const double dy = ys_[i] - ys_[j];
            dist_[i * n_ + j] = std::sqrt(dx * dx + dy * dy);
        }
    }

    // Default span: unbounded. A bounded span was tried first, on the theory
    // that limiting the reversal keeps an iteration constant time. It is the
    // single largest quality effect measured in this project, and it goes the
    // other way: two million descent proposals from the same nearest
    // neighbour tour end at 27579 with the span bounded to fifty and at 24965
    // with it unbounded. Bounding the span forbids exactly the long range
    // repairs a nearest neighbour tour needs. The value stays a parameter so
    // the sweep in anneal_bench neighbourhood can be repeated.
    span_ = n_ > 1 ? n_ - 1 : 1;

    // Whether the hot path reads the matrix or recomputes from coordinates is
    // decided by size, and the threshold is measured rather than assumed. A
    // full tour evaluation through the matrix costs 0.84 ns per city at a
    // thousand cities, 1.07 at two thousand and 6.63 at four thousand, while
    // the coordinate form costs a flat 2.3 ns per city at every size, because
    // two arrays of n doubles stay in cache and a matrix of n^2 does not. The
    // matrix therefore wins until it stops fitting; the crossover on this
    // machine lies between a 32 MB matrix and a 128 MB one, so the threshold
    // is 32 MB. The mode stays switchable so the benchmark can show both.
    use_matrix_ = n_ * n_ * sizeof(double) <= (32u << 20);

    // Candidate lists: for each city, its sixteen nearest. Two things depend
    // on them. The ant construction would otherwise consider every unvisited
    // city at every step, which is quadratic per ant and was measured at two
    // hundred objective evaluations in a quarter of a second on a 256 city
    // instance, four orders of magnitude behind the trajectory methods. And
    // the 2-opt move uses them to name a short edge to install, without which
    // 98 per cent of proposals are uphill.
    candidates_ = std::min<std::size_t>(16, n_ > 0 ? n_ - 1 : 0);
    near_.resize(n_ * candidates_);
    std::vector<int> order(n_);
    for (std::size_t i = 0; i < n_; ++i) {
        std::iota(order.begin(), order.end(), 0);
        const double* row = &dist_[i * n_];
        std::partial_sort(order.begin(), order.begin() + static_cast<long>(candidates_ + 1),
                          order.end(), [row](int a, int b) {
                              return row[static_cast<std::size_t>(a)] < row[static_cast<std::size_t>(b)];
                          });
        std::size_t written = 0;
        for (int candidate : order) {
            if (static_cast<std::size_t>(candidate) == i) continue;
            near_[i * candidates_ + written] = candidate;
            if (++written == candidates_) break;
        }
    }
}

double TspProblem::evaluate(const Solution& x) const {
    double total = 0.0;
    const std::size_t n = x.size();
    for (std::size_t i = 0; i + 1 < n; ++i) total += distance(x[i], x[i + 1]);
    if (n > 1) total += distance(x[n - 1], x[0]);
    return total;
}

double TspProblem::evaluate_from_coords(const Solution& x) const {
    double total = 0.0;
    const std::size_t n = x.size();
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t a = static_cast<std::size_t>(x[i]);
        const std::size_t b = static_cast<std::size_t>(x[(i + 1) % n]);
        const double dx = xs_[a] - xs_[b];
        const double dy = ys_[a] - ys_[b];
        total += std::sqrt(dx * dx + dy * dy);
    }
    return total;
}

bool TspProblem::feasible(const Solution& x) const {
    if (x.size() != n_) return false;
    std::vector<char> seen(n_, 0);
    for (int city : x) {
        if (city < 0 || static_cast<std::size_t>(city) >= n_) return false;
        if (seen[static_cast<std::size_t>(city)]) return false;
        seen[static_cast<std::size_t>(city)] = 1;
    }
    return true;
}

Solution TspProblem::random_solution(Rng& rng) const {
    Solution x(n_);
    std::iota(x.begin(), x.end(), 0);
    for (std::size_t i = n_; i > 1; --i) {
        const std::size_t j = static_cast<std::size_t>(rng.below(i));
        std::swap(x[i - 1], x[j]);
    }
    return x;
}

Solution TspProblem::construct(Rng& rng) const {
    // Nearest neighbour from a random start. Randomising the start is what
    // makes independent multi-start meaningful: a deterministic construction
    // would give every worker the same starting point.
    Solution x;
    x.reserve(n_);
    std::vector<char> used(n_, 0);
    int current = static_cast<int>(rng.below(n_));
    x.push_back(current);
    used[static_cast<std::size_t>(current)] = 1;
    for (std::size_t step = 1; step < n_; ++step) {
        const double* row = &dist_[static_cast<std::size_t>(current) * n_];
        int best = -1;
        double best_d = kInfinity;
        for (std::size_t j = 0; j < n_; ++j) {
            if (used[j]) continue;
            if (row[j] < best_d) {
                best_d = row[j];
                best = static_cast<int>(j);
            }
        }
        current = best;
        used[static_cast<std::size_t>(current)] = 1;
        x.push_back(current);
    }
    return x;
}

void TspProblem::build_index(const Solution& x, std::vector<int>& out) const {
    out.assign(n_, 0);
    for (std::size_t position = 0; position < x.size(); ++position) {
        out[static_cast<std::size_t>(x[position])] = static_cast<int>(position);
    }
}

Move TspProblem::random_move(const Candidate& c, Rng& rng) const {
    // 2-opt: reverse the tour segment between two positions. How those two
    // positions are chosen is the single most consequential decision in this
    // file, so both forms are here and the weaker one is the fallback.
    const std::size_t n = c.x.size();
    Move m;

    // Preferred form: name a city, pick one of its sixteen nearest
    // neighbours, and let the inverse permutation say where that neighbour
    // sits in the tour. The move then always installs a short edge.
    //
    // Measured effect, from anneal_bench neighbourhood: it raises the share
    // of proposals that improve the tour from 0.17 per cent to 3.3 per cent,
    // a factor of twenty, but at two million proposals the descent ends in
    // much the same place either way, 24965 against 25087. Its real value is
    // elsewhere. It shrinks the distribution of uphill moves, whose median
    // falls from 355 to 82, and simulated annealing calibrates its initial
    // temperature from exactly that distribution, so with uniform sampling
    // the chain starts four times too hot and never beats its own
    // construction inside the budget.
    if (candidate_moves_ && candidates_ > 0 && c.index.size() == n_) {
        const std::size_t from = static_cast<std::size_t>(rng.below(n));
        const std::size_t city = static_cast<std::size_t>(c.x[from]);
        const std::size_t pick = static_cast<std::size_t>(rng.below(candidates_));
        const std::size_t neighbour = static_cast<std::size_t>(near_[city * candidates_ + pick]);
        const std::size_t to = static_cast<std::size_t>(c.index[neighbour]);
        std::size_t lo = std::min(from, to);
        std::size_t hi = std::max(from, to);
        if (lo != hi && hi - lo <= span_ && !(lo == 0 && hi == n - 1)) {
            m.i = static_cast<int>(lo);
            m.j = static_cast<int>(hi);
            return m;
        }
        // The neighbour is too far away along the tour to reverse cheaply, or
        // is already adjacent. Falling through to the bounded random move
        // keeps the neighbourhood connected, so no tour is unreachable.
    }

    m.i = static_cast<int>(rng.below(n - 1));
    const std::size_t remaining = n - 1 - static_cast<std::size_t>(m.i);
    const std::size_t reach = std::min(span_, remaining);
    m.j = m.i + 1 + static_cast<int>(rng.below(reach));
    if (m.i == 0 && m.j == static_cast<int>(n) - 1) m.j = static_cast<int>(n) - 2;
    return m;
}

double TspProblem::delta(const Candidate& c, const Move& m) const {
    const Solution& t = c.x;
    const std::size_t n = t.size();
    const std::size_t i = static_cast<std::size_t>(m.i);
    const std::size_t j = static_cast<std::size_t>(m.j);
    if (i >= j || j >= n) return 0.0;
    const std::size_t jn = (j + 1) % n;
    if (jn == i) return 0.0;
    const int a = t[i];
    const int b = t[i + 1];
    const int c2 = t[j];
    const int d2 = t[jn];
    // Removing edges (a,b) and (c2,d2), adding (a,c2) and (b,d2). Four table
    // lookups regardless of the segment length, against O(n) for a full
    // re-evaluation.
    return distance(a, c2) + distance(b, d2) - distance(a, b) - distance(c2, d2);
}

void TspProblem::apply(Candidate& c, const Move& m, double d) const {
    const auto first = c.x.begin() + m.i + 1;
    const auto last = c.x.begin() + m.j + 1;
    std::reverse(first, last);
    // The inverse permutation only changes inside the reversed segment, so
    // maintaining it costs the same order as the reversal itself.
    if (c.index.size() == n_) {
        for (int position = m.i + 1; position <= m.j; ++position) {
            c.index[static_cast<std::size_t>(c.x[static_cast<std::size_t>(position)])] = position;
        }
    }
    c.cost += d;
}

Solution TspProblem::crossover(const Solution& a, const Solution& b, Rng& rng) const {
    // Order crossover (OX). A segment of the first parent is copied in place,
    // and the remaining cities are filled in the order they appear in the
    // second parent. The result is a permutation by construction, which is why
    // recombination belongs to the problem and not to the algorithm.
    const std::size_t n = a.size();
    if (n < 3) return a;
    std::size_t lo = static_cast<std::size_t>(rng.below(n));
    std::size_t hi = static_cast<std::size_t>(rng.below(n));
    if (lo > hi) std::swap(lo, hi);
    Solution child(n, -1);
    std::vector<char> taken(n_, 0);
    for (std::size_t i = lo; i <= hi; ++i) {
        child[i] = a[i];
        taken[static_cast<std::size_t>(a[i])] = 1;
    }
    std::size_t write = (hi + 1) % n;
    for (std::size_t k = 0; k < n; ++k) {
        const int city = b[(hi + 1 + k) % n];
        if (taken[static_cast<std::size_t>(city)]) continue;
        child[write] = city;
        taken[static_cast<std::size_t>(city)] = 1;
        write = (write + 1) % n;
    }
    return child;
}

namespace {
// std::pow with a run time exponent costs about twenty nanoseconds and sits in
// the innermost loop of the ant construction. The exponents actually used are
// small non-negative integers, so the integer case is separated out.
double fast_pow(double base, double exponent, int integer_exponent, bool is_integer) {
    if (!is_integer) return std::pow(base, exponent);
    double result = 1.0;
    for (int i = 0; i < integer_exponent; ++i) result *= base;
    return result;
}
}  // namespace

Solution TspProblem::construct_aco(const std::vector<double>& tau, double alpha, double beta,
                                   Rng& rng) const {
    const int alpha_int = static_cast<int>(alpha);
    const bool alpha_is_int = alpha == static_cast<double>(alpha_int) && alpha_int >= 0 &&
                              alpha_int <= 8;
    const int beta_int = static_cast<int>(beta);
    const bool beta_is_int = beta == static_cast<double>(beta_int) && beta_int >= 0 &&
                             beta_int <= 8;

    Solution x;
    x.reserve(n_);
    std::vector<char> used(n_, 0);
    std::vector<double> weight(candidates_, 0.0);
    std::vector<int> choice(candidates_, 0);
    int current = static_cast<int>(rng.below(n_));
    x.push_back(current);
    used[static_cast<std::size_t>(current)] = 1;

    for (std::size_t step = 1; step < n_; ++step) {
        const std::size_t base = static_cast<std::size_t>(current) * n_;
        const int* near = &near_[static_cast<std::size_t>(current) * candidates_];
        std::size_t count = 0;
        double total = 0.0;
        for (std::size_t c = 0; c < candidates_; ++c) {
            const std::size_t j = static_cast<std::size_t>(near[c]);
            if (used[j]) continue;
            const double eta = 1.0 / (dist_[base + j] + 1e-9);
            const double w = fast_pow(tau[base + j], alpha, alpha_int, alpha_is_int) *
                             fast_pow(eta, beta, beta_int, beta_is_int);
            weight[count] = w;
            choice[count] = static_cast<int>(j);
            total += w;
            ++count;
        }
        int chosen = -1;
        if (count > 0 && total > 0.0) {
            double r = rng.uniform() * total;
            for (std::size_t c = 0; c < count; ++c) {
                r -= weight[c];
                if (r <= 0.0) {
                    chosen = choice[c];
                    break;
                }
            }
            if (chosen < 0) chosen = choice[count - 1];
        } else {
            // Every near neighbour is already on the tour. This happens only
            // in the tail of the construction, so the linear fallback is paid
            // rarely; taking the nearest unvisited city keeps the fallback
            // from undoing the quality the candidate list bought.
            double best = kInfinity;
            for (std::size_t j = 0; j < n_; ++j) {
                if (used[j]) continue;
                if (dist_[base + j] < best) {
                    best = dist_[base + j];
                    chosen = static_cast<int>(j);
                }
            }
        }
        current = chosen;
        used[static_cast<std::size_t>(current)] = 1;
        x.push_back(current);
    }
    return x;
}

void TspProblem::deposit(const Solution& x, double amount, std::vector<double>& tau) const {
    const std::size_t n = x.size();
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t a = static_cast<std::size_t>(x[i]);
        const std::size_t b = static_cast<std::size_t>(x[(i + 1) % n]);
        tau[a * n_ + b] += amount;
        tau[b * n_ + a] += amount;
    }
}

TspProblem make_circle_tsp(int n, double radius) {
    std::vector<double> xs(static_cast<std::size_t>(n));
    std::vector<double> ys(static_cast<std::size_t>(n));
    // Angular order first, then a fixed shuffle of the city indices. Without
    // the shuffle the identity permutation would already be optimal and every
    // algorithm would look brilliant for the wrong reason. The shuffle is
    // seeded from a constant so the instance itself stays reproducible.
    std::vector<int> order(static_cast<std::size_t>(n));
    std::iota(order.begin(), order.end(), 0);
    Rng shuffle_rng(0xC1C1E5EEDULL);
    for (std::size_t i = order.size(); i > 1; --i) {
        std::swap(order[i - 1], order[static_cast<std::size_t>(shuffle_rng.below(i))]);
    }
    for (int i = 0; i < n; ++i) {
        const int slot = order[static_cast<std::size_t>(i)];
        const double angle = 2.0 * kPi * static_cast<double>(slot) / static_cast<double>(n);
        xs[static_cast<std::size_t>(i)] = radius * std::cos(angle);
        ys[static_cast<std::size_t>(i)] = radius * std::sin(angle);
    }
    const double optimum = static_cast<double>(n) * 2.0 * radius * std::sin(kPi / n);
    return TspProblem(std::move(xs), std::move(ys), "circle-" + std::to_string(n), optimum);
}

TspProblem make_grid_tsp(int k, double spacing) {
    if (k % 2 != 0) {
        // An odd square grid graph is bipartite with unequal sides, so it has
        // no Hamiltonian cycle and the lower bound of n * spacing is not
        // attained. Rather than report an optimum that does not exist, the
        // generator refuses.
        throw std::invalid_argument("make_grid_tsp needs an even k for the optimum to be exact");
    }
    const std::size_t n = static_cast<std::size_t>(k) * static_cast<std::size_t>(k);
    std::vector<double> xs(n);
    std::vector<double> ys(n);
    for (int y = 0; y < k; ++y) {
        for (int x = 0; x < k; ++x) {
            const std::size_t index = static_cast<std::size_t>(y) * static_cast<std::size_t>(k) +
                                      static_cast<std::size_t>(x);
            xs[index] = x * spacing;
            ys[index] = y * spacing;
        }
    }
    // No shuffle is needed here: the identity permutation is the row major
    // scan, which pays a long jump at the end of every row and is far from
    // optimal.
    return TspProblem(std::move(xs), std::move(ys), "grid-" + std::to_string(k) + "x" +
                                                        std::to_string(k),
                      static_cast<double>(n) * spacing);
}

TspProblem make_uniform_tsp(int n, std::uint64_t seed, double side) {
    Rng rng(seed);
    std::vector<double> xs(static_cast<std::size_t>(n));
    std::vector<double> ys(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        xs[static_cast<std::size_t>(i)] = rng.uniform() * side;
        ys[static_cast<std::size_t>(i)] = rng.uniform() * side;
    }
    // No optimum is passed: the optimal tour of a random uniform instance is
    // not known, and a made up reference value would corrupt every gap in the
    // results.
    return TspProblem(std::move(xs), std::move(ys), "uniform-" + std::to_string(n));
}

}  // namespace anneal
