#include "anneal/problems.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace anneal {

KnapsackProblem::KnapsackProblem(std::vector<double> weights, std::vector<double> values,
                                 double capacity, std::string label, double optimum_value)
    : n_(weights.size()),
      w_(std::move(weights)),
      v_(std::move(values)),
      capacity_(capacity),
      label_(std::move(label)),
      optimum_(optimum_value > 0.0 ? -optimum_value
                                   : std::numeric_limits<double>::quiet_NaN()) {
    ratio_.resize(n_);
    for (std::size_t i = 0; i < n_; ++i) {
        ratio_[i] = v_[i] / (w_[i] > 0.0 ? w_[i] : 1e-9);
    }
    by_ratio_.resize(n_);
    std::iota(by_ratio_.begin(), by_ratio_.end(), 0);
    std::sort(by_ratio_.begin(), by_ratio_.end(), [this](int a, int b) {
        return ratio_[static_cast<std::size_t>(a)] > ratio_[static_cast<std::size_t>(b)];
    });
}

double KnapsackProblem::evaluate(const Solution& x) const {
    // Branchless multiply-accumulate over two contiguous arrays. This is the
    // one objective in the project with no data dependent control flow and no
    // gather, so it is the one that can vectorise. Whether it actually does is
    // measured, not assumed; see the benchmark named knapsack-objective.
    double total = 0.0;
    for (std::size_t i = 0; i < n_; ++i) {
        total += v_[i] * static_cast<double>(x[i]);
    }
    return -total;
}

double KnapsackProblem::evaluate_branchy(const Solution& x) const {
    // The form most people write first. Kept so the difference between a
    // branch and a multiply by zero can be measured rather than argued about.
    double total = 0.0;
    for (std::size_t i = 0; i < n_; ++i) {
        if (x[i] != 0) total += v_[i];
    }
    return -total;
}

double KnapsackProblem::aux_of(const Solution& x) const {
    double total = 0.0;
    for (std::size_t i = 0; i < n_; ++i) {
        total += w_[i] * static_cast<double>(x[i]);
    }
    return total;
}

bool KnapsackProblem::feasible(const Solution& x) const {
    if (x.size() != n_) return false;
    for (int b : x) {
        if (b != 0 && b != 1) return false;
    }
    return aux_of(x) <= capacity_ + 1e-9;
}

Solution KnapsackProblem::construct(Rng& rng) const {
    // Greedy by value density, with each item skipped with a small
    // probability. Pure greedy would hand every worker the same solution.
    Solution x(n_, 0);
    double load = 0.0;
    for (int idx : by_ratio_) {
        const std::size_t i = static_cast<std::size_t>(idx);
        if (rng.uniform() < 0.1) continue;
        if (load + w_[i] <= capacity_) {
            x[i] = 1;
            load += w_[i];
        }
    }
    return x;
}

Solution KnapsackProblem::random_solution(Rng& rng) const {
    Solution x(n_, 0);
    for (std::size_t i = 0; i < n_; ++i) {
        x[i] = rng.uniform() < 0.5 ? 1 : 0;
    }
    repair(x);
    return x;
}

void KnapsackProblem::repair(Solution& x) const {
    double load = aux_of(x);
    if (load <= capacity_) return;
    // Drop the least dense items first. This is the standard repair and it is
    // what keeps every solution in the search feasible, so no penalty term is
    // needed in the objective.
    for (std::size_t k = n_; k > 0 && load > capacity_; --k) {
        const std::size_t i = static_cast<std::size_t>(by_ratio_[k - 1]);
        if (x[i] == 1) {
            x[i] = 0;
            load -= w_[i];
        }
    }
}

Move KnapsackProblem::random_move(const Candidate& c, Rng& rng) const {
    // Half the moves flip one item, half swap one in for one out. Single
    // flips alone stall once the sack is full: every insertion is infeasible
    // and every removal is a loss, so the search stops moving.
    Move m;
    m.i = static_cast<int>(rng.below(n_));
    if (n_ > 1 && rng.uniform() < 0.5) {
        int j = static_cast<int>(rng.below(n_ - 1));
        if (j >= m.i) ++j;
        m.j = j;
    } else {
        m.j = -1;
    }
    (void)c;
    return m;
}

double KnapsackProblem::delta(const Candidate& c, const Move& m) const {
    const std::size_t i = static_cast<std::size_t>(m.i);
    const double si = c.x[i] != 0 ? -1.0 : 1.0;
    double dw = si * w_[i];
    double dv = si * v_[i];
    if (m.j >= 0) {
        const std::size_t j = static_cast<std::size_t>(m.j);
        const double sj = c.x[j] != 0 ? -1.0 : 1.0;
        dw += sj * w_[j];
        dv += sj * v_[j];
    }
    if (c.aux + dw > capacity_ + 1e-9) return kInfinity;  // infeasible, rejected
    return -dv;  // internal cost is the negated packed value
}

void KnapsackProblem::apply(Candidate& c, const Move& m, double d) const {
    const std::size_t i = static_cast<std::size_t>(m.i);
    c.aux += (c.x[i] != 0 ? -w_[i] : w_[i]);
    c.x[i] ^= 1;
    if (m.j >= 0) {
        const std::size_t j = static_cast<std::size_t>(m.j);
        c.aux += (c.x[j] != 0 ? -w_[j] : w_[j]);
        c.x[j] ^= 1;
    }
    c.cost += d;
}

Solution KnapsackProblem::crossover(const Solution& a, const Solution& b, Rng& rng) const {
    Solution child(n_, 0);
    for (std::size_t i = 0; i < n_; ++i) {
        child[i] = rng.uniform() < 0.5 ? a[i] : b[i];
    }
    repair(child);
    return child;
}

Solution KnapsackProblem::construct_aco(const std::vector<double>& tau, double alpha,
                                        double beta, Rng& rng) const {
    Solution x(n_, 0);
    double load = 0.0;
    std::vector<int> candidates;
    std::vector<double> weight;
    candidates.reserve(n_);
    weight.reserve(n_);
    for (;;) {
        candidates.clear();
        weight.clear();
        double total = 0.0;
        for (std::size_t i = 0; i < n_; ++i) {
            if (x[i] != 0 || load + w_[i] > capacity_) continue;
            const double p = std::pow(tau[i], alpha) * std::pow(ratio_[i], beta);
            candidates.push_back(static_cast<int>(i));
            weight.push_back(p);
            total += p;
        }
        if (candidates.empty()) break;
        std::size_t pick = candidates.size() - 1;
        if (total > 0.0) {
            double r = rng.uniform() * total;
            for (std::size_t k = 0; k < candidates.size(); ++k) {
                r -= weight[k];
                if (r <= 0.0) { pick = k; break; }
            }
        } else {
            pick = static_cast<std::size_t>(rng.below(candidates.size()));
        }
        const std::size_t chosen = static_cast<std::size_t>(candidates[pick]);
        x[chosen] = 1;
        load += w_[chosen];
    }
    return x;
}

void KnapsackProblem::deposit(const Solution& x, double amount,
                              std::vector<double>& tau) const {
    for (std::size_t i = 0; i < n_; ++i) {
        if (x[i] != 0) tau[i] += amount;
    }
}

double KnapsackProblem::dp_optimum(const std::vector<double>& w, const std::vector<double>& v,
                                   double capacity) {
    // Exact, by the textbook table over integral capacities. Only usable
    // because the generator emits integral weights and a modest capacity; the
    // caller is responsible for not asking for a table that will not fit.
    const std::size_t cap = static_cast<std::size_t>(capacity);
    std::vector<double> table(cap + 1, 0.0);
    for (std::size_t i = 0; i < w.size(); ++i) {
        const std::size_t wi = static_cast<std::size_t>(w[i]);
        if (wi > cap) continue;
        for (std::size_t c = cap; c >= wi; --c) {
            const double with_item = table[c - wi] + v[i];
            if (with_item > table[c]) table[c] = with_item;
            if (c == wi) break;  // std::size_t would wrap
        }
    }
    return table[cap];
}

KnapsackProblem make_random_knapsack(int n, std::uint64_t seed, int max_weight,
                                     double capacity_fraction) {
    Rng rng(seed);
    std::vector<double> w(static_cast<std::size_t>(n));
    std::vector<double> v(static_cast<std::size_t>(n));
    double total_weight = 0.0;
    for (int i = 0; i < n; ++i) {
        const double wi = static_cast<double>(rng.range(1, max_weight));
        // Uncorrelated instance: the value is drawn independently of the
        // weight, which is the harder of the two standard families because
        // the greedy density order carries less information.
        const double vi = static_cast<double>(rng.range(1, max_weight));
        w[static_cast<std::size_t>(i)] = wi;
        v[static_cast<std::size_t>(i)] = vi;
        total_weight += wi;
    }
    const double capacity = std::floor(total_weight * capacity_fraction);
    const double optimum = KnapsackProblem::dp_optimum(w, v, capacity);
    return KnapsackProblem(std::move(w), std::move(v), capacity,
                           "knapsack-" + std::to_string(n), optimum);
}

}  // namespace anneal
