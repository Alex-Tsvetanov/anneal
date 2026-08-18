#include "anneal/algorithms.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

#include "anneal/problems.hpp"

namespace anneal {

double cooled(const AnnealingConfig& cfg, double t0, std::uint64_t steps) {
    const double k = static_cast<double>(steps);
    switch (cfg.schedule) {
        case Cooling::Geometric:
            return t0 * std::pow(cfg.alpha, k);
        case Cooling::Linear:
            // Constant decrement, sized so the first step matches the
            // geometric schedule. Without that tie the two schedules would
            // differ in starting slope as well as in shape, and a comparison
            // between them would not isolate the shape.
            return t0 - k * t0 * (1.0 - cfg.alpha);
        case Cooling::Logarithmic:
            // The Geman and Geman form. Slow enough to carry a convergence
            // proof and, as the results show, slow enough to be useless
            // inside a wall clock budget.
            return t0 / std::log2(2.0 + k);
    }
    return t0;
}

namespace {

// ---------------------------------------------------------------------------
// Simulated annealing.
// ---------------------------------------------------------------------------
class Annealing final : public Metaheuristic {
public:
    Annealing(const Problem& problem, std::uint64_t seed, AnnealingConfig cfg)
        : p_(problem), rng_(seed), cfg_(cfg) {
        current_ = p_.make(p_.construct(rng_));
        ++evaluations_;
        best_ = current_.x;
        best_cost_ = current_.cost;
        t0_ = calibrate();
        temperature_ = t0_;
    }

    std::string name() const override { return "annealing"; }

    void step() override {
        const Move m = p_.random_move(current_, rng_);
        const double d = p_.delta(current_, m);
        ++evaluations_;
        const double u = rng_.uniform();
        if (std::isfinite(d) && metropolis_accepts(d, temperature_, u)) {
            p_.apply(current_, m, d);
            if (current_.cost < best_cost_) {
                best_cost_ = current_.cost;
                best_ = current_.x;
            }
        }
        if (++in_chain_ >= cfg_.chain_length) {
            in_chain_ = 0;
            ++cooling_steps_;
            temperature_ = cooled(cfg_, t0_, cooling_steps_);
            if (temperature_ < cfg_.t_end) {
                if (cfg_.reheat) {
                    // Restarting the schedule rather than freezing. A frozen
                    // chain spends the rest of the budget rejecting every
                    // move, which shows up in the time to target as a long
                    // flat tail.
                    cooling_steps_ = 0;
                    temperature_ = t0_;
                } else {
                    temperature_ = cfg_.t_end;
                }
            }
        }
    }

    const Solution& best() const override { return best_; }
    double best_cost() const override { return best_cost_; }
    std::uint64_t evaluations() const override { return evaluations_; }

    bool inject(const Solution& x, double cost) override {
        if (!(cost < current_.cost)) return false;
        current_ = p_.make(x);
        ++evaluations_;
        if (current_.cost < best_cost_) {
            best_cost_ = current_.cost;
            best_ = current_.x;
        }
        return true;
    }

    double temperature() const { return temperature_; }

private:
    // The initial temperature is derived from the instance rather than given
    // as a constant. Sampling uphill moves and solving for the requested
    // acceptance rate is what lets one configuration serve three problems
    // whose objective values differ by four orders of magnitude.
    double calibrate() {
        Candidate probe = current_;
        std::vector<double> uphill;
        uphill.reserve(kCalibrationSamples);
        for (int i = 0; i < kCalibrationSamples; ++i) {
            const Move m = p_.random_move(probe, rng_);
            const double d = p_.delta(probe, m);
            if (std::isfinite(d) && d > 0.0) uphill.push_back(d);
        }
        evaluations_ += kCalibrationSamples;
        if (uphill.empty()) return 1.0;
        // The median, not the mean. The distribution of uphill moves has a
        // long right tail: on a thousand city instance the mean was 175 and
        // the median 82, so calibrating on the mean starts the chain twice as
        // hot as intended and the budget is spent walking back down.
        std::nth_element(uphill.begin(), uphill.begin() + uphill.size() / 2, uphill.end());
        const double typical = uphill[uphill.size() / 2];
        double target = cfg_.target_acceptance;
        if (!(target > 0.0 && target < 1.0)) target = 0.8;
        return -typical / std::log(target);
    }

    static constexpr int kCalibrationSamples = 400;

    const Problem& p_;
    Rng rng_;
    AnnealingConfig cfg_;
    Candidate current_;
    Solution best_;
    double best_cost_ = kInfinity;
    double t0_ = 1.0;
    double temperature_ = 1.0;
    std::uint64_t cooling_steps_ = 0;
    int in_chain_ = 0;
    std::uint64_t evaluations_ = 0;
};

// ---------------------------------------------------------------------------
// Tabu search.
// ---------------------------------------------------------------------------
class Tabu final : public Metaheuristic {
public:
    Tabu(const Problem& problem, std::uint64_t seed, TabuConfig cfg)
        : p_(problem), rng_(seed), cfg_(cfg), table_(cfg.slots) {
        current_ = p_.make(p_.construct(rng_));
        ++evaluations_;
        best_ = current_.x;
        best_cost_ = current_.cost;
    }

    std::string name() const override { return "tabu"; }

    void step() override {
        // Candidate list strategy: sample a fixed number of moves instead of
        // enumerating the neighbourhood. Full enumeration is quadratic for
        // 2-opt, so it would make one tabu iteration cost as much as several
        // thousand annealing iterations and the comparison at equal wall
        // clock would say more about the neighbourhood size than about the
        // method.
        double best_delta = kInfinity;
        Move best_move{};
        bool best_was_tabu = false;
        bool found = false;
        for (int k = 0; k < cfg_.candidate_list; ++k) {
            const Move m = p_.random_move(current_, rng_);
            const double d = p_.delta(current_, m);
            ++evaluations_;
            if (!std::isfinite(d)) continue;
            const bool tabu = table_.is_tabu(p_.move_key(m), iteration_);
            if (!tabu_admits(tabu, current_.cost + d, best_cost_)) continue;
            if (!found || d < best_delta) {
                best_delta = d;
                best_move = m;
                best_was_tabu = tabu;
                found = true;
            }
        }
        ++iteration_;
        if (!found) return;
        if (best_was_tabu) ++aspirations_;
        // Tabu search moves to the best admissible neighbour even when it is
        // worse than where it stands. That is the whole mechanism: the tabu
        // list is what stops it from walking straight back.
        table_.forbid(p_.move_key(best_move), iteration_, cfg_.tenure);
        p_.apply(current_, best_move, best_delta);
        if (current_.cost < best_cost_) {
            best_cost_ = current_.cost;
            best_ = current_.x;
        }
    }

    const Solution& best() const override { return best_; }
    double best_cost() const override { return best_cost_; }
    std::uint64_t evaluations() const override { return evaluations_; }

    bool inject(const Solution& x, double cost) override {
        if (!(cost < current_.cost)) return false;
        current_ = p_.make(x);
        ++evaluations_;
        if (current_.cost < best_cost_) {
            best_cost_ = current_.cost;
            best_ = current_.x;
        }
        return true;
    }

    std::uint64_t aspirations() const { return aspirations_; }

private:
    const Problem& p_;
    Rng rng_;
    TabuConfig cfg_;
    TabuTable table_;
    Candidate current_;
    Solution best_;
    double best_cost_ = kInfinity;
    std::uint64_t iteration_ = 0;
    std::uint64_t aspirations_ = 0;
    std::uint64_t evaluations_ = 0;
};

// ---------------------------------------------------------------------------
// Genetic algorithm.
// ---------------------------------------------------------------------------
class Genetic final : public Metaheuristic {
public:
    Genetic(const Problem& problem, std::uint64_t seed, GeneticConfig cfg)
        : p_(problem), rng_(seed), cfg_(cfg) {
        population_.reserve(static_cast<std::size_t>(cfg_.population));
        // Only an eighth of the population comes from the construction
        // heuristic. Two reasons, both measured rather than assumed: the
        // construction is quadratic in the instance size for the travelling
        // salesman problem, so forty of them cost tens of milliseconds of pure
        // setup and showed up as lost speedup in the thread scan; and forty
        // nearest neighbour tours from forty random starts are nearly the same
        // tour, so the population began with almost no diversity for
        // recombination to work with.
        const int seeded = std::max(1, cfg_.population / 8);
        for (int i = 0; i < cfg_.population; ++i) {
            population_.push_back(
                p_.make(i < seeded ? p_.construct(rng_) : p_.random_solution(rng_)));
            ++evaluations_;
        }
        sort_population();
        best_ = population_.front().x;
        best_cost_ = population_.front().cost;
    }

    std::string name() const override { return "genetic"; }

    void step() override {
        std::vector<Candidate> next;
        next.reserve(population_.size());
        const int elite = std::min(cfg_.elite, cfg_.population);
        for (int i = 0; i < elite; ++i) next.push_back(population_[static_cast<std::size_t>(i)]);
        while (static_cast<int>(next.size()) < cfg_.population) {
            const Candidate& a = tournament();
            const Candidate& b = tournament();
            Solution child = rng_.uniform() < cfg_.crossover_rate
                                 ? p_.crossover(a.x, b.x, rng_)
                                 : a.x;
            Candidate c = p_.make(child);
            ++evaluations_;
            if (rng_.uniform() < cfg_.mutation_rate) {
                for (int k = 0; k < cfg_.mutation_moves; ++k) {
                    const Move m = p_.random_move(c, rng_);
                    const double d = p_.delta(c, m);
                    ++evaluations_;
                    if (std::isfinite(d)) p_.apply(c, m, d);
                }
            }
            next.push_back(std::move(c));
        }
        population_ = std::move(next);
        sort_population();
        if (population_.front().cost < best_cost_) {
            best_cost_ = population_.front().cost;
            best_ = population_.front().x;
        }
    }

    const Solution& best() const override { return best_; }
    double best_cost() const override { return best_cost_; }
    std::uint64_t evaluations() const override { return evaluations_; }

    bool inject(const Solution& x, double cost) override {
        // An immigrant replaces the worst member, never the best. Replacing
        // at random loses elites; replacing the whole population collapses
        // diversity, which is the one thing a population method has that the
        // trajectory methods do not.
        if (population_.empty()) return false;
        if (!(cost < population_.back().cost)) return false;
        population_.back() = p_.make(x);
        ++evaluations_;
        sort_population();
        if (population_.front().cost < best_cost_) {
            best_cost_ = population_.front().cost;
            best_ = population_.front().x;
        }
        return true;
    }

private:
    void sort_population() {
        std::sort(population_.begin(), population_.end(),
                  [](const Candidate& a, const Candidate& b) { return a.cost < b.cost; });
    }

    const Candidate& tournament() {
        std::size_t best = static_cast<std::size_t>(rng_.below(population_.size()));
        for (int i = 1; i < cfg_.tournament; ++i) {
            const std::size_t challenger = static_cast<std::size_t>(rng_.below(population_.size()));
            if (population_[challenger].cost < population_[best].cost) best = challenger;
        }
        return population_[best];
    }

    const Problem& p_;
    Rng rng_;
    GeneticConfig cfg_;
    std::vector<Candidate> population_;
    Solution best_;
    double best_cost_ = kInfinity;
    std::uint64_t evaluations_ = 0;
};

// ---------------------------------------------------------------------------
// Ant colony optimisation.
// ---------------------------------------------------------------------------
class Aco final : public Metaheuristic {
public:
    Aco(const Problem& problem, std::uint64_t seed, AcoConfig cfg)
        : p_(problem), rng_(seed), cfg_(cfg), tau_(problem.pheromone_size(), cfg.tau0) {
        Candidate seed_solution = p_.make(p_.construct(rng_));
        ++evaluations_;
        best_ = seed_solution.x;
        best_cost_ = seed_solution.cost;
    }

    std::string name() const override { return "aco"; }

    void step() override {
        Solution iteration_best;
        double iteration_cost = kInfinity;
        for (int a = 0; a < cfg_.ants; ++a) {
            Candidate c = p_.make(p_.construct_aco(tau_, cfg_.alpha, cfg_.beta, rng_));
            ++evaluations_;
            for (int k = 0; k < cfg_.local_search_moves; ++k) {
                const Move m = p_.random_move(c, rng_);
                const double d = p_.delta(c, m);
                ++evaluations_;
                if (std::isfinite(d) && d < 0.0) p_.apply(c, m, d);
            }
            if (c.cost < iteration_cost) {
                iteration_cost = c.cost;
                iteration_best = c.x;
            }
        }
        evaporate(tau_, cfg_.rho, cfg_.tau_min);
        // Only the iteration best and the global best deposit, and both
        // deposit a constant. A deposit proportional to 1/cost is the usual
        // rule but it is not scale free: the knapsack objective is negative in
        // internal units, so 1/cost changes sign and the trail collapses. A
        // constant keeps one configuration valid across all three problems,
        // and the trail is still bounded because evaporation removes a fixed
        // fraction every iteration.
        if (!iteration_best.empty()) {
            p_.deposit(iteration_best, cfg_.q, tau_);
            if (iteration_cost < best_cost_) {
                best_cost_ = iteration_cost;
                best_ = iteration_best;
            }
        }
        if (!best_.empty()) p_.deposit(best_, 2.0 * cfg_.q, tau_);
    }

    const Solution& best() const override { return best_; }
    double best_cost() const override { return best_cost_; }
    std::uint64_t evaluations() const override { return evaluations_; }

    bool inject(const Solution& x, double cost) override {
        if (!(cost < best_cost_)) return false;
        best_cost_ = cost;
        best_ = x;
        // Reinforcing the immigrant is what actually makes cooperation work
        // for this algorithm: adopting it as the best without touching the
        // trail would leave every ant walking the old one.
        p_.deposit(best_, 2.0 * cfg_.q, tau_);
        return true;
    }

    const std::vector<double>& pheromone() const { return tau_; }

private:
    const Problem& p_;
    Rng rng_;
    AcoConfig cfg_;
    std::vector<double> tau_;
    Solution best_;
    double best_cost_ = kInfinity;
    std::uint64_t evaluations_ = 0;
};

}  // namespace

void evaporate(std::vector<double>& tau, double rho, double tau_min) {
    const double keep = 1.0 - rho;
    for (double& t : tau) {
        t *= keep;
        if (t < tau_min) t = tau_min;
    }
}

MetaheuristicPtr make_annealing(const Problem& problem, std::uint64_t seed,
                                AnnealingConfig cfg) {
    return std::make_unique<Annealing>(problem, seed, cfg);
}

MetaheuristicPtr make_tabu(const Problem& problem, std::uint64_t seed, TabuConfig cfg) {
    return std::make_unique<Tabu>(problem, seed, cfg);
}

MetaheuristicPtr make_genetic(const Problem& problem, std::uint64_t seed,
                              GeneticConfig cfg) {
    return std::make_unique<Genetic>(problem, seed, cfg);
}

MetaheuristicPtr make_aco(const Problem& problem, std::uint64_t seed, AcoConfig cfg) {
    return std::make_unique<Aco>(problem, seed, cfg);
}

MetaheuristicPtr make_algorithm(const std::string& name, const Problem& problem,
                                std::uint64_t seed) {
    if (name == "annealing" || name == "sa") return make_annealing(problem, seed);
    if (name == "tabu" || name == "ts") return make_tabu(problem, seed);
    if (name == "genetic" || name == "ga") return make_genetic(problem, seed);
    if (name == "aco") return make_aco(problem, seed);
    throw std::runtime_error("unknown algorithm: " + name);
}

}  // namespace anneal
