#include "anneal/parallel.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <thread>

namespace anneal {

namespace {
using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// Per worker state, one object per thread, never shared. The padding is what
// keeps two workers' counters off the same cache line; whether that padding
// earns its keep is measured in the benchmark named false-sharing rather than
// asserted here.
struct alignas(64) Worker {
    MetaheuristicPtr search;
    WorkerMetrics metrics;
    Solution best;
    double best_cost = kInfinity;
    double last_published = kInfinity;
    std::uint64_t seen_version = 0;
    std::uint64_t next_exchange = 0;
    char padding[8]{};
};

// True once this worker has done another `interval` objective evaluations.
bool due_to_exchange(Worker& self, std::uint64_t interval) {
    const std::uint64_t evaluations = self.search->evaluations();
    if (evaluations < self.next_exchange) return false;
    self.next_exchange = evaluations + interval;
    return true;
}

}  // namespace

std::uint64_t RunResult::total_sync_wait_ns() const {
    std::uint64_t total = 0;
    for (const WorkerMetrics& m : workers) total += m.sync_wait_ns;
    return total;
}

std::uint64_t RunResult::total_cas_failures() const {
    std::uint64_t total = 0;
    for (const WorkerMetrics& m : workers) total += m.cas_failures;
    return total;
}

bool SharedBest::offer(const Solution& x, double cost, WorkerMetrics& metrics) {
    double current = cost_.load(std::memory_order_acquire);
    if (!(cost < current)) return false;
    const Clock::time_point t0 = Clock::now();
    bool won = false;
    while (cost < current) {
        if (cost_.compare_exchange_weak(current, cost, std::memory_order_acq_rel,
                                        std::memory_order_acquire)) {
            won = true;
            break;
        }
        // compare_exchange_weak refreshes `current` on failure and may also
        // fail spuriously. Both are counted: the number is reported as
        // contention on the shared best, and it is the honest measure of it.
        ++metrics.cas_failures;
    }
    if (won) {
        std::lock_guard<std::mutex> guard(mu_);
        if (cost <= stored_cost_) {
            best_ = x;
            stored_cost_ = cost;
            version_.fetch_add(1, std::memory_order_release);
        }
    }
    metrics.sync_wait_ns += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0).count());
    return won;
}

bool SharedBest::read(Solution& out, double& cost, std::uint64_t& seen_version,
                      WorkerMetrics& metrics) const {
    const std::uint64_t version = version_.load(std::memory_order_acquire);
    if (version == seen_version) return false;
    const Clock::time_point t0 = Clock::now();
    {
        std::lock_guard<std::mutex> guard(mu_);
        out = best_;
        cost = stored_cost_;
        seen_version = version_.load(std::memory_order_relaxed);
    }
    metrics.sync_wait_ns += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0).count());
    return !out.empty();
}

double SharedBest::cost() const {
    std::lock_guard<std::mutex> guard(mu_);
    return stored_cost_;
}

Solution SharedBest::snapshot() const {
    std::lock_guard<std::mutex> guard(mu_);
    return best_;
}

Deadline::Deadline(std::chrono::milliseconds budget) {
    unlimited_ = budget.count() <= 0;
    last_ = Clock::now();
    end_ = last_ + budget;
}

bool Deadline::expired() {
    if (unlimited_) return false;
    ++iterations_;
    if (iterations_ < next_check_) return false;
    const Clock::time_point now = Clock::now();
    const auto gap = now - last_;
    if (gap < std::chrono::milliseconds(1) && stride_ < 4096) {
        stride_ *= 2;
    } else if (gap > std::chrono::milliseconds(5) && stride_ > 1) {
        stride_ /= 2;
    }
    next_check_ = iterations_ + stride_;
    last_ = now;
    return now >= end_;
}

namespace {

// Everything the three schemes share: build the workers, run them until the
// budget runs out or a worker reaches the target, then collect. The only
// difference between the schemes is the `exchange` callback, which is why it
// is the only thing passed in.
template <typename Exchange>
RunResult run_workers(const Factory& factory, int threads, const Budget& budget,
                      std::uint64_t seed, Exchange exchange) {
    if (threads < 1) threads = 1;
    const std::size_t count = static_cast<std::size_t>(threads);
    const std::uint64_t per_worker_evaluations =
        budget.evaluations == 0 ? 0
                                : std::max<std::uint64_t>(1, budget.evaluations / count);

    std::vector<Worker> workers(count);
    std::atomic<bool> stop{false};
    std::atomic<long long> target_ns{-1};
    const Clock::time_point started = Clock::now();

    {
        std::vector<std::jthread> pool;
        pool.reserve(count);
        for (std::size_t w = 0; w < count; ++w) {
            pool.emplace_back([&, w] {
                Worker& self = workers[w];
                self.search = factory(Rng::derive(seed, w), static_cast<int>(w));
                self.best = self.search->best();
                self.best_cost = self.search->best_cost();
                Deadline deadline(budget.wall);
                for (;;) {
                    if (stop.load(std::memory_order_relaxed)) break;
                    if (deadline.expired()) break;
                    if (per_worker_evaluations != 0 &&
                        self.search->evaluations() >= per_worker_evaluations) {
                        break;
                    }
                    self.search->step();
                    ++self.metrics.iterations;
                    if (self.search->best_cost() < self.best_cost) {
                        self.best_cost = self.search->best_cost();
                        self.best = self.search->best();
                        ++self.metrics.improvements;
                    }
                    exchange(self, static_cast<int>(w));
                    if (self.best_cost <= budget.target) {
                        long long expected = -1;
                        const long long elapsed =
                            std::chrono::duration_cast<std::chrono::nanoseconds>(
                                Clock::now() - started)
                                .count();
                        target_ns.compare_exchange_strong(expected, elapsed);
                        stop.store(true, std::memory_order_relaxed);
                        break;
                    }
                }
                self.metrics.evaluations = self.search->evaluations();
            });
        }
        // std::jthread joins on destruction, so an exception thrown anywhere
        // above cannot leave a worker running behind the returning frame.
    }

    RunResult result;
    result.wall_ms = ms_since(started);
    result.workers.reserve(count);
    for (Worker& w : workers) {
        result.workers.push_back(w.metrics);
        result.evaluations += w.metrics.evaluations;
        if (w.best_cost < result.cost) {
            result.cost = w.best_cost;
            result.best = w.best;
        }
    }
    const long long ns = target_ns.load();
    if (ns >= 0) {
        result.reached_target = true;
        result.time_to_target_ms = static_cast<double>(ns) / 1e6;
    }
    return result;
}

}  // namespace

RunResult run_multistart(const Factory& factory, int threads, const Budget& budget,
                         std::uint64_t seed) {
    return run_workers(factory, threads, budget, seed, [](Worker&, int) {});
}

RunResult run_cooperative(const Factory& factory, int threads, const Budget& budget,
                          std::uint64_t seed, std::uint64_t sync_evaluations) {
    SharedBest shared;
    const std::uint64_t interval = std::max<std::uint64_t>(1, sync_evaluations);
    return run_workers(factory, threads, budget, seed, [&](Worker& self, int) {
        if (!due_to_exchange(self, interval)) return;
        if (self.best_cost < self.last_published) {
            if (shared.offer(self.best, self.best_cost, self.metrics)) {
                ++self.metrics.offers;
            }
            self.last_published = self.best_cost;
        }
        // The lock free peek filters out the common case, where the shared
        // best has nothing to offer this worker, without taking the mutex.
        if (shared.peek() < self.best_cost) {
            Solution incoming;
            double incoming_cost = kInfinity;
            if (shared.read(incoming, incoming_cost, self.seen_version, self.metrics) &&
                incoming_cost < self.best_cost) {
                if (self.search->inject(incoming, incoming_cost)) {
                    ++self.metrics.adopted;
                    self.best = incoming;
                    self.best_cost = incoming_cost;
                }
            }
        }
    });
}

namespace {

struct Mailbox {
    std::mutex mu;
    Solution solution;
    double cost = kInfinity;
    bool full = false;
};

}  // namespace

RunResult run_island(const Factory& factory, int threads, const Budget& budget,
                     std::uint64_t seed, std::uint64_t migration_evaluations) {
    if (threads < 1) threads = 1;
    const std::size_t count = static_cast<std::size_t>(threads);
    // One mailbox per island. unique_ptr because a mutex is neither copyable
    // nor movable, so the vector cannot hold them by value and still grow.
    std::vector<std::unique_ptr<Mailbox>> boxes;
    boxes.reserve(count);
    for (std::size_t i = 0; i < count; ++i) boxes.push_back(std::make_unique<Mailbox>());
    const std::uint64_t interval = std::max<std::uint64_t>(1, migration_evaluations);

    return run_workers(factory, threads, budget, seed, [&](Worker& self, int id) {
        if (!due_to_exchange(self, interval)) return;
        const Clock::time_point t0 = Clock::now();
        // Ring topology: island i emigrates to island i+1 only. A fully
        // connected topology spreads a good solution in one hop and collapses
        // into the cooperative scheme; the ring keeps the islands genuinely
        // separate for a number of migrations proportional to their count.
        const std::size_t next = (static_cast<std::size_t>(id) + 1) % count;
        {
            Mailbox& out = *boxes[next];
            std::lock_guard<std::mutex> guard(out.mu);
            if (self.best_cost < out.cost) {
                out.solution = self.best;
                out.cost = self.best_cost;
                out.full = true;
                ++self.metrics.offers;
            }
        }
        Solution incoming;
        double incoming_cost = kInfinity;
        {
            Mailbox& in = *boxes[static_cast<std::size_t>(id)];
            std::lock_guard<std::mutex> guard(in.mu);
            if (in.full) {
                incoming = in.solution;
                incoming_cost = in.cost;
                in.full = false;
            }
        }
        self.metrics.sync_wait_ns += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0).count());
        if (!incoming.empty() && self.search->inject(incoming, incoming_cost)) {
            ++self.metrics.adopted;
            if (incoming_cost < self.best_cost) {
                self.best_cost = incoming_cost;
                self.best = incoming;
            }
        }
    });
}

RunResult run_scheme(const std::string& scheme, const Factory& factory, int threads,
                     const Budget& budget, std::uint64_t seed) {
    if (scheme == "multistart") return run_multistart(factory, threads, budget, seed);
    if (scheme == "cooperative") return run_cooperative(factory, threads, budget, seed);
    if (scheme == "island") return run_island(factory, threads, budget, seed);
    throw std::runtime_error("unknown scheme: " + scheme);
}

}  // namespace anneal
