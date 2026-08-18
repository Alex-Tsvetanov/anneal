// The three parallel execution schemes, and the shared state they need.
//
// All three drive the same Metaheuristic contract and differ only in what they
// exchange and when. That is the point: a comparison between the schemes is
// then a comparison of communication policy, not of two different searches
// that happen to carry the same name.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

#include "anneal/core.hpp"

namespace anneal {

struct Budget {
    // Wall clock limit. Zero means no limit.
    std::chrono::milliseconds wall{0};
    // Total objective evaluations across all workers. Zero means no limit.
    // Each worker gets an equal share, so a run under an evaluation budget is
    // reproducible while a run under a wall clock budget cannot be.
    std::uint64_t evaluations = 0;
    // Stop as soon as any worker reaches this cost. Used for time to target.
    double target = -kInfinity;
};

struct WorkerMetrics {
    std::uint64_t iterations = 0;
    std::uint64_t evaluations = 0;
    std::uint64_t improvements = 0;   // times this worker improved its own best
    std::uint64_t offers = 0;         // times it published to shared state
    std::uint64_t adopted = 0;        // times it took a solution from elsewhere
    std::uint64_t cas_failures = 0;   // compare and swap retries on the shared best
    std::uint64_t sync_wait_ns = 0;   // time inside the shared best or a mailbox
};

struct RunResult {
    Solution best;
    double cost = kInfinity;
    double wall_ms = 0.0;
    std::uint64_t evaluations = 0;
    bool reached_target = false;
    double time_to_target_ms = -1.0;
    std::vector<WorkerMetrics> workers;

    std::uint64_t total_sync_wait_ns() const;
    std::uint64_t total_cas_failures() const;
};

// A worker builds its own metaheuristic from its own seed. Passing a factory
// rather than a list of objects keeps every worker's state thread local by
// construction, so the only shared state in the process is the one the scheme
// deliberately introduces.
using Factory = std::function<MetaheuristicPtr(std::uint64_t seed, int worker)>;

// The one piece of shared mutable state in the cooperative scheme.
//
// The atomic holds only the cost and is the fast path: every worker reads it
// often and writes it rarely, so a mutex on the read path would serialise the
// common case for the sake of the rare one. The solution vector itself cannot
// be atomic, so it is guarded by a mutex and re-checked under that mutex; the
// atomic alone would let a worker that wins the compare and swap be overtaken
// before it copies, leaving the published cost and the published solution
// describing different things.
class SharedBest {
public:
    bool offer(const Solution& x, double cost, WorkerMetrics& metrics);
    // Cheap, lock free, may be stale by one update.
    double peek() const { return cost_.load(std::memory_order_acquire); }
    // Copies out only if the version moved since the caller last looked.
    bool read(Solution& out, double& cost, std::uint64_t& seen_version,
              WorkerMetrics& metrics) const;
    double cost() const;
    Solution snapshot() const;

private:
    std::atomic<double> cost_{kInfinity};
    std::atomic<std::uint64_t> version_{0};
    mutable std::mutex mu_;
    Solution best_;
    double stored_cost_ = kInfinity;
};

// Iteration counter that decides when it is worth reading the clock. A
// steady_clock read costs about as much as an annealing iteration, so checking
// the deadline every iteration would be measuring the timer. The stride grows
// until consecutive checks are at least a millisecond apart, which bounds the
// overshoot past the deadline at roughly one millisecond.
class Deadline {
public:
    explicit Deadline(std::chrono::milliseconds budget);
    bool expired();
    bool unlimited() const { return unlimited_; }

private:
    using Clock = std::chrono::steady_clock;
    Clock::time_point end_;
    Clock::time_point last_;
    std::uint64_t iterations_ = 0;
    std::uint64_t next_check_ = 1;
    std::uint64_t stride_ = 1;
    bool unlimited_ = false;
};

// Scheme 1. No communication at all. This is the reference against which the
// cost of communication in the other two is measured.
RunResult run_multistart(const Factory& factory, int threads, const Budget& budget,
                         std::uint64_t seed);

// Scheme 2. One shared best solution. Every `sync_evaluations` objective
// evaluations a worker publishes its own best and, if the shared one is better
// than what it holds, adopts it.
//
// The period is counted in objective evaluations rather than in iterations
// because an iteration means something different in each algorithm: annealing
// runs a million a second and one ant colony sweep runs about a hundred and
// fifty times a second. A period in iterations that is reasonable for the
// first means the second never exchanges anything at all, which was measured
// before it was fixed.
RunResult run_cooperative(const Factory& factory, int threads, const Budget& budget,
                          std::uint64_t seed, std::uint64_t sync_evaluations = 64);

// Scheme 3. Island model on a ring. Every `migration_evaluations` objective
// evaluations a worker posts its best into the next island's mailbox and
// drains its own. Nothing is global, so an island keeps its own trajectory
// much longer than a worker in the cooperative scheme does.
RunResult run_island(const Factory& factory, int threads, const Budget& budget,
                     std::uint64_t seed, std::uint64_t migration_evaluations = 512);

// Dispatch by name, for the demo and the benchmark harness.
RunResult run_scheme(const std::string& scheme, const Factory& factory, int threads,
                     const Budget& budget, std::uint64_t seed);

}  // namespace anneal
