// Summary statistics and the derived performance quantities.
//
// A metaheuristic is stochastic, so a single run is an anecdote. Everything
// reported by the harness passes through here, and the summary carries the
// spread rather than only the centre.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace anneal {

struct Summary {
    std::size_t n = 0;
    double min = 0.0;
    double q1 = 0.0;
    double median = 0.0;
    double q3 = 0.0;
    double max = 0.0;
    double mean = 0.0;
    double iqr() const { return q3 - q1; }
};

// Quantile by linear interpolation between the closest ranks, which is the
// definition used by NumPy and by R's default. It is stated here because the
// three common definitions disagree on small samples, and the repetition
// counts in this project are small.
double quantile(std::vector<double> sorted_or_not, double p);

Summary summarise(std::vector<double> values);

// Speedup against the single thread baseline, and efficiency, which is the
// speedup divided by the thread count. Both are computed from medians, never
// from a single pair of runs.
double speedup(double baseline_time, double parallel_time);
double efficiency(double speedup_value, int threads);

// Relative gap to a reference value. The denominator is clamped away from
// zero because one of the three problems has an optimum of exactly zero
// conflicts, where a relative gap is undefined and an absolute one is what
// the reader wants.
double relative_gap(double cost, double reference);

// Fixed width table rendering, so the demo output lines up in a terminal and
// can be pasted into the report without reformatting.
std::string pad_left(const std::string& s, std::size_t width);
std::string pad_right(const std::string& s, std::size_t width);
std::string format_number(double value, int decimals);

}  // namespace anneal
