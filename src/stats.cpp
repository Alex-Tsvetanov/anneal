#include "anneal/stats.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>

namespace anneal {

double quantile(std::vector<double> values, double p) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    if (values.size() == 1) return values.front();
    const double position = p * static_cast<double>(values.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(position));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
    const double weight = position - static_cast<double>(lower);
    return values[lower] * (1.0 - weight) + values[upper] * weight;
}

Summary summarise(std::vector<double> values) {
    Summary s;
    if (values.empty()) return s;
    std::sort(values.begin(), values.end());
    s.n = values.size();
    s.min = values.front();
    s.max = values.back();
    s.q1 = quantile(values, 0.25);
    s.median = quantile(values, 0.5);
    s.q3 = quantile(values, 0.75);
    s.mean = std::accumulate(values.begin(), values.end(), 0.0) /
             static_cast<double>(values.size());
    return s;
}

double speedup(double baseline_time, double parallel_time) {
    if (parallel_time <= 0.0) return 0.0;
    return baseline_time / parallel_time;
}

double efficiency(double speedup_value, int threads) {
    if (threads <= 0) return 0.0;
    return speedup_value / static_cast<double>(threads);
}

double relative_gap(double cost, double reference) {
    const double denominator = std::max(1.0, std::fabs(reference));
    return (cost - reference) / denominator;
}

std::string pad_left(const std::string& s, std::size_t width) {
    if (s.size() >= width) return s;
    return std::string(width - s.size(), ' ') + s;
}

std::string pad_right(const std::string& s, std::size_t width) {
    if (s.size() >= width) return s;
    return s + std::string(width - s.size(), ' ');
}

std::string format_number(double value, int decimals) {
    char buffer[64];
    std::snprintf(buffer, sizeof buffer, "%.*f", decimals, value);
    return std::string(buffer);
}

}  // namespace anneal
