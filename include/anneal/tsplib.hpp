// Minimal TSPLIB reader and writer for the EUC_2D subset.
//
// The archive is not vendored and is not fetched at build time, so the demo
// relies on generated instances. The parser is still here and still tested,
// because the point of the format is that a reader can drop a downloaded
// instance next to the binary and measure against a published optimum; the
// writer exists so the generated instances can be exported in the same format
// and so the parser can be tested against a round trip.
#pragma once

#include <string>
#include <vector>

#include "anneal/problems.hpp"

namespace anneal {

struct TsplibInstance {
    std::string name;
    std::string comment;
    std::string edge_weight_type;
    std::vector<double> xs;
    std::vector<double> ys;
};

// Throws std::runtime_error on a malformed file or an unsupported
// EDGE_WEIGHT_TYPE. Parsing is deliberately strict: a silently mis-parsed
// instance produces plausible numbers that are wrong, which is the worst
// failure mode a benchmark can have.
TsplibInstance parse_tsplib(const std::string& text);
TsplibInstance read_tsplib_file(const std::string& path);

std::string write_tsplib(const TsplibInstance& instance);

TspProblem tsplib_to_problem(const TsplibInstance& instance, double optimum = -1.0);

}  // namespace anneal
