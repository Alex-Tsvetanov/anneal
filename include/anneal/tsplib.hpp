// Minimal TSPLIB reader and writer for the EUC_2D subset.
//
// One published instance (berlin52) is vendored under data/ so a measurement
// against a known optimum does not need the network. The parser stays general
// so other EUC_2D files can be dropped in the same way. The writer exists so
// generated instances can be exported and so the parser can be tested against
// a round trip. ATT and GEO weight types are refused: their published optima
// are not comparable under Euclidean length.
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

// Convert a parsed EUC_2D instance. Pass Euc2dMode::TsplibNint together with
// a published TSPLIB optimum; Continuous is for round-trip tests of generated
// instances whose lengths were never integers.
TspProblem tsplib_to_problem(const TsplibInstance& instance, double optimum = -1.0,
                             Euc2dMode mode = Euc2dMode::Continuous);

}  // namespace anneal
