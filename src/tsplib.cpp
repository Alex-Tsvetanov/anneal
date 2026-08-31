#include "anneal/tsplib.hpp"

#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace anneal {

namespace {

std::string trim(const std::string& s) {
    std::size_t a = 0;
    std::size_t b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

}  // namespace

TsplibInstance parse_tsplib(const std::string& text) {
    TsplibInstance out;
    std::istringstream in(text);
    std::string line;
    std::size_t dimension = 0;
    bool in_coords = false;

    while (std::getline(in, line)) {
        const std::string t = trim(line);
        if (t.empty()) continue;
        if (t == "EOF") break;
        if (t == "NODE_COORD_SECTION") {
            in_coords = true;
            continue;
        }
        if (!in_coords) {
            const std::size_t colon = t.find(':');
            if (colon == std::string::npos) continue;
            const std::string key = trim(t.substr(0, colon));
            const std::string value = trim(t.substr(colon + 1));
            if (key == "NAME") out.name = value;
            else if (key == "COMMENT") out.comment = value;
            else if (key == "EDGE_WEIGHT_TYPE") out.edge_weight_type = value;
            else if (key == "DIMENSION") dimension = static_cast<std::size_t>(std::stoul(value));
            continue;
        }
        // Coordinate line: index x y. The index is read and discarded: TSPLIB
        // numbers nodes from one, and relying on that numbering rather than on
        // the order of the section is how off-by-one instance corruption gets
        // in.
        std::istringstream ls(t);
        long long index = 0;
        double x = 0.0;
        double y = 0.0;
        if (!(ls >> index >> x >> y)) {
            throw std::runtime_error("TSPLIB: malformed coordinate line: " + t);
        }
        out.xs.push_back(x);
        out.ys.push_back(y);
    }

    if (out.xs.empty()) {
        throw std::runtime_error("TSPLIB: no NODE_COORD_SECTION found");
    }
    if (dimension != 0 && dimension != out.xs.size()) {
        throw std::runtime_error("TSPLIB: DIMENSION says " + std::to_string(dimension) +
                                 " but " + std::to_string(out.xs.size()) +
                                 " coordinates were read");
    }
    if (!out.edge_weight_type.empty() && out.edge_weight_type != "EUC_2D") {
        throw std::runtime_error("TSPLIB: unsupported EDGE_WEIGHT_TYPE " +
                                 out.edge_weight_type);
    }
    return out;
}

TsplibInstance read_tsplib_file(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot open " + path);
    std::ostringstream buffer;
    buffer << file.rdbuf();
    return parse_tsplib(buffer.str());
}

std::string write_tsplib(const TsplibInstance& instance) {
    std::ostringstream out;
    out << "NAME : " << instance.name << "\n";
    out << "TYPE : TSP\n";
    if (!instance.comment.empty()) out << "COMMENT : " << instance.comment << "\n";
    out << "DIMENSION : " << instance.xs.size() << "\n";
    out << "EDGE_WEIGHT_TYPE : EUC_2D\n";
    out << "NODE_COORD_SECTION\n";
    out.setf(std::ios::fixed);
    out.precision(6);
    for (std::size_t i = 0; i < instance.xs.size(); ++i) {
        out << (i + 1) << " " << instance.xs[i] << " " << instance.ys[i] << "\n";
    }
    out << "EOF\n";
    return out.str();
}

TspProblem tsplib_to_problem(const TsplibInstance& instance, double optimum, Euc2dMode mode) {
    return TspProblem(instance.xs, instance.ys,
                      instance.name.empty() ? "tsplib" : instance.name, optimum, mode);
}

}  // namespace anneal
