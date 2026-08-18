// Assert based test runner. No external framework: the project has to build on
// a machine with a compiler and CMake and nothing else, and a test framework
// fetched at configure time is a dependency the reader has to satisfy before
// the first build works.
#pragma once

#include <cmath>
#include <cstdio>
#include <exception>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace check {

struct Case {
    std::string name;
    std::function<void()> body;
};

inline std::vector<Case>& registry() {
    static std::vector<Case> cases;
    return cases;
}

struct Registrar {
    Registrar(const char* name, std::function<void()> body) {
        registry().push_back(Case{name, std::move(body)});
    }
};

struct Failure {
    std::string what;
};

inline void fail(const char* file, int line, const std::string& what) {
    char buf[64];
    std::snprintf(buf, sizeof buf, ":%d: ", line);
    throw Failure{std::string(file) + buf + what};
}

inline int run(int argc, char** argv) {
    const std::string filter = argc > 1 ? argv[1] : std::string();
    int ran = 0;
    int failed = 0;
    for (const Case& c : registry()) {
        if (!filter.empty() && c.name != filter) continue;
        ++ran;
        try {
            c.body();
            std::printf("[ ok ] %s\n", c.name.c_str());
        } catch (const Failure& f) {
            std::printf("[FAIL] %s\n       %s\n", c.name.c_str(), f.what.c_str());
            ++failed;
        } catch (const std::exception& e) {
            std::printf("[FAIL] %s\n       unexpected exception: %s\n", c.name.c_str(),
                        e.what());
            ++failed;
        }
    }
    if (ran == 0) {
        std::printf("no test matched '%s'; the CMake list and the sources disagree\n",
                    filter.c_str());
        return 2;
    }
    std::printf("%d run, %d failed\n", ran, failed);
    return failed == 0 ? 0 : 1;
}

}  // namespace check

#define TEST(name)                                             \
    static void name();                                        \
    static ::check::Registrar registrar_##name(#name, name);   \
    static void name()

#define CHECK(cond)                                                                 \
    do {                                                                            \
        if (!(cond)) ::check::fail(__FILE__, __LINE__, "CHECK(" #cond ") failed");   \
    } while (0)

#define CHECK_MSG(cond, msg)                                                        \
    do {                                                                            \
        if (!(cond))                                                                \
            ::check::fail(__FILE__, __LINE__,                                       \
                          std::string("CHECK(" #cond ") failed: ") + (msg));        \
    } while (0)

#define CHECK_NEAR(a, b, eps)                                                       \
    do {                                                                            \
        const double check_a = (a);                                                 \
        const double check_b = (b);                                                 \
        const double check_eps = (eps);                                             \
        if (!(std::fabs(check_a - check_b) <= check_eps)) {                         \
            char check_buf[192];                                                    \
            std::snprintf(check_buf, sizeof check_buf,                              \
                          #a " = %.12g differs from " #b " = %.12g by more than %.3g", \
                          check_a, check_b, check_eps);                             \
            ::check::fail(__FILE__, __LINE__, check_buf);                           \
        }                                                                           \
    } while (0)

#define CHECK_EQ(a, b)                                                              \
    do {                                                                            \
        if (!((a) == (b)))                                                          \
            ::check::fail(__FILE__, __LINE__, "CHECK_EQ(" #a ", " #b ") failed");    \
    } while (0)
