// The generator is header-only by design (every call is on the hot path and
// must inline). This translation unit exists so the library has a definition
// to anchor the header to, and so a compile error in rng.hpp is caught by the
// library build rather than only by whoever includes it first.
#include "anneal/rng.hpp"

namespace anneal {
static_assert(sizeof(Rng) == 32, "xoshiro256++ state is four 64-bit words");
}  // namespace anneal
