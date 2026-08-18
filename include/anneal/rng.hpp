// Deterministic pseudo-random generator.
//
// std::mt19937_64 would have done for the bit stream, but the standard
// distribution objects (std::uniform_int_distribution and friends) are not
// specified down to the bit, so the same seed gives different draws on a
// different standard library. Reproducibility is a stated requirement of this
// project, so the generator and the bounded draw are both written out here.
// xoshiro256++ is 1.34 GB/s class and passes BigCrush; splitmix64 seeds it so
// that a low-entropy seed such as 1 still produces a well mixed state.
#pragma once

#include <cstdint>
#include <limits>

namespace anneal {

class Rng {
public:
    explicit Rng(std::uint64_t seed) noexcept {
        std::uint64_t z = seed;
        for (std::uint64_t& part : s_) {
            z += 0x9E3779B97F4A7C15ULL;
            std::uint64_t t = z;
            t = (t ^ (t >> 30)) * 0xBF58476D1CE4E5B9ULL;
            t = (t ^ (t >> 27)) * 0x94D049BB133111EBULL;
            part = t ^ (t >> 31);
        }
    }

    std::uint64_t next() noexcept {
        const std::uint64_t result = rotl(s_[0] + s_[3], 23) + s_[0];
        const std::uint64_t t = s_[1] << 17;
        s_[2] ^= s_[0];
        s_[3] ^= s_[1];
        s_[1] ^= s_[2];
        s_[0] ^= s_[3];
        s_[2] ^= t;
        s_[3] = rotl(s_[3], 45);
        return result;
    }

    // Uniform in [0, 1). 53 significant bits, the most a double can hold.
    double uniform() noexcept {
        return static_cast<double>(next() >> 11) * 0x1.0p-53;
    }

    // Uniform in [0, n). Rejection sampling, so the draw is exactly uniform;
    // the naive `next() % n` is biased towards small values whenever n does
    // not divide 2^64, and n here is typically a problem size such as 100.
    std::uint64_t below(std::uint64_t n) noexcept {
        if (n <= 1) return 0;
        const std::uint64_t threshold = (0ULL - n) % n;  // 2^64 mod n
        std::uint64_t r = next();
        while (r < threshold) r = next();
        return r % n;
    }

    // Uniform in [lo, hi], inclusive.
    int range(int lo, int hi) noexcept {
        return lo + static_cast<int>(below(static_cast<std::uint64_t>(hi - lo + 1)));
    }

    // Independent stream for worker `index`. Deriving each worker's seed from
    // the run seed keeps a whole run reproducible from one number, while the
    // splitmix64 step keeps neighbouring worker indices from producing
    // correlated streams.
    static std::uint64_t derive(std::uint64_t seed, std::uint64_t index) noexcept {
        std::uint64_t z = seed + (index + 1) * 0x9E3779B97F4A7C15ULL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }

private:
    static std::uint64_t rotl(std::uint64_t x, int k) noexcept {
        return (x << k) | (x >> (64 - k));
    }
    std::uint64_t s_[4];
};

}  // namespace anneal
