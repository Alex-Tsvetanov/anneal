# Anneal

A parallel metaheuristics framework for combinatorial optimisation. Course project for
**Metaheuristics**, MEng in Computer and Software Engineering, Faculty of Computer Systems
and Technologies, Technical University of Sofia.

## What it is

Many practical scheduling, routing and packing problems are too large to solve exactly, so
they are attacked with metaheuristics: general search strategies that trade the guarantee of
optimality for a bound on running time. Anneal puts several of those strategies behind one
problem interface, runs them under several parallel execution schemes, and measures what
each combination actually costs. The point of the project is the measurement: speedup and
efficiency against thread count, solution quality against a fixed budget, and an honest
account of where the time goes.

## Zero dependencies

The whole thing builds with a C++20 compiler and CMake, and nothing else. No package
manager, no network access at configure time, no vendored archives. The test runner is
ninety lines in `tests/check.hpp`; the timing harness is `std::chrono::steady_clock`; the
benchmark instances are generated rather than downloaded. This is not minimalism for its
own sake: a build that needs a package manager is a build nobody runs, and a measurement
that needs a download is a measurement nobody repeats.

## Build

Verified on the machine described in the report: Windows, g++ 15.2.0 (MinGW-w64),
CMake 4.3.2, Ninja 1.13.2, 12 logical CPUs.

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Any generator works; Ninja is just what was used. Release is the default if you do not pass
`CMAKE_BUILD_TYPE`.

## Run

One command shows the whole system working: four metaheuristics on one instance under one
time budget, then the best of them across every thread count the machine has, then the three
parallel schemes side by side, then the instances whose optimum is known.

```bash
cmake --build build --target demo
# or directly:
./build/anneal_demo
```

Tests, 28 of them, registered individually with CTest:

```bash
ctest --test-dir build --output-on-failure
```

The measurement programme. Each sub-command prints the command that produced its table, so
a number in the report can be traced back to a run:

```bash
./build/anneal_bench                 # everything, takes a few minutes
./build/anneal_bench objective       # cost of one objective evaluation
./build/anneal_bench neighbourhood   # how the 2-opt move is chosen
./build/anneal_bench falsesharing    # padded against packed per-worker counters
./build/anneal_bench profile         # time attribution by ablation
./build/anneal_bench quality         # four algorithms on four instances
./build/anneal_bench schemes         # the three parallel schemes
./build/anneal_bench speedup         # thread scan, time to target
```

## What is implemented

**Problem contract** (`include/anneal/core.hpp`): solution representation, objective,
incremental evaluation with cached state, neighbourhood operator, construction heuristic,
recombination, and pheromone hooks. Implemented for three problems:

| Problem | Representation | Move | Known optimum |
|---|---|---|---|
| Travelling salesman | permutation | 2-opt over a neighbour list | closed form on circle and grid instances |
| 0/1 knapsack | 0/1 indicator | flip or swap, repaired | exact, by dynamic programming |
| Graph colouring | colour per vertex | recolour a conflicting vertex | zero, by planted proper colouring |

**Metaheuristics** (`src/algorithms.cpp`): simulated annealing with three cooling schedules
and a temperature calibrated from the instance, tabu search with a direct-mapped tabu table
and an aspiration criterion, a generational genetic algorithm with tournament selection and
elitism, and ant colony optimisation with candidate lists, evaporation and an elitist
deposit.

**Parallel schemes** (`src/parallel.cpp`), all on `std::jthread` and `std::atomic`:
independent multi-start, a cooperative scheme with one shared best solution updated through
a compare-and-swap loop, and a ring island model with periodic migration through mailboxes.
Exchange periods are counted in objective evaluations, not iterations, because one iteration
means a million-per-second for annealing and about a hundred and fifty for an ant colony.

**Measurement harness** (`include/anneal/experiment.hpp`): repeated runs with seed control,
median and interquartile range over every column, time-to-target distributions with a
pilot-calibrated target, and per-worker instrumentation for synchronisation time, adopted
migrants and failed atomic updates.

## Instances

Nothing is downloaded. Every instance is generated, and every stated optimum is either
proved in closed form or computed exactly:

- **Circle TSP**: points in convex position, so the optimal tour is the hull order and its
  length is `n · 2R · sin(π/n)`. Not used for comparisons, because the nearest neighbour
  construction already solves it.
- **Grid TSP**: `k × k` points, `k` even. No tour edge can be shorter than the spacing, so
  `n · spacing` is a lower bound, and a tour attaining it exists. The test constructs one.
- **Knapsack**: uncorrelated integral weights, optimum by dynamic programming, verified
  against exhaustive enumeration on a 16-item instance.
- **Graph colouring**: random k-partite graph with a planted k-clique, so a zero-conflict
  colouring exists by construction and k colours are genuinely needed.
- **Uniform TSP**: random points. The optimum is **not** known, the class returns NaN, and
  quality is reported as a tour length rather than a gap.

A TSPLIB reader and writer for the EUC_2D subset is included and tested, so a published
instance can be dropped in from outside, but no measurement depends on one.

## Documentation

The project report lives in `docs/` and is written in Bulgarian, because the subject is
taught in Bulgarian and the layout is normative for the faculty. It follows the TU-Sofia
FKST formatting rules: A4, Times metrics at 12pt, 1.5 line spacing, Roman-numbered section
headings, tables captioned above and figures captioned below.

```bash
cd docs
latexmk -pdf Main.tex   # output: docs/build/Main.pdf
```

`latexmk` exits 0 even when the bibliography silently fails, so check the log rather than the
exit code:

```bash
grep "You've used" docs/build/Main.blg   # must match the entry count in references.bib
```

## Selected results

All measured on the machine above; the full tables and their caveats are in the report.

- Time to a fixed target falls by up to **7.2x at 12 threads**, and is superlinear at two to
  four threads. That is a property of the heavily skewed time-to-target distribution, not of
  the machine, and the interquartile range column shows it.
- The cooperative scheme improves median tour quality by **0.93%** over independent
  multi-start and costs about **1%** of CPU time in synchronisation. On the knapsack the
  ordering reverses.
- Move generation takes **70%** of an annealing iteration; the objective takes under **11%**.
  The expectation that the objective dominates was wrong.
- Bounding the 2-opt reversal span, which looked like an optimisation, cost **10.5%** of
  solution quality.
- Per-worker counters sharing a cache line are **11 to 15 times** slower than padded ones.

## Status

- [x] Problem interface, implemented for three problems
- [x] Simulated annealing, tabu search, genetic algorithm, ant colony optimisation
- [x] Independent multi-start, cooperative shared best, island model
- [x] Instance generators with known optima; TSPLIB EUC_2D reader and writer
- [x] 28 correctness tests, registered individually with CTest
- [x] Benchmark harness with seed control and quartile summaries
- [x] Time attribution by ablation
- [x] Results chapter filled from measurements

## License

MIT. See [LICENSE](LICENSE).
