cmake_minimum_required(VERSION 3.20)
project(anneal VERSION 0.1.0 LANGUAGES CXX
        DESCRIPTION "Parallel metaheuristics framework built for measurement")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Zero mandatory third-party dependencies. Threads is a toolchain component,
# not a package to install: std::jthread needs pthread on POSIX toolchains.
find_package(Threads REQUIRED)

option(ANNEAL_NATIVE "Build with -march=native (not portable, off by default)" OFF)

add_library(anneal
  src/rng.cpp
  src/tsp.cpp
  src/knapsack.cpp
  src/coloring.cpp
  src/tsplib.cpp
  src/algorithms.cpp
  src/parallel.cpp
  src/experiment.cpp
  src/stats.cpp
)
target_include_directories(anneal PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(anneal PUBLIC Threads::Threads)
target_compile_features(anneal PUBLIC cxx_std_20)
# Absolute path to the in-tree data/ directory so berlin52.tsp is found without
# a working-directory convention and without downloading anything at build time.
target_compile_definitions(anneal PUBLIC ANNEAL_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

if(MSVC)
  target_compile_options(anneal PRIVATE /W4 /permissive-)
else()
  target_compile_options(anneal PRIVATE -Wall -Wextra -Wpedantic -Wshadow)
  if(ANNEAL_NATIVE)
    target_compile_options(anneal PUBLIC -march=native)
  endif()
endif()

# ---------------------------------------------------------------------------
# Tests. The runner is in tests/check.hpp, about 90 lines, no external
# framework. Each case is registered with CTest separately so a failure names
# the case rather than the whole binary.
# ---------------------------------------------------------------------------
enable_testing()
add_executable(anneal_tests tests/test_main.cpp)
target_link_libraries(anneal_tests PRIVATE anneal)
if(NOT MSVC)
  target_compile_options(anneal_tests PRIVATE -Wall -Wextra)
endif()

set(ANNEAL_TEST_CASES
  rng_reproducible
  rng_bounded_range
  stats_median_quartiles
  tsp_circle_optimum_is_known
  tsp_grid_optimum_is_known
  tsp_delta_matches_full_evaluation
  tsp_move_preserves_permutation
  tsp_crossover_preserves_permutation
  tsplib_parse_reads_coordinates
  tsplib_roundtrip_preserves_distances
  berlin52_published_optimum_is_attained
  knapsack_delta_matches_full_evaluation
  knapsack_dp_optimum_matches_brute_force
  knapsack_solution_stays_feasible
  knapsack_vectorised_matches_scalar
  coloring_delta_matches_full_evaluation
  coloring_planted_partition_is_conflict_free
  coloring_crossover_stays_in_palette
  annealing_accepts_worse_only_when_warm
  annealing_never_reports_worse_than_start
  tabu_forbids_recent_move
  tabu_aspiration_overrides_tabu
  genetic_population_improves
  aco_evaporation_and_deposit
  metaheuristics_are_reproducible
  multistart_matches_best_of_workers
  cooperative_best_is_monotone
  island_migration_transfers_solution
  parallel_schemes_return_feasible_solutions
)
foreach(case IN LISTS ANNEAL_TEST_CASES)
  add_test(NAME ${case} COMMAND anneal_tests ${case})
endforeach()

# ---------------------------------------------------------------------------
# Demo and measurement harness.
# ---------------------------------------------------------------------------
add_executable(anneal_demo apps/demo.cpp)
target_link_libraries(anneal_demo PRIVATE anneal)

add_executable(anneal_bench benchmarks/bench.cpp)
target_link_libraries(anneal_bench PRIVATE anneal)

foreach(exe anneal_demo anneal_bench)
  if(MSVC)
    target_compile_options(${exe} PRIVATE /W4)
  else()
    target_compile_options(${exe} PRIVATE -Wall -Wextra -Wpedantic -Wshadow)
  endif()
endforeach()

# `cmake --build build --target demo` runs the demo.
add_custom_target(demo
  COMMAND $<TARGET_FILE:anneal_demo>
  DEPENDS anneal_demo
  USES_TERMINAL
  COMMENT "Running the Anneal demo")
