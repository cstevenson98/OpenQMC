#include <benchmark/benchmark.h>
#include <complex>
#include <random>
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "core/types.cuh"
#include "la/VectImplGPU.cuh"

// Helper function to generate random complex vectors
t_hostVect generateRandomVector2(size_t size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1.0, 1.0);

  t_hostVect vec(size);
  for (size_t i = 0; i < size; ++i) {
    vec[i] = th_cplx(dis(gen), dis(gen));
  }
  return vec;
}

// Benchmark vector addition
static void BM_VectorAdditionGPU(benchmark::State &state) {
  const size_t size = state.range(0);
  auto vec1_data = generateRandomVector2(size);
  auto vec2_data = generateRandomVector2(size);

  VectImplGPU vec1(vec1_data);
  VectImplGPU vec2(vec2_data);
  VectImplGPU result(size); // Pre-allocate result vector

  // Calculate theoretical FLOPS (4 FLOPS per element: 2 adds for real, 2 for
  // imaginary)
  const double theoretical_flops = 4.0 * size;

  for (auto _ : state) {
    vec1.AddInPlace(vec2, result);
    benchmark::DoNotOptimize(result);
  }

  state.counters["FLOPS"] =
      benchmark::Counter(theoretical_flops, benchmark::Counter::kIsRate,
                         benchmark::Counter::kIs1024);
}

// Benchmark vector dot product
static void BM_VectorDotProductGPU(benchmark::State &state) {
  const size_t size = state.range(0);
  auto vec1_data = generateRandomVector2(size);
  auto vec2_data = generateRandomVector2(size);

  VectImplGPU vec1(vec1_data);
  VectImplGPU vec2(vec2_data);

  // Calculate theoretical FLOPS (8 FLOPS per element: 4 mults, 2 adds for real,
  // same for imaginary)
  const double theoretical_flops = 8.0 * size;

  for (auto _ : state) {
    auto result = vec1.Dot(vec2);
    benchmark::DoNotOptimize(result);
  }

  state.counters["FLOPS"] =
      benchmark::Counter(theoretical_flops, benchmark::Counter::kIsRate,
                         benchmark::Counter::kIs1024);
}

// Benchmark vector scaling
static void BM_VectorScalingGPU(benchmark::State &state) {
  const size_t size = state.range(0);
  auto vec_data = generateRandomVector2(size);
  th_cplx scale_factor(2.0, 1.0);

  VectImplGPU vec(vec_data);
  VectImplGPU result(size); // Pre-allocate result vector

  // Calculate theoretical FLOPS (4 FLOPS per element: 2 mults for real, 2 for
  // imaginary)
  const double theoretical_flops = 4.0 * size;

  for (auto _ : state) {
    vec.ScaleInPlace(scale_factor, result);
    benchmark::DoNotOptimize(result);
  }

  state.counters["FLOPS"] =
      benchmark::Counter(theoretical_flops, benchmark::Counter::kIsRate,
                         benchmark::Counter::kIs1024);
}

// Benchmark vector norm
static void BM_VectorNormGPU(benchmark::State &state) {
  const size_t size = state.range(0);
  auto vec_data = generateRandomVector2(size);

  VectImplGPU vec(vec_data);

  // Calculate theoretical FLOPS (3 FLOPS per element: 2 mults for real/imag, 1
  // add)
  const double theoretical_flops = 3.0 * size;

  for (auto _ : state) {
    auto result = vec.Norm();
    benchmark::DoNotOptimize(result);
  }

  state.counters["FLOPS"] =
      benchmark::Counter(theoretical_flops, benchmark::Counter::kIsRate,
                         benchmark::Counter::kIs1024);
}

// Benchmark vector conjugate
static void BM_VectorConjugateGPU(benchmark::State &state) {
  const size_t size = state.range(0);
  auto vec_data = generateRandomVector2(size);

  VectImplGPU vec(vec_data);
  VectImplGPU result(size); // Pre-allocate result vector

  // Calculate theoretical FLOPS (1 FLOP per element: 1 negation for imaginary
  // part)
  const double theoretical_flops = 1.0 * size;

  for (auto _ : state) {
    vec.ConjInPlace(result);
    benchmark::DoNotOptimize(result);
  }

  state.counters["FLOPS"] =
      benchmark::Counter(theoretical_flops, benchmark::Counter::kIsRate,
                         benchmark::Counter::kIs1024);
}

// Register benchmarks with different vector sizes
BENCHMARK(BM_VectorAdditionGPU)
    ->RangeMultiplier(10)
    ->Range(1000, 10000000)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_VectorDotProductGPU)
    ->RangeMultiplier(10)
    ->Range(1000, 10000000)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_VectorScalingGPU)
    ->RangeMultiplier(10)
    ->Range(1000, 10000000)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_VectorNormGPU)
    ->RangeMultiplier(10)
    ->Range(1000, 10000000)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_VectorConjugateGPU)
    ->RangeMultiplier(10)
    ->Range(1000, 10000000)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();