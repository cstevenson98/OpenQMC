#include <benchmark/benchmark.h>
#include <complex>
#include <random>
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>

#include "core/types.cuh"
#include "core/types.h"
#include "la/SparseImplGPU.cuh"
#include "la/VectImplGPU.cuh"

// Helper function to generate random sparse matrix with given density
t_hostMat generateRandomSparseMatrix(size_t rows, size_t cols, double density) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1.0, 1.0);
  std::uniform_real_distribution<> prob(0.0, 1.0);

  t_hostMat mat(rows, std::vector<std::complex<double>>(cols));
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      if (prob(gen) < density) {
        mat[i][j] = std::complex<double>(dis(gen), dis(gen));
      } else {
        mat[i][j] = std::complex<double>(0.0, 0.0);
      }
    }
  }
  return mat;
}

// Helper function to generate random vector
t_hostVect generateRandomVector3(size_t size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1.0, 1.0);

  t_hostVect vec(size);
  for (size_t i = 0; i < size; ++i) {
    vec[i] = std::complex<double>(dis(gen), dis(gen));
  }
  return vec;
}

// Benchmark matrix-vector multiplication
static void BM_SparseMatrixVectorMultGPU(benchmark::State &state) {
  const size_t rows = state.range(0);
  const size_t cols = state.range(0);
  const double density = 0.01; // 1% non-zero elements

  auto mat_data = generateRandomSparseMatrix(rows, cols, density);
  auto vec_data = generateRandomVector3(cols);

  SparseImplGPU mat(mat_data);
  VectImplGPU vec(vec_data);

  // Calculate theoretical FLOPS (8 FLOPS per non-zero element: 4 mults, 2 adds
  // for real, same for imaginary)
  const double theoretical_flops = 8.0 * mat.NNZ();

  for (auto _ : state) {
    auto result = mat.VectMult(vec);
    benchmark::DoNotOptimize(result);
  }

  state.counters["FLOPS"] =
      benchmark::Counter(theoretical_flops, benchmark::Counter::kIsRate,
                         benchmark::Counter::kIs1024);
  state.counters["NNZ"] = mat.NNZ();
}

// Benchmark matrix-vector multiplication (in-place)
static void BM_SparseMatrixVectorMultInPlaceGPU(benchmark::State &state) {
  const size_t rows = state.range(0);
  const size_t cols = state.range(0);
  const double density = 0.01; // 1% non-zero elements

  auto mat_data = generateRandomSparseMatrix(rows, cols, density);
  auto vec_data = generateRandomVector3(cols);

  SparseImplGPU mat(mat_data);
  VectImplGPU vec(vec_data);
  VectImplGPU result(rows); // Pre-allocate result vector

  // Calculate theoretical FLOPS (8 FLOPS per non-zero element: 4 mults, 2 adds
  // for real, same for imaginary)
  const double theoretical_flops = 8.0 * mat.NNZ();

  for (auto _ : state) {
    mat.VectMultInPlace(vec, result);
    benchmark::DoNotOptimize(result);
  }

  state.counters["FLOPS"] =
      benchmark::Counter(theoretical_flops, benchmark::Counter::kIsRate,
                         benchmark::Counter::kIs1024);
  state.counters["NNZ"] = mat.NNZ();
}

// Benchmark matrix-matrix multiplication
static void BM_SparseMatrixMultGPU(benchmark::State &state) {
  const size_t rows = state.range(0);
  const size_t cols = state.range(0);
  const double density = 0.01; // 1% non-zero elements

  auto matA_data = generateRandomSparseMatrix(rows, cols, density);
  auto matB_data = generateRandomSparseMatrix(cols, rows, density);

  SparseImplGPU matA(matA_data);
  SparseImplGPU matB(matB_data);

  // Calculate theoretical FLOPS (8 FLOPS per non-zero element pair: 4 mults, 2
  // adds for real, same for imaginary)
  const double theoretical_flops = 8.0 * matA.NNZ() * matB.NNZ() / cols;

  for (auto _ : state) {
    auto result = matA.RightMult(matB);
    benchmark::DoNotOptimize(result);
  }

  state.counters["FLOPS"] =
      benchmark::Counter(theoretical_flops, benchmark::Counter::kIsRate,
                         benchmark::Counter::kIs1024);
  state.counters["NNZ_A"] = matA.NNZ();
  state.counters["NNZ_B"] = matB.NNZ();
}

// Benchmark matrix addition
static void BM_SparseMatrixAddGPU(benchmark::State &state) {
  const size_t rows = state.range(0);
  const size_t cols = state.range(0);
  const double density = 0.1; // 10% non-zero elements

  auto matA_data = generateRandomSparseMatrix(rows, cols, density);
  auto matB_data = generateRandomSparseMatrix(rows, cols, density);

  SparseImplGPU matA(matA_data);
  SparseImplGPU matB(matB_data);

  // Calculate theoretical FLOPS (4 FLOPS per overlapping non-zero element: 2
  // adds for real, 2 for imaginary)
  const double theoretical_flops = 4.0 * (matA.NNZ() + matB.NNZ()) / 2;

  for (auto _ : state) {
    auto result = matA.Add(matB);
    benchmark::DoNotOptimize(result);
  }

  state.counters["FLOPS"] =
      benchmark::Counter(theoretical_flops, benchmark::Counter::kIsRate,
                         benchmark::Counter::kIs1024);
  state.counters["NNZ_A"] = matA.NNZ();
  state.counters["NNZ_B"] = matB.NNZ();
}

// Benchmark scalar multiplication
static void BM_SparseMatrixScaleGPU(benchmark::State &state) {
  const size_t rows = state.range(0);
  const size_t cols = state.range(0);
  const double density = 0.1; // 10% non-zero elements

  auto mat_data = generateRandomSparseMatrix(rows, cols, density);
  auto alpha = std::complex<double>(2.0, 1.0);

  SparseImplGPU mat(mat_data);

  // Calculate theoretical FLOPS (4 FLOPS per non-zero element: 2 mults for
  // real, 2 for imaginary)
  const double theoretical_flops = 4.0 * mat.NNZ();

  for (auto _ : state) {
    auto result = mat.Scale(alpha);
    benchmark::DoNotOptimize(result);
  }

  state.counters["FLOPS"] =
      benchmark::Counter(theoretical_flops, benchmark::Counter::kIsRate,
                         benchmark::Counter::kIs1024);
  state.counters["NNZ"] = mat.NNZ();
}

// Register benchmarks with different matrix sizes
BENCHMARK(BM_SparseMatrixVectorMultGPU)
    ->RangeMultiplier(2)
    ->Range(1 << 6, 1 << 10) // Test with matrices from 256x256 to 4096x4096
    ->MinTime(-5.0);         // Limit to 5 seconds

BENCHMARK(BM_SparseMatrixVectorMultInPlaceGPU)
    ->RangeMultiplier(2)
    ->Range(1 << 6, 1 << 13) // Test with matrices from 256x256 to 4096x4096
    ->MinTime(-5.0);         // Limit to 5 seconds

BENCHMARK(BM_SparseMatrixMultGPU)
    ->RangeMultiplier(2)
    ->Range(1 << 6, 1 << 10) // Test with matrices from 256x256 to 4096x4096
    ->MinTime(-5.0);         // Limit to 5 seconds

BENCHMARK(BM_SparseMatrixAddGPU)
    ->RangeMultiplier(2)
    ->Range(1 << 6, 1 << 10) // Test with matrices from 256x256 to 4096x4096
    ->MinTime(-5.0);         // Limit to 5 seconds

BENCHMARK(BM_SparseMatrixScaleGPU)
    ->RangeMultiplier(2)
    ->Range(1 << 6, 1 << 10) // Test with matrices from 256x256 to 4096x4096
    ->MinTime(-5.0);         // Limit to 5 seconds

BENCHMARK_MAIN();