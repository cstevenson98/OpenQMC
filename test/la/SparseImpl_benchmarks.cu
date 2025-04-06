#include <benchmark/benchmark.h>
#include <complex>
#include <random>
#include <vector>

#include "core/eigen_types.h"
#include "core/types.h"
#include "la/SparseImpl.cuh"
#include "la/Vect.h"
#include "la/VectImpl.cuh"

// Helper function to generate random sparse matrix with given density
t_hostMat generateRandomSparseMatrix22(size_t rows, size_t cols,
                                       double density) {
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
t_hostVect generateRandomVector22(size_t size) {
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
static void BM_SparseMatrixVectorMult(benchmark::State &state) {
  const size_t rows = state.range(0);
  const size_t cols = state.range(0);
  const double density = 0.01; // 1% non-zero elements

  auto mat_data = generateRandomSparseMatrix22(rows, cols, density);
  auto vec_data = generateRandomVector22(cols);

  SparseImpl mat(mat_data);
  VectImpl vec(vec_data);

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
}

// Benchmark matrix-matrix multiplication
static void BM_SparseMatrixMult(benchmark::State &state) {
  const size_t rows = state.range(0);
  const size_t cols = state.range(0);
  const double density = 0.01; // 1% non-zero elements

  auto matA_data = generateRandomSparseMatrix22(rows, cols, density);
  auto matB_data = generateRandomSparseMatrix22(cols, rows, density);

  SparseImpl matA(matA_data);
  SparseImpl matB(matB_data);

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
}

// Benchmark matrix addition
static void BM_SparseMatrixAdd(benchmark::State &state) {
  const size_t rows = state.range(0);
  const size_t cols = state.range(0);
  const double density = 0.1; // 10% non-zero elements

  auto matA_data = generateRandomSparseMatrix22(rows, cols, density);
  auto matB_data = generateRandomSparseMatrix22(rows, cols, density);

  SparseImpl matA(matA_data);
  SparseImpl matB(matB_data);

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
}

// Benchmark scalar multiplication
static void BM_SparseMatrixScale(benchmark::State &state) {
  const size_t rows = state.range(0);
  const size_t cols = state.range(0);
  const double density = 0.1; // 10% non-zero elements

  auto mat_data = generateRandomSparseMatrix22(rows, cols, density);
  auto alpha = std::complex<double>(2.0, 1.0);

  SparseImpl mat(mat_data);

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
}

// Register benchmarks with different matrix sizes
BENCHMARK(BM_SparseMatrixVectorMult)
    ->RangeMultiplier(2)
    ->Range(1 << 6, 1 << 10); // Test with matrices from 256x256 to 4096x4096

BENCHMARK(BM_SparseMatrixMult)
    ->RangeMultiplier(2)
    ->Range(1 << 6, 1 << 10); // Test with matrices from 256x256 to 4096x4096

BENCHMARK(BM_SparseMatrixAdd)
    ->RangeMultiplier(2)
    ->Range(1 << 6, 1 << 10); // Test with matrices from 256x256 to 4096x4096

BENCHMARK(BM_SparseMatrixScale)
    ->RangeMultiplier(2)
    ->Range(1 << 6, 1 << 10); // Test with matrices from 256x256 to 4096x4096

BENCHMARK_MAIN();