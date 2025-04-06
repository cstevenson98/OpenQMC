#include <gtest/gtest.h>

#include <complex>
#include <cuda_runtime.h>

#include "la/Dense.h"
#include "la/DenseImpl.cuh"
#include "la/SparseImpl.cuh"
#include "la/SparseImplGPU.cuh"
#include "la/Vect.h"

// Test constructor with dimensions
TEST(SparseImplGPUTests, ConstructorWithDimensionsTest) {
  SparseImplGPU A(2, 3);
  EXPECT_EQ(A.DimX, 2);
  EXPECT_EQ(A.DimY, 3);
  EXPECT_EQ(A.NNZ(), 0); // Empty matrix should have 0 non-zero elements
}

// Test constructor with host matrix
TEST(SparseImplGPUTests, ConstructorWithHostMatTest) {
  t_hostMat data(2, std::vector<std::complex<double>>(3));
  data[0][0] = std::complex<double>(1.0, 2.0);
  data[0][2] = std::complex<double>(3.0, 4.0);
  data[1][1] = std::complex<double>(5.0, 6.0);

  SparseImplGPU A(data);
  EXPECT_EQ(A.DimX, 2);
  EXPECT_EQ(A.DimY, 3);
  EXPECT_EQ(A.NNZ(), 3); // Should have 3 non-zero elements

  // Check that the values are correctly stored
  t_hostMat hostData = A.GetHostData();
  EXPECT_EQ(hostData[0][0], std::complex<double>(1.0, 2.0));
  EXPECT_EQ(hostData[0][2], std::complex<double>(3.0, 4.0));
  EXPECT_EQ(hostData[1][1], std::complex<double>(5.0, 6.0));
}

// Test constructor from CPU SparseImpl
TEST(SparseImplGPUTests, ConstructorFromSparseImplTest) {
  t_hostMat data(2, std::vector<std::complex<double>>(3));
  data[0][0] = std::complex<double>(1.0, 2.0);
  data[0][2] = std::complex<double>(3.0, 4.0);
  data[1][1] = std::complex<double>(5.0, 6.0);

  SparseImpl cpuMatrix(data);
  SparseImplGPU A(cpuMatrix);

  EXPECT_EQ(A.DimX, 2);
  EXPECT_EQ(A.DimY, 3);
  EXPECT_EQ(A.NNZ(), 3); // Should have 3 non-zero elements

  // Check that the values are correctly stored
  t_hostMat hostData = A.GetHostData();
  EXPECT_EQ(hostData[0][0], std::complex<double>(1.0, 2.0));
  EXPECT_EQ(hostData[0][2], std::complex<double>(3.0, 4.0));
  EXPECT_EQ(hostData[1][1], std::complex<double>(5.0, 6.0));
}

// Test RightMult method
TEST(SparseImplGPUTests, RightMultTest) {
  // Create two sparse matrices
  t_hostMat dataA(2, std::vector<std::complex<double>>(2));
  dataA[0][0] = std::complex<double>(1.0, 0.0);
  dataA[0][1] = std::complex<double>(2.0, 0.0);
  dataA[1][1] = std::complex<double>(3.0, 0.0);

  t_hostMat dataB(2, std::vector<std::complex<double>>(3));
  dataB[0][0] = std::complex<double>(4.0, 0.0);
  dataB[0][2] = std::complex<double>(5.0, 0.0);
  dataB[1][1] = std::complex<double>(6.0, 0.0);

  SparseImplGPU A(dataA);
  SparseImplGPU B(dataB);

  // Multiply matrices
  SparseImplGPU C = A.RightMult(B);

  // Expected result: C = A * B
  // A = [1 2]
  //     [0 3]
  // B = [4 0 5]
  //     [0 6 0]
  // C = [4 12 5]
  //     [0 18 0]

  EXPECT_EQ(C.DimX, 2);
  EXPECT_EQ(C.DimY, 3);

  // Check that the values are correctly computed
  t_hostMat hostData = C.GetHostData();
  EXPECT_EQ(hostData[0][0], std::complex<double>(4.0, 0.0));
  EXPECT_EQ(hostData[0][1], std::complex<double>(12.0, 0.0));
  EXPECT_EQ(hostData[0][2], std::complex<double>(5.0, 0.0));
  EXPECT_EQ(hostData[1][0], std::complex<double>(0.0, 0.0));
  EXPECT_EQ(hostData[1][1], std::complex<double>(18.0, 0.0));
  EXPECT_EQ(hostData[1][2], std::complex<double>(0.0, 0.0));
}

// Test RightMult with complex values
TEST(SparseImplGPUTests, RightMultComplexTest) {
  // Create two sparse matrices with complex values
  t_hostMat dataA(2, std::vector<std::complex<double>>(2));
  dataA[0][0] = std::complex<double>(1.0, 0.0);
  dataA[0][1] = std::complex<double>(2.0, 0.0);
  dataA[1][1] = std::complex<double>(3.0, 0.0);

  // Ie the dense matrix:
  // [1 2]
  // [0 3]

  t_hostMat dataB(2, std::vector<std::complex<double>>(3));
  dataB[0][0] = std::complex<double>(4.0, 0.0);
  dataB[0][2] = std::complex<double>(5.0, 0.0);
  dataB[1][1] = std::complex<double>(6.0, 0.0);

  // Ie the dense matrix:
  // [4 0 5]
  // [0 6 0]

  SparseImplGPU A(dataA);
  SparseImplGPU B(dataB);

  // Multiply matrices
  SparseImplGPU C = A.RightMult(B);

  // result is
  // [4 12 5]
  // [0 18 0]

  EXPECT_EQ(C.DimX, 2);
  EXPECT_EQ(C.DimY, 3);

  // Check that the values are correctly computed
  t_hostMat hostData = C.GetHostData();
  EXPECT_EQ(hostData[0][0], std::complex<double>(4.0, 0.0));
  EXPECT_EQ(hostData[0][1], std::complex<double>(12.0, 0.0));
  EXPECT_EQ(hostData[0][2], std::complex<double>(5.0, 0.0));
  EXPECT_EQ(hostData[1][0], std::complex<double>(0.0, 0.0));
  EXPECT_EQ(hostData[1][1], std::complex<double>(18.0, 0.0));
  EXPECT_EQ(hostData[1][2], std::complex<double>(0.0, 0.0));
}

// Test RightMult with larger matrices
TEST(SparseImplGPUTests, RightMultLargeTest) {
  // Create two larger sparse matrices
  const int dimX = 10;
  const int dimY = 15;
  const int dimZ = 20;

  t_hostMat dataA(dimX, std::vector<std::complex<double>>(
                            dimY, std::complex<double>(0.0, 0.0)));
  t_hostMat dataB(dimY, std::vector<std::complex<double>>(
                            dimZ, std::complex<double>(0.0, 0.0)));

  // Fill with some non-zero values
  for (int i = 0; i < dimX; ++i) {
    for (int j = 0; j < dimY; ++j) {
      if ((i + j) % 3 == 0) {
        dataA[i][j] = std::complex<double>(i + j, i - j);
      }
    }
  }

  for (int i = 0; i < dimY; ++i) {
    for (int j = 0; j < dimZ; ++j) {
      if ((i + j) % 4 == 0) {
        dataB[i][j] = std::complex<double>(i * j, i + j);
      }
    }
  }

  SparseImplGPU A(dataA);
  SparseImplGPU B(dataB);

  // Multiply matrices
  SparseImplGPU C = A.RightMult(B);

  // Verify dimensions
  EXPECT_EQ(C.DimX, dimX);
  EXPECT_EQ(C.DimY, dimZ);

  // Compare with CPU result
  SparseImpl cpuA(dataA);
  SparseImpl cpuB(dataB);
  SparseImpl cpuC = cpuA.RightMult(cpuB);

  // Convert GPU result to host
  t_hostMat hostData = C.GetHostData();

  // Compare values
  for (int i = 0; i < dimX; ++i) {
    for (int j = 0; j < dimZ; ++j) {
      EXPECT_NEAR(hostData[i][j].real(), cpuC.CPUData.coeff(i, j).real(),
                  1e-10);
      EXPECT_NEAR(hostData[i][j].imag(), cpuC.CPUData.coeff(i, j).imag(),
                  1e-10);
    }
  }
}

// Test VectMult method with real values
TEST(SparseImplGPUTests, VectMultRealTest) {
  // Create a sparse matrix
  t_hostMat dataA(2, std::vector<std::complex<double>>(3));
  dataA[0][0] = std::complex<double>(1.0, 0.0);
  dataA[0][2] = std::complex<double>(2.0, 0.0);
  dataA[1][1] = std::complex<double>(3.0, 0.0);

  // Create a vector
  t_hostVect dataX(3);
  dataX[0] = std::complex<double>(4.0, 0.0);
  dataX[1] = std::complex<double>(5.0, 0.0);
  dataX[2] = std::complex<double>(6.0, 0.0);

  SparseImplGPU A(dataA);
  VectImplGPU X(dataX);

  // Multiply matrix by vector
  VectImplGPU Y = A.VectMult(X);

  // Expected result: Y = A * X
  // A = [1 0 2]
  //     [0 3 0]
  // X = [4]
  //     [5]
  //     [6]
  // Y = [16]
  //     [15]

  EXPECT_EQ(Y.Dim, 2);

  // Check that the values are correctly computed
  t_hostVect hostData = Y.GetHostData();
  EXPECT_EQ(hostData[0], std::complex<double>(16.0, 0.0));
  EXPECT_EQ(hostData[1], std::complex<double>(15.0, 0.0));
}

// Test VectMult with larger matrices
TEST(SparseImplGPUTests, VectMultLargeTest) {
  // Create a larger sparse matrix
  const int dimX = 10;
  const int dimY = 15;

  t_hostMat dataA(dimX, std::vector<std::complex<double>>(
                            dimY, std::complex<double>(0.0, 0.0)));
  t_hostVect dataX(dimY, std::complex<double>(0.0, 0.0));

  // Fill with some non-zero values
  for (int i = 0; i < dimX; ++i) {
    for (int j = 0; j < dimY; ++j) {
      if ((i + j) % 3 == 0) {
        dataA[i][j] = std::complex<double>(i + j, i - j);
      }
    }
  }

  for (int j = 0; j < dimY; ++j) {
    if (j % 2 == 0) {
      dataX[j] = std::complex<double>(j, j + 1);
    }
  }

  SparseImplGPU A(dataA);
  VectImplGPU X(dataX);

  // Multiply matrix by vector
  VectImplGPU Y = A.VectMult(X);

  // Verify dimensions
  EXPECT_EQ(Y.Dim, dimX);

  // Compare with CPU result
  SparseImpl cpuA(dataA);
  VectImpl cpuX(dataX);
  VectImpl cpuY = cpuA.VectMult(cpuX);

  // Convert GPU result to host
  t_hostVect hostData = Y.GetHostData();

  // Compare values
  for (int i = 0; i < dimX; ++i) {
    EXPECT_NEAR(hostData[i].real(), cpuY[i].real(), 1e-10);
    EXPECT_NEAR(hostData[i].imag(), cpuY[i].imag(), 1e-10);
  }
}