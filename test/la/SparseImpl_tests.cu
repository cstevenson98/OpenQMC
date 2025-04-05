#include <gtest/gtest.h>

#include <complex>
#include <cuda_runtime.h>

#include "la/Dense.h"
#include "la/DenseImpl.cuh"
#include "la/SparseImpl.cuh"
#include "la/Vect.h"

// Test constructor with dimensions
TEST(SparseImplTests, ConstructorWithDimensionsTest) {
  SparseImpl A(2, 3);
  EXPECT_EQ(A.DimX, 2);
  EXPECT_EQ(A.DimY, 3);
  EXPECT_EQ(A.CPUData.nonZeros(),
            0); // Empty matrix should have 0 non-zero elements
}

// Test constructor with host matrix
TEST(SparseImplTests, ConstructorWithHostMatTest) {
  t_hostMat data = {{{1.0, 2.0}, {0.0, 0.0}, {3.0, 4.0}},
                    {{0.0, 0.0}, {5.0, 6.0}, {0.0, 0.0}}};
  SparseImpl A(data);
  EXPECT_EQ(A.DimX, 2);
  EXPECT_EQ(A.DimY, 3);
  EXPECT_EQ(A.CPUData.nonZeros(), 3); // Should have 3 non-zero elements

  // Check that the values are correctly stored
  EXPECT_EQ(A.CPUData.coeff(0, 0), std::complex<double>(1.0, 2.0));
  EXPECT_EQ(A.CPUData.coeff(0, 2), std::complex<double>(3.0, 4.0));
  EXPECT_EQ(A.CPUData.coeff(1, 1), std::complex<double>(5.0, 6.0));
}

// Test Scale method
TEST(SparseImplTests, ScaleTest) {
  t_hostMat data = {{{1.0, 2.0}, {0.0, 0.0}, {3.0, 4.0}},
                    {{0.0, 0.0}, {5.0, 6.0}, {0.0, 0.0}}};
  SparseImpl A(data);
  SparseImpl B = A.Scale(std::complex<double>(2.0, 0.0));

  EXPECT_EQ(B.DimX, 2);
  EXPECT_EQ(B.DimY, 3);
  EXPECT_EQ(B.CPUData.nonZeros(), 3);

  // Check that the values are correctly scaled
  EXPECT_EQ(B.CPUData.coeff(0, 0), std::complex<double>(2.0, 4.0));
  EXPECT_EQ(B.CPUData.coeff(0, 2), std::complex<double>(6.0, 8.0));
  EXPECT_EQ(B.CPUData.coeff(1, 1), std::complex<double>(10.0, 12.0));
}

// Test Transpose method
TEST(SparseImplTests, TransposeTest) {
  t_hostMat data = {{{1.0, 2.0}, {0.0, 0.0}, {3.0, 4.0}},
                    {{0.0, 0.0}, {5.0, 6.0}, {0.0, 0.0}}};
  SparseImpl A(data);
  SparseImpl B = A.Transpose();

  EXPECT_EQ(B.DimX, 3);
  EXPECT_EQ(B.DimY, 2);
  EXPECT_EQ(B.CPUData.nonZeros(), 3);

  // Check that the values are correctly transposed
  EXPECT_EQ(B.CPUData.coeff(0, 0), std::complex<double>(1.0, 2.0));
  EXPECT_EQ(B.CPUData.coeff(1, 1), std::complex<double>(5.0, 6.0));
  EXPECT_EQ(B.CPUData.coeff(2, 0), std::complex<double>(3.0, 4.0));
}

// Test HermitianC method
TEST(SparseImplTests, HermitianCTest) {
  t_hostMat data = {{{1.0, 2.0}, {0.0, 0.0}, {3.0, 4.0}},
                    {{0.0, 0.0}, {5.0, 6.0}, {0.0, 0.0}}};
  SparseImpl A(data);
  SparseImpl B = A.HermitianC();

  EXPECT_EQ(B.DimX, 3);
  EXPECT_EQ(B.DimY, 2);
  EXPECT_EQ(B.CPUData.nonZeros(), 3);

  // Check that the values are correctly conjugated and transposed
  EXPECT_EQ(B.CPUData.coeff(0, 0), std::complex<double>(1.0, -2.0));
  EXPECT_EQ(B.CPUData.coeff(1, 1), std::complex<double>(5.0, -6.0));
  EXPECT_EQ(B.CPUData.coeff(2, 0), std::complex<double>(3.0, -4.0));
}

// Test ToDense method
TEST(SparseImplTests, ToDenseTest) {
  t_hostMat data = {{{1.0, 2.0}, {0.0, 0.0}, {3.0, 4.0}},
                    {{0.0, 0.0}, {5.0, 6.0}, {0.0, 0.0}}};
  SparseImpl A(data);

  // Check the elements of A are correct
  EXPECT_EQ(A.CPUData.coeff(0, 0), std::complex<double>(1.0, 2.0));
  EXPECT_EQ(A.CPUData.coeff(0, 1), std::complex<double>(0.0, 0.0));
  EXPECT_EQ(A.CPUData.coeff(0, 2), std::complex<double>(3.0, 4.0));
  EXPECT_EQ(A.CPUData.coeff(1, 0), std::complex<double>(0.0, 0.0));
  EXPECT_EQ(A.CPUData.coeff(1, 1), std::complex<double>(5.0, 6.0));
  EXPECT_EQ(A.CPUData.coeff(1, 2), std::complex<double>(0.0, 0.0));

  DenseImpl B = A.ToDense();

  EXPECT_EQ(B.DimX, 2);
  EXPECT_EQ(B.DimY, 3);
}

// Test VectMult method
TEST(SparseImplTests, VectMultTest) {
  auto d00 = t_cplx(1.0, 2.0);
  auto d02 = t_cplx(3.0, 4.0);
  auto d11 = t_cplx(5.0, 6.0);

  t_hostMat data = {{d00, {0.0, 0.0}, d02}, //
                    {{0.0, 0.0}, d11, {0.0, 0.0}}};
  SparseImpl A(data);

  auto v0 = t_cplx(7.0, 8.0);
  auto v1 = t_cplx(9.0, 10.0);
  auto v2 = t_cplx(11.0, 12.0);
  t_hostVect vectData = {v0, v1, v2};
  VectImpl v(vectData);

  VectImpl result = A.VectMult(v);

  EXPECT_EQ(result.GetData()[0], d00 * v0 + d02 * v2);
  EXPECT_EQ(result.GetData()[1], d11 * v1);
}

// Test RightMult method
TEST(SparseImplTests, RightMultTest) {
  t_hostMat data1 = {{{1.0, 2.0}, {0.0, 0.0}, {3.0, 4.0}},
                     {{0.0, 0.0}, {5.0, 6.0}, {0.0, 0.0}}};
  SparseImpl A(data1);

  t_hostMat data2 = {{{7.0, 8.0}, {0.0, 0.0}},
                     {{0.0, 0.0}, {9.0, 10.0}},
                     {{0.0, 0.0}, {11.0, 12.0}}};
  SparseImpl B(data2);

  SparseImpl C = A.RightMult(B);

  // The result should be:
  // [1+2i  0    3+4i] [7+8i  0    ]   [(1+2i)(7+8i)  (3+4i)(11+12i)]
  // [0     5+6i  0  ] [0     9+10i] = [0              (5+6i)(9+10i) ]
  //                    [0     11+12i]

  // (1,1): (1+2i)(7+8i) = 7-16 + i(8+14) = -9 + i(22)
  // (1,2): (3+4i)(11+12i) = 33-48 + i(36+44) = -15 + i(80)
  // (2,1): 0
  // (2,2): (5+6i)(9+10i) = 45-60 + i(50+54) = -15 + i(104)

  EXPECT_EQ(C.DimX, 2);
  EXPECT_EQ(C.DimY, 2);
  EXPECT_EQ(C.CPUData.nonZeros(), 3);

  // Check the elements
  EXPECT_EQ(C.CPUData.coeff(0, 0), std::complex<double>(-9.0, 22.0));
  EXPECT_EQ(C.CPUData.coeff(0, 1), std::complex<double>(-15.0, 80.0));
  EXPECT_EQ(C.CPUData.coeff(1, 1), std::complex<double>(-15.0, 104.0));
}

// Test operator+ method
TEST(SparseImplTests, AdditionOperatorTest) {
  t_hostMat data1 = {{{1.0, 2.0}, {0.0, 0.0}, {3.0, 4.0}},
                     {{0.0, 0.0}, {5.0, 6.0}, {0.0, 0.0}}};
  SparseImpl A(data1);

  t_hostMat data2 = {{{0.0, 0.0}, {7.0, 8.0}, {0.0, 0.0}},
                     {{9.0, 10.0}, {0.0, 0.0}, {11.0, 12.0}}};
  SparseImpl B(data2);

  SparseImpl C = A + B;

  // The result should be:
  // [1+2i  7+8i  3+4i  ]
  // [9+10i 5+6i  11+12i]

  EXPECT_EQ(C.DimX, 2);
  EXPECT_EQ(C.DimY, 3);
  EXPECT_EQ(C.CPUData.nonZeros(), 6);

  // Check that all elements are present
  EXPECT_EQ(C.CPUData.coeff(0, 0), std::complex<double>(1.0, 2.0));
  EXPECT_EQ(C.CPUData.coeff(0, 1), std::complex<double>(7.0, 8.0));
  EXPECT_EQ(C.CPUData.coeff(0, 2), std::complex<double>(3.0, 4.0));
  EXPECT_EQ(C.CPUData.coeff(1, 0), std::complex<double>(9.0, 10.0));
  EXPECT_EQ(C.CPUData.coeff(1, 1), std::complex<double>(5.0, 6.0));
  EXPECT_EQ(C.CPUData.coeff(1, 2), std::complex<double>(11.0, 12.0));
}

// Test operator- method
TEST(SparseImplTests, SubtractionOperatorTest) {
  t_hostMat data1 = {{{1.0, 2.0}, {0.0, 0.0}, {3.0, 4.0}},
                     {{0.0, 0.0}, {5.0, 6.0}, {0.0, 0.0}}};
  SparseImpl A(data1);

  t_hostMat data2 = {{{0.0, 0.0}, {7.0, 8.0}, {0.0, 0.0}},
                     {{9.0, 10.0}, {0.0, 0.0}, {11.0, 12.0}}};
  SparseImpl B(data2);

  SparseImpl C = A - B;

  // The result should be:
  // [1+2i  -7-8i  3+4i  ]
  // [-9-10i 5+6i  -11-12i]

  EXPECT_EQ(C.DimX, 2);
  EXPECT_EQ(C.DimY, 3);
  EXPECT_EQ(C.CPUData.nonZeros(), 6);

  // Check that all elements are present
  EXPECT_EQ(C.CPUData.coeff(0, 0), std::complex<double>(1.0, 2.0));
  EXPECT_EQ(C.CPUData.coeff(0, 1), std::complex<double>(-7.0, -8.0));
  EXPECT_EQ(C.CPUData.coeff(0, 2), std::complex<double>(3.0, 4.0));
  EXPECT_EQ(C.CPUData.coeff(1, 0), std::complex<double>(-9.0, -10.0));
  EXPECT_EQ(C.CPUData.coeff(1, 1), std::complex<double>(5.0, 6.0));
  EXPECT_EQ(C.CPUData.coeff(1, 2), std::complex<double>(-11.0, -12.0));
}

// Test operator* method for scalar multiplication
TEST(SparseImplTests, ScalarMultiplicationOperatorTest) {
  t_hostMat data = {{{1.0, 2.0}, {0.0, 0.0}, {3.0, 4.0}},
                    {{0.0, 0.0}, {5.0, 6.0}, {0.0, 0.0}}};
  SparseImpl A(data);
  SparseImpl B = A * std::complex<double>(2.0, 0.0);

  EXPECT_EQ(B.DimX, 2);
  EXPECT_EQ(B.DimY, 3);
  EXPECT_EQ(B.CPUData.nonZeros(), 3);

  // Check that the values are correctly scaled
  EXPECT_EQ(B.CPUData.coeff(0, 0), std::complex<double>(2.0, 4.0));
  EXPECT_EQ(B.CPUData.coeff(0, 2), std::complex<double>(6.0, 8.0));
  EXPECT_EQ(B.CPUData.coeff(1, 1), std::complex<double>(10.0, 12.0));
}

// Test operator* method for matrix multiplication
TEST(SparseImplTests, MatrixMultiplicationOperatorTest) {
  t_hostMat data1 = {{{1.0, 2.0}, {0.0, 0.0}, {3.0, 4.0}},
                     {{0.0, 0.0}, {5.0, 6.0}, {0.0, 0.0}}};
  SparseImpl A(data1);

  t_hostMat data2 = {{{7.0, 8.0}, {0.0, 0.0}},
                     {{0.0, 0.0}, {9.0, 10.0}},
                     {{0.0, 0.0}, {11.0, 12.0}}};
  SparseImpl B(data2);

  SparseImpl C = A * B;

  // The result should be the same as RightMult
  EXPECT_EQ(C.DimX, 2);
  EXPECT_EQ(C.DimY, 2);
  EXPECT_EQ(C.CPUData.nonZeros(), 3);

  // Check the elements
  EXPECT_EQ(C.CPUData.coeff(0, 0), std::complex<double>(-9.0, 22.0));
  EXPECT_EQ(C.CPUData.coeff(0, 1), std::complex<double>(-15.0, 80.0));
  EXPECT_EQ(C.CPUData.coeff(1, 1), std::complex<double>(-15.0, 104.0));
}

// Test NNZ method
TEST(SparseImplTests, NNZTest) {
  t_hostMat data = {{{1.0, 2.0}, {0.0, 0.0}, {3.0, 4.0}},
                    {{0.0, 0.0}, {5.0, 6.0}, {0.0, 0.0}}};
  SparseImpl A(data);

  EXPECT_EQ(A.NNZ(), 3);
}