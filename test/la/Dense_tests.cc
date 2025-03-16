#include "la/Dense.h"
#include <complex>
#include <gtest/gtest.h>

TEST(DenseTests, DefaultConstructorTest) {
  Dense A;
  EXPECT_EQ(A.DimX(), 0);
  EXPECT_EQ(A.DimY(), 0);
}

TEST(DenseTests, ConstructorTest) {
  Dense A(2, 2);
  EXPECT_EQ(A.DimX(), 2);
  EXPECT_EQ(A.DimY(), 2);

  // also check all elements are zero
  for (int i = 0; i < A.DimX(); i++) {
    for (int j = 0; j < A.DimY(); j++) {
      EXPECT_EQ(A.GetData(i, j), std::complex<double>(0.0, 0.0));
    }
  }
}

TEST(DenseTests, ConstructorThrowTest) {
  // Test for invalid dimensions
  EXPECT_THROW(Dense(-1, 2), std::invalid_argument);
  EXPECT_THROW(Dense(2, -1), std::invalid_argument);
  EXPECT_THROW(Dense(-1, -1), std::invalid_argument);
}

TEST(DenseTests, CopyConstructorTest) {
  Dense A(2, 2);
  A.GetData(0, 0) = {1.0, 2.0};
  A.GetData(1, 1) = {3.0, 4.0};
  Dense B(A);
  EXPECT_EQ(B.DimX(), 2);
  EXPECT_EQ(B.DimY(), 2);
  EXPECT_EQ(B.GetData(0, 0), std::complex<double>(1.0, 2.0));
  EXPECT_EQ(B.GetData(1, 1), std::complex<double>(3.0, 4.0));
}

TEST(DenseTests, MoveConstructorTest) {
  Dense A(2, 2);
  A.GetData(0, 0) = {1.0, 2.0};
  A.GetData(1, 1) = {3.0, 4.0};
  Dense B(std::move(A));
  EXPECT_EQ(B.DimX(), 2);
  EXPECT_EQ(B.DimY(), 2);
  EXPECT_EQ(B.GetData(0, 0), std::complex<double>(1.0, 2.0));
  EXPECT_EQ(B.GetData(1, 1), std::complex<double>(3.0, 4.0));
}

TEST(DenseTests, CopyAssignmentTest) {
  Dense A(2, 2);
  A.GetData(0, 0) = {1.0, 2.0};
  A.GetData(1, 1) = {3.0, 4.0};
  Dense B;
  B = A;
  EXPECT_EQ(B.DimX(), 2);
  EXPECT_EQ(B.DimY(), 2);
  EXPECT_EQ(B.GetData(0, 0), std::complex<double>(1.0, 2.0));
  EXPECT_EQ(B.GetData(1, 1), std::complex<double>(3.0, 4.0));
}

TEST(DenseTests, CopyConstructorNoThrowEmptyTest) {
  // Test for invalid copy constructor (e.g., from an empty matrix)
  Dense A;
  EXPECT_NO_THROW(Dense B(A));
}

// extend to sizes nxm where n = {2^0, 2^1, ..., 2^5} and
// m = {2^0, 2^1, ..., 2^5}
TEST(DenseTests, CopyAssignmentTest_nxm) {
  for (int n = 1; n <= 32; n *= 2) {
    for (int m = 1; m <= 32; m *= 2) {
      Dense A(n, m);
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
          A.GetData(i, j) = {static_cast<double>(i), static_cast<double>(j)};
        }
      }
      Dense B;
      B = A;
      EXPECT_EQ(B.DimX(), n);
      EXPECT_EQ(B.DimY(), m);
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
          EXPECT_EQ(B.GetData(i, j), std::complex<double>(i, j));
        }
      }
    }
  }
}

TEST(DenseTests, GetDataTest) {
  Dense A(2, 2);
  A.GetData(0, 0) = {1.0, 2.0};
  EXPECT_EQ(A.GetData(0, 0), std::complex<double>(1.0, 2.0));
}

TEST(DenseTests, GetDataThrowTest) {
  Dense A(2, 2);
  // Test for out-of-bounds access
  EXPECT_THROW(A.GetData(2, 0), std::out_of_range);
  EXPECT_THROW(A.GetData(0, 2), std::out_of_range);
  EXPECT_THROW(A.GetData(-1, 0), std::out_of_range);
  EXPECT_THROW(A.GetData(0, -1), std::out_of_range);
}

TEST(DenseTests, SubscriptOperatorTest) {
  Dense A(2, 2);
  A.GetData(0, 0) = {1.0, 2.0};
  auto val00 = A[0, 0];
  EXPECT_EQ(val00, std::complex<double>(1.0, 2.0));
  auto val01 = A[0, 1];
  EXPECT_EQ(val01, std::complex<double>(0.0, 0.0));
}

TEST(DenseTests, AdditionTest) {
  auto a00 = std::complex<double>(1.0, 2.0);
  auto a01 = std::complex<double>(3.0, 4.0);
  auto a10 = std::complex<double>(5.0, 6.0);
  auto a11 = std::complex<double>(7.0, 8.0);
  auto b00 = std::complex<double>(9.0, 10.0);
  auto b01 = std::complex<double>(11.0, 12.0);
  auto b10 = std::complex<double>(13.0, 14.0);
  auto b11 = std::complex<double>(15.0, 16.0);
  auto a_data = t_hostMat{{a00, a01}, {a10, a11}};
  auto b_data = t_hostMat{{b00, b01}, {b10, b11}};
  Dense A(a_data);
  Dense B(b_data);
  Dense C = A + B;

  // Check shape of C
  EXPECT_EQ(C.DimX(), 2);
  EXPECT_EQ(C.DimY(), 2);

  // Check values of C
  EXPECT_EQ(C.GetData(0, 0), a00 + b00);
  EXPECT_EQ(C.GetData(0, 1), a01 + b01);
  EXPECT_EQ(C.GetData(1, 0), a10 + b10);
  EXPECT_EQ(C.GetData(1, 1), a11 + b11);
}

TEST(DenseTests, AdditionThrowTest) {
  Dense A(2, 2);
  Dense B(3, 3);
  // Test for dimension mismatch
  EXPECT_THROW(Dense C = A + B, std::invalid_argument);
}

TEST(DenseTests, SubtractionTest) {
  auto a00 = std::complex<double>(5.0, 6.0);
  auto a01 = std::complex<double>(7.0, 8.0);
  auto a10 = std::complex<double>(9.0, 10.0);
  auto a11 = std::complex<double>(11.0, 12.0);
  auto b00 = std::complex<double>(1.0, 2.0);
  auto b01 = std::complex<double>(3.0, 4.0);
  auto b10 = std::complex<double>(5.0, 6.0);
  auto b11 = std::complex<double>(7.0, 8.0);
  auto a_data = t_hostMat{{a00, a01}, {a10, a11}};
  auto b_data = t_hostMat{{b00, b01}, {b10, b11}};
  Dense A(a_data);
  Dense B(b_data);
  Dense C = A - B;

  // Check shape of C
  EXPECT_EQ(C.DimX(), 2);
  EXPECT_EQ(C.DimY(), 2);

  // Check values of C
  EXPECT_EQ(C.GetData(0, 0), a00 - b00);
  EXPECT_EQ(C.GetData(0, 1), a01 - b01);
  EXPECT_EQ(C.GetData(1, 0), a10 - b10);
  EXPECT_EQ(C.GetData(1, 1), a11 - b11);
}

TEST(DenseTests, ScalarMultiplicationTest) {
  auto a00 = std::complex<double>(1.0, 2.0);
  auto a01 = std::complex<double>(3.0, 4.0);
  auto a10 = std::complex<double>(5.0, 6.0);
  auto a11 = std::complex<double>(7.0, 8.0);
  auto a_data = t_hostMat{{a00, a01}, {a10, a11}};
  Dense A(a_data);
  auto alpha = std::complex<double>(2.0, 0.0);
  Dense B = A * alpha;

  // Check shape of B
  EXPECT_EQ(B.DimX(), 2);
  EXPECT_EQ(B.DimY(), 2);

  // Check values of B
  EXPECT_EQ(B.GetData(0, 0), a00 * alpha);
  EXPECT_EQ(B.GetData(0, 1), a01 * alpha);
  EXPECT_EQ(B.GetData(1, 0), a10 * alpha);
  EXPECT_EQ(B.GetData(1, 1), a11 * alpha);
}

TEST(DenseTests, MatrixMultiplicationTest) {
  auto a00 = std::complex<double>(1.0, 2.0);
  auto a01 = std::complex<double>(3.0, 4.0);
  auto a10 = std::complex<double>(5.0, 6.0);
  auto a11 = std::complex<double>(7.0, 8.0);
  auto b00 = std::complex<double>(1.0, 0.0);
  auto b01 = std::complex<double>(2.0, 0.0);
  auto b10 = std::complex<double>(3.0, 0.0);
  auto b11 = std::complex<double>(4.0, 0.0);
  auto a_data = t_hostMat{{a00, a01}, {a10, a11}};
  auto b_data = t_hostMat{{b00, b01}, {b10, b11}};
  Dense A(a_data);
  Dense B(b_data);
  Dense C = A * B;

  // Check shape of C
  EXPECT_EQ(C.DimX(), 2);
  EXPECT_EQ(C.DimY(), 2);

  // Check values of C
  EXPECT_EQ(C.GetData(0, 0), a00 * b00 + a01 * b10);
  EXPECT_EQ(C.GetData(0, 1), a00 * b01 + a01 * b11);
  EXPECT_EQ(C.GetData(1, 0), a10 * b00 + a11 * b10);
  EXPECT_EQ(C.GetData(1, 1), a10 * b01 + a11 * b11);
}

TEST(DenseTests, TransposeTestSquare) {
  Dense A(2, 2);
  A.GetData(0, 1) = {1.0, 2.0};
  A.GetData(1, 0) = {3.0, 4.0};
  Dense B = A.Transpose();
  EXPECT_EQ(B.GetData(1, 0), std::complex<double>(1.0, 2.0));
  EXPECT_EQ(B.GetData(0, 1), std::complex<double>(3.0, 4.0));
}

TEST(DenseTests, TransposeTestRectangular) {
  Dense A(2, 3);
  A.GetData(0, 1) = {1.0, 2.0};
  A.GetData(1, 2) = {3.0, 4.0};
  Dense B = A.Transpose();
  EXPECT_EQ(B.DimX(), 3);
  EXPECT_EQ(B.DimY(), 2);
  EXPECT_EQ(B.GetData(1, 0), std::complex<double>(1.0, 2.0));
  EXPECT_EQ(B.GetData(2, 1), std::complex<double>(3.0, 4.0));
}

TEST(DenseTests, HermitianCTest) {
  Dense A(2, 2);
  A.GetData(0, 1) = {1.0, 2.0};
  A.GetData(1, 0) = {3.0, 4.0};
  Dense B = A.HermitianC();
  EXPECT_EQ(B.GetData(1, 0), std::complex<double>(1.0, -2.0));
  EXPECT_EQ(B.GetData(0, 1), std::complex<double>(3.0, -4.0));
}

TEST(DenseTests, FlattenedDataTest) {
  Dense A(2, 2);
  A.GetData(0, 0) = {1.0, 2.0};
  A.GetData(1, 1) = {3.0, 4.0};
  auto data = A.FlattenedData();
  EXPECT_EQ(data.size(), 4);
  EXPECT_EQ(data[0], std::complex<double>(1.0, 2.0));
  EXPECT_EQ(data[3], std::complex<double>(3.0, 4.0));
}

TEST(DenseTests, PrintTest) {
  Dense A(2, 2);
  A.GetData(0, 0) = {1.0, 2.0};
  A.GetData(1, 1) = {3.0, 4.0};
  A.Print();
}

TEST(DenseTests, ConstructorWithDataTest) {
  t_hostMat data = {{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}};
  Dense A(data);
  EXPECT_EQ(A.DimX(), 2);
  EXPECT_EQ(A.DimY(), 2);
  EXPECT_EQ(A.GetData(0, 0), std::complex<double>(1.0, 2.0));
  EXPECT_EQ(A.GetData(1, 1), std::complex<double>(7.0, 8.0));
}

TEST(DenseTests, TransposeTest_nxm) {
  for (int n = 1; n <= 32; n *= 2) {
    for (int m = 1; m <= 32; m *= 2) {
      Dense A(n, m);
      // set n,mth element to n + mi
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
          A.GetData(i, j) = {static_cast<double>(i), static_cast<double>(j)};
        }
      }
      Dense B = A.Transpose();
      EXPECT_EQ(B.DimX(), m);
      EXPECT_EQ(B.DimY(), n);

      // check
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
          EXPECT_EQ(A.GetData(i, j), B.GetData(j, i));
        }
      }
    }
  }
}

TEST(DenseTests, SubtractionThrowTest) {
  Dense A(2, 2);
  Dense B(3, 3);
  // Test for dimension mismatch
  EXPECT_THROW(Dense C = A - B, std::invalid_argument);
}

TEST(DenseTests, MatrixMultiplicationThrowTest) {
  Dense A(2, 3);
  Dense B(4, 2);
  // Test for dimension mismatch
  EXPECT_THROW(Dense C = A * B, std::invalid_argument);
}

TEST(DenseTests, ScalarMultiplicationDoesntThrowTest) {
  // test many scalar values
  for (int i = 0; i < 100; i++) {
    Dense A(2, 2);
    EXPECT_NO_THROW(Dense B = A * std::complex<double>(i, 0.0));
  }
}
