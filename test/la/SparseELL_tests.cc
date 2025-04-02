#include <gtest/gtest.h>

#include <complex>

#include "la/SparseELL.h"

TEST(SparseELLTests, DefaultConstructorTest) {
  SparseELL A;
  EXPECT_EQ(A.DimX(), 0);
  EXPECT_EQ(A.DimY(), 0);
  EXPECT_EQ(A.MaxNnzPerRow(), 0);
}

TEST(SparseELLTests, ConstructorTest) {
  SparseELL A(2, 2, 2);
  EXPECT_EQ(A.DimX(), 2);
  EXPECT_EQ(A.DimY(), 2);
  EXPECT_EQ(A.MaxNnzPerRow(), 2);
}

TEST(SparseELLTests, ConstructorThrowTest) {
  // Test for invalid dimensions
  EXPECT_THROW(SparseELL(-1, 2, 2), std::invalid_argument);
  EXPECT_THROW(SparseELL(2, -1, 2), std::invalid_argument);
  EXPECT_THROW(SparseELL(2, 2, -1), std::invalid_argument);
}

TEST(SparseELLTests, CopyConstructorTest) {
  SparseELL A(2, 2, 2);
  SparseELL B(A);
  EXPECT_EQ(B.DimX(), 2);
  EXPECT_EQ(B.DimY(), 2);
  EXPECT_EQ(B.MaxNnzPerRow(), 2);
}

TEST(SparseELLTests, MoveConstructorTest) {
  SparseELL A(2, 2, 2);
  SparseELL B(std::move(A));
  EXPECT_EQ(B.DimX(), 2);
  EXPECT_EQ(B.DimY(), 2);
  EXPECT_EQ(B.MaxNnzPerRow(), 2);
}

TEST(SparseELLTests, CopyAssignmentTest) {
  SparseELL A(2, 2, 2);
  SparseELL B;
  B = A;
  EXPECT_EQ(B.DimX(), 2);
  EXPECT_EQ(B.DimY(), 2);
  EXPECT_EQ(B.MaxNnzPerRow(), 2);
}

TEST(SparseELLTests, CopyConstructorNoThrowEmptyTest) {
  SparseELL A;
  EXPECT_NO_THROW(SparseELL B(A));
}

// Test with various sizes
TEST(SparseELLTests, CopyAssignmentTest_nxm) {
  for (int n = 1; n <= 32; n *= 2) {
    for (int m = 1; m <= 32; m *= 2) {
      SparseELL A(n, m, 2);
      SparseELL B;
      B = A;
      EXPECT_EQ(B.DimX(), n);
      EXPECT_EQ(B.DimY(), m);
      EXPECT_EQ(B.MaxNnzPerRow(), 2);
    }
  }
}

TEST(SparseELLTests, AdditionTest) {
  SparseELL A(2, 2, 2);
  SparseELL B(2, 2, 2);
  SparseELL C = A.Add(B);
  EXPECT_EQ(C.DimX(), 2);
  EXPECT_EQ(C.DimY(), 2);
  EXPECT_EQ(C.MaxNnzPerRow(), 2);
}

TEST(SparseELLTests, RightMultTest) {
  SparseELL A(2, 2, 2);
  SparseELL B(2, 2, 2);
  SparseELL C = A.RightMult(B);
  EXPECT_EQ(C.DimX(), 2);
  EXPECT_EQ(C.DimY(), 2);
  EXPECT_EQ(C.MaxNnzPerRow(), 2);
}

TEST(SparseELLTests, ScaleTest) {
  SparseELL A(2, 2, 2);
  auto alpha = std::complex<double>(2.0, 0.0);
  SparseELL B = A.Scale(alpha);
  EXPECT_EQ(B.DimX(), 2);
  EXPECT_EQ(B.DimY(), 2);
  EXPECT_EQ(B.MaxNnzPerRow(), 2);
}

TEST(SparseELLTests, TransposeTest) {
  SparseELL A(2, 3, 2);
  SparseELL B = A.Transpose();
  EXPECT_EQ(B.DimX(), 3);
  EXPECT_EQ(B.DimY(), 2);
  EXPECT_EQ(B.MaxNnzPerRow(), 2);
}

TEST(SparseELLTests, HermitianCTest) {
  SparseELL A(2, 2, 2);
  SparseELL B = A.HermitianC();
  EXPECT_EQ(B.DimX(), 2);
  EXPECT_EQ(B.DimY(), 2);
  EXPECT_EQ(B.MaxNnzPerRow(), 2);
}

TEST(SparseELLTests, PrintTest) {
  SparseELL A(2, 2, 2);
  // This test is just to ensure that the Print function does not crash
  EXPECT_NO_THROW(A.Print(0, 2));
}

TEST(SparseELLTests, TransposeTest_nxm) {
  for (int n = 1; n <= 32; n *= 2) {
    for (int m = 1; m <= 32; m *= 2) {
      SparseELL A(n, m, 2);
      SparseELL B = A.Transpose();
      EXPECT_EQ(B.DimX(), m);
      EXPECT_EQ(B.DimY(), n);
      EXPECT_EQ(B.MaxNnzPerRow(), 2);
    }
  }
}

TEST(SparseELLTests, ScalarMultiplicationDoesntThrowTest) {
  // test many scalar values
  for (int i = 0; i < 100; i++) {
    SparseELL A(2, 2, 2);
    EXPECT_NO_THROW(SparseELL B = A.Scale(std::complex<double>(i, 0.0)));
  }
}

TEST(SparseELLTests, AdditionThrowTest) {
  SparseELL A(2, 2, 2);
  SparseELL B(3, 3, 2);
  // Test for dimension mismatch
  EXPECT_THROW(SparseELL C = A.Add(B), std::invalid_argument);
}

TEST(SparseELLTests, RightMultThrowTest) {
  SparseELL A(2, 3, 2);
  SparseELL B(4, 2, 2);
  // Test for dimension mismatch
  EXPECT_THROW(SparseELL C = A.RightMult(B), std::invalid_argument);
}