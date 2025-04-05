#include <gtest/gtest.h>

#include <complex>

#include "la/Dense.h"
#include "la/Sparse.h"
#include "la/Vect.h"

// Test constructors
TEST(SparseTests, ConstructorWithDimensionsTest) {
  Sparse A(2, 3);
  EXPECT_EQ(A.DimX(), 2);
  EXPECT_EQ(A.DimY(), 3);
  EXPECT_EQ(A.NNZ(), 0);  // Empty matrix should have 0 non-zero elements
}

TEST(SparseTests, ConstructorWithHostMatTest) {
  t_hostMat data = {{{1.0, 2.0}, {0.0, 0.0}, {3.0, 4.0}},
                    {{0.0, 0.0}, {5.0, 6.0}, {0.0, 0.0}}};
  Sparse A(data);
  EXPECT_EQ(A.DimX(), 2);
  EXPECT_EQ(A.DimY(), 3);
  EXPECT_EQ(A.NNZ(), 3);  // Should have 3 non-zero elements

  // Check that the values are correctly stored
  const t_hostMat hostData = A.GetHostData();
  EXPECT_EQ(hostData[0][0], std::complex<double>(1.0, 2.0));
  EXPECT_EQ(hostData[0][2], std::complex<double>(3.0, 4.0));
  EXPECT_EQ(hostData[1][1], std::complex<double>(5.0, 6.0));
}

TEST(SparseTests, CopyConstructorTest) {
  t_hostMat data = {{{1.0, 2.0}, {0.0, 0.0}, {3.0, 4.0}},
                    {{0.0, 0.0}, {5.0, 6.0}, {0.0, 0.0}}};
  Sparse A(data);
  Sparse B(A);
  EXPECT_EQ(B.DimX(), 2);
  EXPECT_EQ(B.DimY(), 3);
  EXPECT_EQ(B.NNZ(), 3);
}

TEST(SparseTests, MoveConstructorTest) {
  t_hostMat data = {{{1.0, 2.0}, {0.0, 0.0}, {3.0, 4.0}},
                    {{0.0, 0.0}, {5.0, 6.0}, {0.0, 0.0}}};
  Sparse A(data);
  Sparse B(std::move(A));
  EXPECT_EQ(B.DimX(), 2);
  EXPECT_EQ(B.DimY(), 3);
  EXPECT_EQ(B.NNZ(), 3);
}

TEST(SparseTests, CopyAssignmentTest) {
  t_hostMat data = {{{1.0, 2.0}, {0.0, 0.0}, {3.0, 4.0}},
                    {{0.0, 0.0}, {5.0, 6.0}, {0.0, 0.0}}};
  Sparse A(data);
  Sparse B(1, 1);
  B = A;
  EXPECT_EQ(B.DimX(), 2);
  EXPECT_EQ(B.DimY(), 3);
  EXPECT_EQ(B.NNZ(), 3);
}

// Test matrix operations
TEST(SparseTests, ScaleTest) {
  t_hostMat data = {{{1.0, 2.0}, {0.0, 0.0}, {3.0, 4.0}},
                    {{0.0, 0.0}, {5.0, 6.0}, {0.0, 0.0}}};
  Sparse A(data);
  Sparse B = A.Scale(std::complex<double>(2.0, 0.0));
  EXPECT_EQ(B.DimX(), 2);
  EXPECT_EQ(B.DimY(), 3);
  EXPECT_EQ(B.NNZ(), 3);
}

// TEST(SparseTests, AddTest) {
//   t_hostMat data1 = {{{1.0, 2.0}, {0.0, 0.0}, {3.0, 4.0}},
//                      {{0.0, 0.0}, {5.0, 6.0}, {0.0, 0.0}}};
//   t_hostMat data2 = {{{0.0, 0.0}, {7.0, 8.0}, {0.0, 0.0}},
//                      {{9.0, 10.0}, {0.0, 0.0}, {11.0, 12.0}}};
//   Sparse A(data1);
//   Sparse B(data2);
//   Sparse C = A.Add(B);
//   C.Print();
//   EXPECT_EQ(C.DimX(), 2);
//   EXPECT_EQ(C.DimY(), 3);
//   EXPECT_EQ(C.NNZ(), 6);
// }

TEST(SparseTests, RightMultTest) {
  t_hostMat data1 = {{{1.0, 2.0}, {0.0, 0.0}, {3.0, 4.0}},
                     {{0.0, 0.0}, {5.0, 6.0}, {0.0, 0.0}}};
  t_hostMat data2 = {{{7.0, 8.0}, {0.0, 0.0}},
                     {{0.0, 0.0}, {9.0, 10.0}},
                     {{0.0, 0.0}, {11.0, 12.0}}};
  Sparse A(data1);
  Sparse B(data2);
  Sparse C = A.RightMult(B);
  EXPECT_EQ(C.DimX(), 2);
  EXPECT_EQ(C.DimY(), 2);
}

TEST(SparseTests, TransposeTest) {
  t_hostMat data = {{{1.0, 2.0}, {0.0, 0.0}, {3.0, 4.0}},
                    {{0.0, 0.0}, {5.0, 6.0}, {0.0, 0.0}}};
  Sparse A(data);
  Sparse B = A.Transpose();
  EXPECT_EQ(B.DimX(), 3);
  EXPECT_EQ(B.DimY(), 2);
  EXPECT_EQ(B.NNZ(), 3);
}

TEST(SparseTests, HermitianCTest) {
  t_hostMat data = {{{1.0, 2.0}, {0.0, 0.0}, {3.0, 4.0}},
                    {{0.0, 0.0}, {5.0, 6.0}, {0.0, 0.0}}};
  Sparse A(data);
  Sparse B = A.HermitianC();
  EXPECT_EQ(B.DimX(), 3);
  EXPECT_EQ(B.DimY(), 2);
  EXPECT_EQ(B.NNZ(), 3);
}

TEST(SparseTests, ToDenseTest) {
  t_hostMat data = {{{1.0, 2.0}, {0.0, 0.0}, {3.0, 4.0}},
                    {{0.0, 0.0}, {5.0, 6.0}, {0.0, 0.0}}};
  Sparse A(data);
  Dense B = A.ToDense();
  EXPECT_EQ(B.DimX(), 2);
  EXPECT_EQ(B.DimY(), 3);
}

TEST(SparseTests, VectMultTest) {
  t_hostMat data = {{{1.0, 2.0}, {0.0, 0.0}, {3.0, 4.0}},
                    {{0.0, 0.0}, {5.0, 6.0}, {0.0, 0.0}}};
  Sparse A(data);
  t_hostVect vectData = {{7.0, 8.0}, {9.0, 10.0}, {11.0, 12.0}};
  Vect v(vectData);
  Vect result = A.VectMult(v);
  EXPECT_EQ(result.size(), 2);
}

// Test operators
// TEST(SparseTests, AdditionOperatorTest) {
//   t_hostMat data1 = {{{1.0, 2.0}, {0.0, 0.0}, {3.0, 4.0}},
//                      {{0.0, 0.0}, {5.0, 6.0}, {0.0, 0.0}}};
//   t_hostMat data2 = {{{0.0, 0.0}, {7.0, 8.0}, {0.0, 0.0}},
//                      {{9.0, 10.0}, {0.0, 0.0}, {11.0, 12.0}}};
//   Sparse A(data1);
//   Sparse B(data2);
//   Sparse C = A + B;
//   EXPECT_EQ(C.DimX(), 2);
//   EXPECT_EQ(C.DimY(), 3);
//   EXPECT_EQ(C.NNZ(), 5);
// }

TEST(SparseTests, SubtractionOperatorTest) {
  t_hostMat data1 = {{{1.0, 2.0}, {0.0, 0.0}, {3.0, 4.0}},
                     {{0.0, 0.0}, {5.0, 6.0}, {0.0, 0.0}}};
  t_hostMat data2 = {{{0.0, 0.0}, {7.0, 8.0}, {0.0, 0.0}},
                     {{9.0, 10.0}, {0.0, 0.0}, {11.0, 12.0}}};
  Sparse A(data1);
  Sparse B(data2);
  Sparse C = A - B;
  EXPECT_EQ(C.DimX(), 2);
  EXPECT_EQ(C.DimY(), 3);
}

TEST(SparseTests, ScalarMultiplicationOperatorTest) {
  t_hostMat data = {{{1.0, 2.0}, {0.0, 0.0}, {3.0, 4.0}},
                    {{0.0, 0.0}, {5.0, 6.0}, {0.0, 0.0}}};
  Sparse A(data);
  Sparse B = A * std::complex<double>(2.0, 0.0);
  EXPECT_EQ(B.DimX(), 2);
  EXPECT_EQ(B.DimY(), 3);
  EXPECT_EQ(B.NNZ(), 3);
}

TEST(SparseTests, MatrixMultiplicationOperatorTest) {
  t_hostMat data1 = {{{1.0, 2.0}, {0.0, 0.0}, {3.0, 4.0}},
                     {{0.0, 0.0}, {5.0, 6.0}, {0.0, 0.0}}};
  t_hostMat data2 = {{{7.0, 8.0}, {0.0, 0.0}},
                     {{0.0, 0.0}, {9.0, 10.0}},
                     {{0.0, 0.0}, {11.0, 12.0}}};
  Sparse A(data1);
  Sparse B(data2);
  Sparse C = A * B;
  EXPECT_EQ(C.DimX(), 2);
  EXPECT_EQ(C.DimY(), 2);
}

TEST(SparseTests, ScalarMultiplicationFriendOperatorTest) {
  t_hostMat data = {{{1.0, 2.0}, {0.0, 0.0}, {3.0, 4.0}},
                    {{0.0, 0.0}, {5.0, 6.0}, {0.0, 0.0}}};
  Sparse A(data);
  Sparse B = std::complex<double>(2.0, 0.0) * A;
  EXPECT_EQ(B.DimX(), 2);
  EXPECT_EQ(B.DimY(), 3);
  EXPECT_EQ(B.NNZ(), 3);
}

// Test utility functions
TEST(SparseTests, PrintFunctionsTest) {
  t_hostMat data = {{{1.0, 2.0}, {0.0, 0.0}, {3.0, 4.0}},
                    {{0.0, 0.0}, {5.0, 6.0}, {0.0, 0.0}}};
  Sparse A(data);
  // These functions don't return anything, so we just call them to ensure they
  // don't crash
  A.Print();
  A.PrintRe();
  A.PrintIm();
  A.PrintAbs();
}

// Test conversion functions
TEST(SparseTests, ToSparseCOOTest) {
  t_hostMat data = {{{1.0, 2.0}, {0.0, 0.0}, {3.0, 4.0}},
                    {{0.0, 0.0}, {5.0, 6.0}, {0.0, 0.0}}};
  Dense d(data);
  Sparse s = ToSparseCOO(d);
  EXPECT_EQ(s.DimX(), 2);
  EXPECT_EQ(s.DimY(), 3);
  EXPECT_EQ(s.NNZ(), 3);
}

TEST(SparseTests, SparseRowsCOOTest) {
  t_hostMat data = {{{1.0, 2.0}, {0.0, 0.0}, {3.0, 4.0}},
                    {{0.0, 0.0}, {5.0, 6.0}, {0.0, 0.0}}};
  Sparse s(data);
  std::vector<CompressedRow> rows = SparseRowsCOO(s);
  EXPECT_EQ(rows.size(), 2);
}

TEST(SparseTests, SparseColsCOOTest) {
  t_hostMat data = {{{1.0, 2.0}, {0.0, 0.0}, {3.0, 4.0}},
                    {{0.0, 0.0}, {5.0, 6.0}, {0.0, 0.0}}};
  Sparse s(data);
  std::vector<CompressedRow> cols = SparseColsCOO(s);
  EXPECT_EQ(cols.size(), 3);
}
