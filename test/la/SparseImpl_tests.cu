#include <gtest/gtest.h>

#include <complex>

#include "la/Dense.h"
#include "la/SparseImpl.cuh"
#include "la/Vect.h"

// Helper function to create a small test matrix
t_hostMat
createTestMatrix(int rows, int cols,
                 const std::vector<std::vector<std::complex<double>>> &values) {
  t_hostMat matrix(rows, std::vector<std::complex<double>>(
                             cols, std::complex<double>(0.0, 0.0)));
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      matrix[i][j] = values[i][j];
    }
  }
  return matrix;
}

// Test constructor with dimensions
TEST(SparseImplTests, ConstructorWithDimensionsTest) {
  SparseImpl A(2, 3);
  EXPECT_EQ(A.DimX, 2);
  EXPECT_EQ(A.DimY, 3);
  EXPECT_EQ(A.Data.size(), 0); // Empty matrix should have 0 non-zero elements
}

// Test constructor with host matrix
TEST(SparseImplTests, ConstructorWithHostMatTest) {
  t_hostMat data = {{{1.0, 2.0}, {0.0, 0.0}, {3.0, 4.0}},
                    {{0.0, 0.0}, {5.0, 6.0}, {0.0, 0.0}}};
  SparseImpl A(data);
  EXPECT_EQ(A.DimX, 2);
  EXPECT_EQ(A.DimY, 3);
  EXPECT_EQ(A.Data.size(), 3); // Should have 3 non-zero elements

  // Check that the values are correctly stored
  EXPECT_EQ(A.Data[0].Coords[0], 0);
  EXPECT_EQ(A.Data[0].Coords[1], 0);
  EXPECT_EQ(A.Data[0].Val, std::complex<double>(1.0, 2.0));

  EXPECT_EQ(A.Data[1].Coords[0], 0);
  EXPECT_EQ(A.Data[1].Coords[1], 2);
  EXPECT_EQ(A.Data[1].Val, std::complex<double>(3.0, 4.0));

  EXPECT_EQ(A.Data[2].Coords[0], 1);
  EXPECT_EQ(A.Data[2].Coords[1], 1);
  EXPECT_EQ(A.Data[2].Val, std::complex<double>(5.0, 6.0));
}

// Test Scale method
TEST(SparseImplTests, ScaleTest) {
  t_hostMat data = {{{1.0, 2.0}, {0.0, 0.0}, {3.0, 4.0}},
                    {{0.0, 0.0}, {5.0, 6.0}, {0.0, 0.0}}};
  SparseImpl A(data);
  SparseImpl B = A.Scale(std::complex<double>(2.0, 0.0));

  EXPECT_EQ(B.DimX, 2);
  EXPECT_EQ(B.DimY, 3);
  EXPECT_EQ(B.Data.size(), 3);

  // Check that the values are correctly scaled
  EXPECT_EQ(B.Data[0].Val, std::complex<double>(2.0, 4.0));
  EXPECT_EQ(B.Data[1].Val, std::complex<double>(6.0, 8.0));
  EXPECT_EQ(B.Data[2].Val, std::complex<double>(10.0, 12.0));
}

// Test Transpose method
TEST(SparseImplTests, TransposeTest) {
  t_hostMat data = {{{1.0, 2.0}, {0.0, 0.0}, {3.0, 4.0}},
                    {{0.0, 0.0}, {5.0, 6.0}, {0.0, 0.0}}};
  SparseImpl A(data);
  SparseImpl B = A.Transpose();

  EXPECT_EQ(B.DimX, 3);
  EXPECT_EQ(B.DimY, 2);
  EXPECT_EQ(B.Data.size(), 3);

  // Check that the values are correctly transposed
  EXPECT_EQ(B.Data[0].Coords[0], 0);
  EXPECT_EQ(B.Data[0].Coords[1], 0);
  EXPECT_EQ(B.Data[0].Val, std::complex<double>(1.0, 2.0));

  EXPECT_EQ(B.Data[1].Coords[0], 1);
  EXPECT_EQ(B.Data[1].Coords[1], 1);
  EXPECT_EQ(B.Data[1].Val, std::complex<double>(5.0, 6.0));

  EXPECT_EQ(B.Data[2].Coords[0], 2);
  EXPECT_EQ(B.Data[2].Coords[1], 0);
  EXPECT_EQ(B.Data[2].Val, std::complex<double>(3.0, 4.0));
}

// Test HermitianC method
TEST(SparseImplTests, HermitianCTest) {
  t_hostMat data = {{{1.0, 2.0}, {0.0, 0.0}, {3.0, 4.0}},
                    {{0.0, 0.0}, {5.0, 6.0}, {0.0, 0.0}}};
  SparseImpl A(data);
  SparseImpl B = A.HermitianC();

  EXPECT_EQ(B.DimX, 3);
  EXPECT_EQ(B.DimY, 2);
  EXPECT_EQ(B.Data.size(), 3);

  // Check that the values are correctly conjugated and transposed
  EXPECT_EQ(B.Data[0].Coords[0], 0);
  EXPECT_EQ(B.Data[0].Coords[1], 0);
  EXPECT_EQ(B.Data[0].Val, std::complex<double>(1.0, -2.0));

  EXPECT_EQ(B.Data[1].Coords[0], 1);
  EXPECT_EQ(B.Data[1].Coords[1], 1);
  EXPECT_EQ(B.Data[1].Val, std::complex<double>(5.0, -6.0));

  EXPECT_EQ(B.Data[2].Coords[0], 2);
  EXPECT_EQ(B.Data[2].Coords[1], 0);
  EXPECT_EQ(B.Data[2].Val, std::complex<double>(3.0, -4.0));
}

// Test ToDense method
TEST(SparseImplTests, ToDenseTest) {
  t_hostMat data = {{{1.0, 2.0}, {0.0, 0.0}, {3.0, 4.0}},
                    {{0.0, 0.0}, {5.0, 6.0}, {0.0, 0.0}}};
  SparseImpl A(data);
  Dense B = A.ToDense();

  EXPECT_EQ(B.DimX(), 2);
  EXPECT_EQ(B.DimY(), 3);

  // Check that the values are correctly converted to dense format
  EXPECT_EQ(B.GetData(0, 0), std::complex<double>(1.0, 2.0));
  EXPECT_EQ(B.GetData(0, 1), std::complex<double>(0.0, 0.0));
  EXPECT_EQ(B.GetData(0, 2), std::complex<double>(3.0, 4.0));
  EXPECT_EQ(B.GetData(1, 0), std::complex<double>(0.0, 0.0));
  EXPECT_EQ(B.GetData(1, 1), std::complex<double>(5.0, 6.0));
  EXPECT_EQ(B.GetData(1, 2), std::complex<double>(0.0, 0.0));
}

// Test SortByRow method
TEST(SparseImplTests, SortByRowTest) {
  // Create a matrix with elements in random order
  std::vector<COOTuple> data = {COOTuple(1, 1, std::complex<double>(5.0, 6.0)),
                                COOTuple(0, 0, std::complex<double>(1.0, 2.0)),
                                COOTuple(0, 2, std::complex<double>(3.0, 4.0))};

  SparseImpl A(2, 3);
  A.Data = data;
  A.SortByRow();

  // Check that the elements are sorted by row and column
  EXPECT_EQ(A.Data[0].Coords[0], 0);
  EXPECT_EQ(A.Data[0].Coords[1], 0);
  EXPECT_EQ(A.Data[0].Val, std::complex<double>(1.0, 2.0));

  EXPECT_EQ(A.Data[1].Coords[0], 0);
  EXPECT_EQ(A.Data[1].Coords[1], 2);
  EXPECT_EQ(A.Data[1].Val, std::complex<double>(3.0, 4.0));

  EXPECT_EQ(A.Data[2].Coords[0], 1);
  EXPECT_EQ(A.Data[2].Coords[1], 1);
  EXPECT_EQ(A.Data[2].Val, std::complex<double>(5.0, 6.0));
}

// Test GetRows method
TEST(SparseImplTests, GetRowsTest) {
  t_hostMat data = {{{1.0, 2.0}, {0.0, 0.0}, {3.0, 4.0}},
                    {{0.0, 0.0}, {5.0, 6.0}, {0.0, 0.0}}};
  SparseImpl A(data);
  std::vector<CompressedRow> rows = A.GetRows();

  EXPECT_EQ(rows.size(), 2);

  // Check the first row
  EXPECT_EQ(rows[0].Index, 0);
  EXPECT_EQ(rows[0].RowData.size(), 2);
  EXPECT_EQ(rows[0].RowData[0].Coords[0], 0);
  EXPECT_EQ(rows[0].RowData[0].Coords[1], 0);
  EXPECT_EQ(rows[0].RowData[0].Val, std::complex<double>(1.0, 2.0));
  EXPECT_EQ(rows[0].RowData[1].Coords[0], 0);
  EXPECT_EQ(rows[0].RowData[1].Coords[1], 2);
  EXPECT_EQ(rows[0].RowData[1].Val, std::complex<double>(3.0, 4.0));

  // Check the second row
  EXPECT_EQ(rows[1].Index, 1);
  EXPECT_EQ(rows[1].RowData.size(), 1);
  EXPECT_EQ(rows[1].RowData[0].Coords[0], 1);
  EXPECT_EQ(rows[1].RowData[0].Coords[1], 1);
  EXPECT_EQ(rows[1].RowData[0].Val, std::complex<double>(5.0, 6.0));
}

// Test GetCols method
// TEST(SparseImplTests, GetColsTest) {
//   t_hostMat data = {{{1.0, 2.0}, {0.0, 0.0}, {3.0, 4.0}},
//                     {{0.0, 0.0}, {5.0, 6.0}, {0.0, 0.0}}};
//   SparseImpl A(data);
//   std::vector<CompressedRow> cols = A.GetCols();

//   EXPECT_EQ(cols.size(), 3);

//   // Check the first column
//   EXPECT_EQ(cols[0].Index, 0);
//   EXPECT_EQ(cols[0].RowData.size(), 1);
//   EXPECT_EQ(cols[0].RowData[0].Coords[0], 0);
//   EXPECT_EQ(cols[0].RowData[0].Coords[1], 0);
//   EXPECT_EQ(cols[0].RowData[0].Val, std::complex<double>(1.0, 2.0));

//   // Check the second column
//   EXPECT_EQ(cols[1].Index, 1);
//   EXPECT_EQ(cols[1].RowData.size(), 1);
//   EXPECT_EQ(cols[1].RowData[0].Coords[0], 1);
//   EXPECT_EQ(cols[1].RowData[0].Coords[1], 1);
//   EXPECT_EQ(cols[1].RowData[0].Val, std::complex<double>(5.0, 6.0));

//   // Check the third column
//   EXPECT_EQ(cols[2].Index, 2);
//   EXPECT_EQ(cols[2].RowData.size(), 1);
//   EXPECT_EQ(cols[2].RowData[0].Coords[0], 0);
//   EXPECT_EQ(cols[2].RowData[0].Coords[1], 2);
//   EXPECT_EQ(cols[2].RowData[0].Val, std::complex<double>(3.0, 4.0));
// }

// Test SparseVectorSum function
TEST(SparseImplTests, SparseVectorSumTest) {
  CompressedRow A(0);
  A.RowData.push_back(COOTuple(0, 0, std::complex<double>(1.0, 2.0)));
  A.RowData.push_back(COOTuple(0, 2, std::complex<double>(3.0, 4.0)));

  CompressedRow B(0);
  B.RowData.push_back(COOTuple(0, 1, std::complex<double>(5.0, 6.0)));
  B.RowData.push_back(COOTuple(0, 2, std::complex<double>(7.0, 8.0)));

  CompressedRow C = SparseVectorSum(A, B);

  EXPECT_EQ(C.Index, 0);
  EXPECT_EQ(C.RowData.size(), 3);

  // Check the first element
  EXPECT_EQ(C.RowData[0].Coords[0], 0);
  EXPECT_EQ(C.RowData[0].Coords[1], 0);
  EXPECT_EQ(C.RowData[0].Val, std::complex<double>(1.0, 2.0));

  // Check the second element
  EXPECT_EQ(C.RowData[1].Coords[0], 0);
  EXPECT_EQ(C.RowData[1].Coords[1], 1);
  EXPECT_EQ(C.RowData[1].Val, std::complex<double>(5.0, 6.0));

  // Check the third element
  EXPECT_EQ(C.RowData[2].Coords[0], 0);
  EXPECT_EQ(C.RowData[2].Coords[1], 2);
  EXPECT_EQ(C.RowData[2].Val, std::complex<double>(10.0, 12.0));
}

// Test SparseDot function
TEST(SparseImplTests, SparseDotTest) {
  auto a = std::complex<double>(1.0, 2.0);
  auto b = std::complex<double>(3.0, 4.0);
  auto c = std::complex<double>(5.0, 6.0);
  auto d = std::complex<double>(7.0, 8.0);
  auto e = std::complex<double>(9.0, 10.0);
  CompressedRow A(0);
  A.RowData.push_back(COOTuple(0, 0, a));
  A.RowData.push_back(COOTuple(0, 2, b));

  CompressedRow B(0);
  B.RowData.push_back(COOTuple(0, 0, c));
  B.RowData.push_back(COOTuple(0, 2, d));
  B.RowData.push_back(COOTuple(0, 1, e));
  std::complex<double> dot = SparseDot(A, B);

  EXPECT_EQ(dot, a * c + b * d);
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
  EXPECT_EQ(C.Data.size(), 3);

  // Check the elements
  bool found1 = false, found2 = false, found3 = false;

  for (const auto &elem : C.Data) {
    if (elem.Coords[0] == 0 && elem.Coords[1] == 0) {
      EXPECT_EQ(elem.Val, std::complex<double>(-9.0, 22.0));
      found1 = true;
    } else if (elem.Coords[0] == 0 && elem.Coords[1] == 1) {
      EXPECT_EQ(elem.Val, std::complex<double>(-15.0, 80.0));
      found2 = true;
    } else if (elem.Coords[0] == 1 && elem.Coords[1] == 1) {
      EXPECT_EQ(elem.Val, std::complex<double>(-15.0, 104.0));
      found3 = true;
    }
  }

  EXPECT_TRUE(found1);
  EXPECT_TRUE(found2);
  EXPECT_TRUE(found3);
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
  EXPECT_EQ(C.Data.size(), 5);

  // Check that all elements are present
  bool found1 = false, found2 = false, found3 = false, found4 = false,
       found5 = false;

  for (const auto &elem : C.Data) {
    if (elem.Coords[0] == 0 && elem.Coords[1] == 0) {
      EXPECT_EQ(elem.Val, std::complex<double>(1.0, 2.0));
      found1 = true;
    } else if (elem.Coords[0] == 0 && elem.Coords[1] == 1) {
      EXPECT_EQ(elem.Val, std::complex<double>(7.0, 8.0));
      found2 = true;
    } else if (elem.Coords[0] == 0 && elem.Coords[1] == 2) {
      EXPECT_EQ(elem.Val, std::complex<double>(3.0, 4.0));
      found3 = true;
    } else if (elem.Coords[0] == 1 && elem.Coords[1] == 0) {
      EXPECT_EQ(elem.Val, std::complex<double>(9.0, 10.0));
      found4 = true;
    } else if (elem.Coords[0] == 1 && elem.Coords[1] == 1) {
      EXPECT_EQ(elem.Val, std::complex<double>(5.0, 6.0));
      found5 = true;
    } else if (elem.Coords[0] == 1 && elem.Coords[1] == 2) {
      EXPECT_EQ(elem.Val, std::complex<double>(11.0, 12.0));
    }
  }

  EXPECT_TRUE(found1);
  EXPECT_TRUE(found2);
  EXPECT_TRUE(found3);
  EXPECT_TRUE(found4);
  EXPECT_TRUE(found5);
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
  EXPECT_EQ(C.Data.size(), 5);

  // Check that all elements are present
  bool found1 = false, found2 = false, found3 = false, found4 = false,
       found5 = false;

  for (const auto &elem : C.Data) {
    if (elem.Coords[0] == 0 && elem.Coords[1] == 0) {
      EXPECT_EQ(elem.Val, std::complex<double>(1.0, 2.0));
      found1 = true;
    } else if (elem.Coords[0] == 0 && elem.Coords[1] == 1) {
      EXPECT_EQ(elem.Val, std::complex<double>(-7.0, -8.0));
      found2 = true;
    } else if (elem.Coords[0] == 0 && elem.Coords[1] == 2) {
      EXPECT_EQ(elem.Val, std::complex<double>(3.0, 4.0));
      found3 = true;
    } else if (elem.Coords[0] == 1 && elem.Coords[1] == 0) {
      EXPECT_EQ(elem.Val, std::complex<double>(-9.0, -10.0));
      found4 = true;
    } else if (elem.Coords[0] == 1 && elem.Coords[1] == 1) {
      EXPECT_EQ(elem.Val, std::complex<double>(5.0, 6.0));
      found5 = true;
    } else if (elem.Coords[0] == 1 && elem.Coords[1] == 2) {
      EXPECT_EQ(elem.Val, std::complex<double>(-11.0, -12.0));
    }
  }

  EXPECT_TRUE(found1);
  EXPECT_TRUE(found2);
  EXPECT_TRUE(found3);
  EXPECT_TRUE(found4);
  EXPECT_TRUE(found5);
}

// Test operator* method for scalar multiplication
TEST(SparseImplTests, ScalarMultiplicationOperatorTest) {
  t_hostMat data = {{{1.0, 2.0}, {0.0, 0.0}, {3.0, 4.0}},
                    {{0.0, 0.0}, {5.0, 6.0}, {0.0, 0.0}}};
  SparseImpl A(data);
  SparseImpl B = A * std::complex<double>(2.0, 0.0);

  EXPECT_EQ(B.DimX, 2);
  EXPECT_EQ(B.DimY, 3);
  EXPECT_EQ(B.Data.size(), 3);

  // Check that the values are correctly scaled
  EXPECT_EQ(B.Data[0].Val, std::complex<double>(2.0, 4.0));
  EXPECT_EQ(B.Data[1].Val, std::complex<double>(6.0, 8.0));
  EXPECT_EQ(B.Data[2].Val, std::complex<double>(10.0, 12.0));
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
  EXPECT_EQ(C.Data.size(), 3);

  // Check the elements
  bool found1 = false, found2 = false, found3 = false;

  for (const auto &elem : C.Data) {
    if (elem.Coords[0] == 0 && elem.Coords[1] == 0) {
      EXPECT_EQ(elem.Val, std::complex<double>(-9.0, 22.0));
      found1 = true;
    } else if (elem.Coords[0] == 0 && elem.Coords[1] == 1) {
      EXPECT_EQ(elem.Val, std::complex<double>(-15.0, 80.0));
      found2 = true;
    } else if (elem.Coords[0] == 1 && elem.Coords[1] == 1) {
      EXPECT_EQ(elem.Val, std::complex<double>(-15.0, 104.0));
      found3 = true;
    }
  }

  EXPECT_TRUE(found1);
  EXPECT_TRUE(found2);
  EXPECT_TRUE(found3);
}

// Test NNZ method
TEST(SparseImplTests, NNZTest) {
  t_hostMat data = {{{1.0, 2.0}, {0.0, 0.0}, {3.0, 4.0}},
                    {{0.0, 0.0}, {5.0, 6.0}, {0.0, 0.0}}};
  SparseImpl A(data);

  EXPECT_EQ(A.NNZ(), 3);
}