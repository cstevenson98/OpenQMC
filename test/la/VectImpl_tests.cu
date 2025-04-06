#include <gtest/gtest.h>
#include <thrust/complex.h>

#include <complex>

#include "core/types.cuh"
#include "la/VectImpl.cuh"

TEST(VectImplTests, DefaultConstructorTest) {
  VectImpl A;
  EXPECT_EQ(A.size(), 0);
}

TEST(VectImplTests, ConstructorWithSizeTest) {
  VectImpl A(5);
  EXPECT_EQ(A.size(), 5);

  // Check all elements are zero
  for (int i = 0; i < A.size(); ++i) {
    EXPECT_EQ(A[i], std::complex<double>(0.0, 0.0));
  }
}

TEST(VectImplTests, ConstructorWithDataTest) {
  t_hostVect data = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
  VectImpl A(data);
  EXPECT_EQ(A.size(), 3);
  EXPECT_EQ(A[0], std::complex<double>(1.0, 2.0));
  EXPECT_EQ(A[1], std::complex<double>(3.0, 4.0));
  EXPECT_EQ(A[2], std::complex<double>(5.0, 6.0));
}

TEST(VectImplTests, AdditionTest) {
  t_hostVect data1 = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
  t_hostVect data2 = {{7.0, 8.0}, {9.0, 10.0}, {11.0, 12.0}};
  VectImpl A(data1);
  VectImpl B(data2);
  VectImpl C = A + B;
  EXPECT_EQ(C.size(), 3);
  EXPECT_EQ(C[0], std::complex<double>(8.0, 10.0));
  EXPECT_EQ(C[1], std::complex<double>(12.0, 14.0));
  EXPECT_EQ(C[2], std::complex<double>(16.0, 18.0));
}

TEST(VectImplTests, SubtractionTest) {
  t_hostVect data1 = {{7.0, 8.0}, {9.0, 10.0}, {11.0, 12.0}};
  t_hostVect data2 = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
  VectImpl A(data1);
  VectImpl B(data2);
  VectImpl C = A - B;
  EXPECT_EQ(C.size(), 3);
  EXPECT_EQ(C[0], std::complex<double>(6.0, 6.0));
  EXPECT_EQ(C[1], std::complex<double>(6.0, 6.0));
  EXPECT_EQ(C[2], std::complex<double>(6.0, 6.0));
}

TEST(VectImplTests, ScalarMultiplicationTest) {
  t_hostVect data = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
  VectImpl A(data);
  VectImpl B = A * std::complex<double>(2.0, 0.0);
  EXPECT_EQ(B.size(), 3);
  EXPECT_EQ(B[0], std::complex<double>(2.0, 4.0));
  EXPECT_EQ(B[1], std::complex<double>(6.0, 8.0));
  EXPECT_EQ(B[2], std::complex<double>(10.0, 12.0));
}

TEST(VectImplTests, ConjugateTest) {
  t_hostVect data = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
  VectImpl A(data);
  VectImpl B = A.Conj();
  EXPECT_EQ(B.size(), 3);
  EXPECT_EQ(B[0], std::complex<double>(1.0, -2.0));
  EXPECT_EQ(B[1], std::complex<double>(3.0, -4.0));
  EXPECT_EQ(B[2], std::complex<double>(5.0, -6.0));
}

TEST(VectImplTests, DotProductTest) {
  auto a1 = std::complex<double>(1.0, 2.0);
  auto a2 = std::complex<double>(3.0, 4.0);
  auto a3 = std::complex<double>(5.0, 6.0);

  auto b1 = std::complex<double>(7.0, 8.0);
  auto b2 = std::complex<double>(9.0, 10.0);
  auto b3 = std::complex<double>(11.0, 12.0);

  t_hostVect data1 = {a1, a2, a3};
  t_hostVect data2 = {b1, b2, b3};
  VectImpl A(data1);
  VectImpl B(data2);

  std::complex<double> dotProduct = A.Dot(B);
  EXPECT_EQ(dotProduct,
            std::conj(a1) * b1 + std::conj(a2) * b2 + std::conj(a3) * b3);
}

TEST(VectImplTests, NormTest) {
  auto a1 = std::complex<double>(1.0, 2.0);
  auto a2 = std::complex<double>(3.0, 4.0);
  auto a3 = std::complex<double>(5.0, 6.0);
  t_hostVect data = {a1, a2, a3};
  VectImpl A(data);
  double norm = A.Norm();
  EXPECT_DOUBLE_EQ(norm,
                   std::sqrt(std::norm(a1) + std::norm(a2) + std::norm(a3)));
}

TEST(VectImplTests, PrintTest) {
  t_hostVect data = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
  VectImpl A(data);
  // This test is just to ensure that the Print function does not crash
  EXPECT_NO_THROW(A.Print(0));
}

TEST(VectImplTests, GetDataTest) {
  t_hostVect data = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
  VectImpl A(data);
  auto dataRetrieved = A.GetData();
  EXPECT_EQ(dataRetrieved.size(), 3);
  EXPECT_EQ(dataRetrieved[0], std::complex<double>(1.0, 2.0));
  EXPECT_EQ(dataRetrieved[1], std::complex<double>(3.0, 4.0));
  EXPECT_EQ(dataRetrieved[2], std::complex<double>(5.0, 6.0));
}

TEST(VectImplTests, GetHostDataTest) {
  t_hostVect data = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
  VectImpl A(data);
  auto hostData = A.GetHostData();
  EXPECT_EQ(hostData.size(), 3);
  EXPECT_EQ(hostData[0], std::complex<double>(1.0, 2.0));
  EXPECT_EQ(hostData[1], std::complex<double>(3.0, 4.0));
  EXPECT_EQ(hostData[2], std::complex<double>(5.0, 6.0));
}

TEST(VectImplTests, SetDataTest) {
  VectImpl A;
  std::vector<std::complex<double>> data = {std::complex<double>(1.0, 2.0),
                                            std::complex<double>(3.0, 4.0),
                                            std::complex<double>(5.0, 6.0)};
  A.SetData(data);
  EXPECT_EQ(A.size(), 3);
  EXPECT_EQ(A[0], std::complex<double>(1.0, 2.0));
  EXPECT_EQ(A[1], std::complex<double>(3.0, 4.0));
  EXPECT_EQ(A[2], std::complex<double>(5.0, 6.0));
}