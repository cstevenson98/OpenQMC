#include <complex>
#include <gtest/gtest.h>
#include <thrust/complex.h>

#include "la/VectImplGPU.cuh"

TEST(VectImplGPUTests, DefaultConstructorTest) {
  VectImplGPU A;
  EXPECT_EQ(A.size(), 0);
}

TEST(VectImplGPUTests, ConstructorWithSizeTest) {
  VectImplGPU A(5);
  EXPECT_EQ(A.size(), 5);

  // Check all elements are zero
  for (int i = 0; i < A.size(); ++i) {
    EXPECT_EQ(A[i], std::complex<double>(0.0, 0.0));
  }
}

TEST(VectImplGPUTests, ConstructorWithDataTest) {
  std::vector<std::complex<double>> data = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
  VectImplGPU A(data);
  EXPECT_EQ(A.size(), 3);
  EXPECT_EQ(A[0], std::complex<double>(1.0, 2.0));
  EXPECT_EQ(A[1], std::complex<double>(3.0, 4.0));
  EXPECT_EQ(A[2], std::complex<double>(5.0, 6.0));
}

TEST(VectImplGPUTests, AdditionTest) {
  std::vector<std::complex<double>> data1 = {
      {1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
  std::vector<std::complex<double>> data2 = {
      {7.0, 8.0}, {9.0, 10.0}, {11.0, 12.0}};
  VectImplGPU A(data1);
  VectImplGPU B(data2);
  VectImplGPU C = A + B;
  EXPECT_EQ(C.size(), 3);
  EXPECT_EQ(C[0], std::complex<double>(8.0, 10.0));
  EXPECT_EQ(C[1], std::complex<double>(12.0, 14.0));
  EXPECT_EQ(C[2], std::complex<double>(16.0, 18.0));
}

TEST(VectImplGPUTests, SubtractionTest) {
  std::vector<std::complex<double>> data1 = {
      {7.0, 8.0}, {9.0, 10.0}, {11.0, 12.0}};
  std::vector<std::complex<double>> data2 = {
      {1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
  VectImplGPU A(data1);
  VectImplGPU B(data2);
  VectImplGPU C = A - B;
  EXPECT_EQ(C.size(), 3);
  EXPECT_EQ(C[0], std::complex<double>(6.0, 6.0));
  EXPECT_EQ(C[1], std::complex<double>(6.0, 6.0));
  EXPECT_EQ(C[2], std::complex<double>(6.0, 6.0));
}

TEST(VectImplGPUTests, ScalarMultiplicationTest) {
  std::vector<std::complex<double>> data = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
  VectImplGPU A(data);
  VectImplGPU B = A * std::complex<double>(2.0, 0.0);
  EXPECT_EQ(B.size(), 3);
  EXPECT_EQ(B[0], std::complex<double>(2.0, 4.0));
  EXPECT_EQ(B[1], std::complex<double>(6.0, 8.0));
  EXPECT_EQ(B[2], std::complex<double>(10.0, 12.0));
}

TEST(VectImplGPUTests, ConjugateTest) {
  std::vector<std::complex<double>> data = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
  VectImplGPU A(data);
  VectImplGPU B = A.Conj();
  EXPECT_EQ(B.size(), 3);
  EXPECT_EQ(B[0], std::complex<double>(1.0, -2.0));
  EXPECT_EQ(B[1], std::complex<double>(3.0, -4.0));
  EXPECT_EQ(B[2], std::complex<double>(5.0, -6.0));
}

TEST(VectImplGPUTests, DotProductTest) {
  std::vector<std::complex<double>> data1 = {
      {1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
  std::vector<std::complex<double>> data2 = {
      {7.0, 8.0}, {9.0, 10.0}, {11.0, 12.0}};
  VectImplGPU A(data1);
  VectImplGPU B(data2);

  std::complex<double> expected = std::conj(data1[0]) * data2[0] +
                                  std::conj(data1[1]) * data2[1] +
                                  std::conj(data1[2]) * data2[2];
  std::complex<double> result = A.Dot(B);
  EXPECT_EQ(result, expected);
}

TEST(VectImplGPUTests, NormTest) {
  std::vector<std::complex<double>> data = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
  VectImplGPU A(data);
  double expected =
      std::sqrt(std::norm(data[0]) + std::norm(data[1]) + std::norm(data[2]));
  double result = A.Norm();
  EXPECT_EQ(result, expected);
}

TEST(VectImplGPUTests, GetDataTest) {
  std::vector<std::complex<double>> data = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
  VectImplGPU A(data);
  auto retrieved_data = A.GetData();
  EXPECT_EQ(retrieved_data.size(), 3);
  EXPECT_EQ(retrieved_data[0], data[0]);
  EXPECT_EQ(retrieved_data[1], data[1]);
  EXPECT_EQ(retrieved_data[2], data[2]);
}

TEST(VectImplGPUTests, SetDataTest) {
  std::vector<std::complex<double>> data = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
  VectImplGPU A;
  A.SetData(data);
  EXPECT_EQ(A.size(), 3);
  EXPECT_EQ(A[0], data[0]);
  EXPECT_EQ(A[1], data[1]);
  EXPECT_EQ(A[2], data[2]);
}

TEST(VectImplGPUTests, PrintTest) {
  std::vector<std::complex<double>> data = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
  VectImplGPU A(data);
  // This test is just to ensure that the Print functions do not crash
  EXPECT_NO_THROW(A.Print(0));
  EXPECT_NO_THROW(A.PrintRe());
  EXPECT_NO_THROW(A.PrintIm());
  EXPECT_NO_THROW(A.PrintAbs());
}