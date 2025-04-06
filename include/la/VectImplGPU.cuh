//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by Conor Stevenson on 15/3/2025.
//

#pragma once

#include <cusparse.h>
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

#include "core/types.cuh"
#include "la/CuSparseSingleton.cuh"
#include "la/Vect.h"
#include <complex>

/**
 * @brief A class representing the GPU implementation of vectors using Thrust.
 * Supports vector algebra and uses complex numbers.
 */
class VectImplGPU {
public:
  int Dim; ///< Dimension of the vector

  // Device data
  t_devcVect deviceData_; ///< Device vector data

  // cuSPARSE descriptor
  cusparseDnVecDescr_t vecDescr_; ///< cuSPARSE dense vector descriptor

  /**
   * @brief Constructor to initialize VectImplGPU vector with given dimension.
   *
   * @param dim Dimension of the vector.
   */
  VectImplGPU(int dim);

  /**
   * @brief Destructor to clean up cuSPARSE resources.
   */
  ~VectImplGPU();

  /**
   * @brief Constructor to initialize VectImplGPU vector from a host vector.
   *
   * @param in Host vector to initialize from.
   */
  explicit VectImplGPU(const t_hostVect &in);

  /**
   * @brief Constructor to initialize VectImplGPU vector from a CPU VectImpl.
   *
   * @param cpuVector CPU vector to initialize from.
   */
  explicit VectImplGPU(const class VectImpl &cpuVector);

  /**
   * @brief Gets the host data of the VectImplGPU.
   *
   * @return t_hostVect Host data.
   */
  t_hostVect GetHostData() const;

  /**
   * @brief Computes the conjugate of the VectImplGPU.
   *
   * @return VectImplGPU Conjugated vector.
   */
  VectImplGPU Conj() const;

  /**
   * @brief Computes the conjugate of the VectImplGPU and stores result in
   * output.
   *
   * @param output Vector to store the result.
   */
  void ConjInPlace(VectImplGPU &output) const;

  /**
   * @brief Adds two VectImplGPU objects.
   *
   * @param A Another VectImplGPU object to add.
   * @return VectImplGPU Result of the addition.
   */
  VectImplGPU Add(const VectImplGPU &A) const;

  /**
   * @brief Adds two VectImplGPU objects and stores result in output.
   *
   * @param A Another VectImplGPU object to add.
   * @param output Vector to store the result.
   */
  void AddInPlace(const VectImplGPU &A, VectImplGPU &output) const;

  /**
   * @brief Subtracts one VectImplGPU object from another.
   *
   * @param A Another VectImplGPU object to subtract.
   * @return VectImplGPU Result of the subtraction.
   */
  VectImplGPU Subtract(const VectImplGPU &A) const;

  /**
   * @brief Subtracts one VectImplGPU object from another and stores result in
   * output.
   *
   * @param A Another VectImplGPU object to subtract.
   * @param output Vector to store the result.
   */
  void SubtractInPlace(const VectImplGPU &A, VectImplGPU &output) const;

  /**
   * @brief Scales the VectImplGPU by a scalar value.
   *
   * @param alpha Scalar value to multiply.
   * @return VectImplGPU Result of the scalar multiplication.
   */
  VectImplGPU Scale(const th_cplx &alpha) const;

  /**
   * @brief Scales the VectImplGPU by a scalar value and stores result in
   * output.
   *
   * @param alpha Scalar value to multiply.
   * @param output Vector to store the result.
   */
  void ScaleInPlace(const th_cplx &alpha, VectImplGPU &output) const;

  /**
   * @brief Computes the dot product of two VectImplGPU vectors.
   *
   * @param B Another VectImplGPU object to compute dot product with.
   * @return th_cplx Result of the dot product.
   */
  th_cplx Dot(const VectImplGPU &B) const;

  /**
   * @brief Computes the norm of the VectImplGPU vector.
   *
   * @return double The norm of the vector.
   */
  double Norm() const;

  /**
   * @brief Overloaded addition operator for VectImplGPU vectors.
   *
   * @param A Another VectImplGPU object to add.
   * @return VectImplGPU Result of the addition.
   */
  VectImplGPU operator+(const VectImplGPU &A) const;

  /**
   * @brief Overloaded subtraction operator for VectImplGPU vectors.
   *
   * @param A Another VectImplGPU object to subtract.
   * @return VectImplGPU Result of the subtraction.
   */
  VectImplGPU operator-(const VectImplGPU &A) const;

  /**
   * @brief Overloaded multiplication operator for scalar multiplication.
   *
   * @param alpha Scalar value to multiply.
   * @return VectImplGPU Result of the scalar multiplication.
   */
  VectImplGPU operator*(const th_cplx &alpha) const;

  /**
   * @brief Overloaded multiplication operator for VectImplGPU vectors.
   *
   * @param A Another VectImplGPU object to multiply.
   * @return th_cplx Result of the multiplication.
   */
  th_cplx operator*(const VectImplGPU &A) const;

  /**
   * @brief Overloaded subscript operator to access vector elements.
   *
   * @param i Index of the element.
   * @return std::complex<double> Element at the specified position.
   */
  std::complex<double> operator[](unsigned int i) const;

  /**
   * @brief Prints the VectImplGPU.
   *
   * @param kind Type of data to print (real, imaginary, etc.).
   */
  void Print(unsigned int kind) const;

  /**
   * @brief Prints the real part of the VectImplGPU.
   */
  void PrintRe() const;

  /**
   * @brief Prints the imaginary part of the VectImplGPU.
   */
  void PrintIm() const;

  /**
   * @brief Prints the absolute value of the VectImplGPU.
   */
  void PrintAbs() const;

  /**
   * @brief Gets the data of the VectImplGPU.
   *
   * @return std::vector<std::complex<double>> Data of the vector.
   */
  std::vector<std::complex<double>> GetData() const;

  /**
   * @brief Sets the data of the VectImplGPU.
   *
   * @param data Data to set.
   */
  void SetData(const std::vector<std::complex<double>> &data);

  /**
   * @brief Copies data from a host vector.
   *
   * @param hostData Host vector data to copy from.
   */
  void CopyFromHost(const t_hostVect &hostData);

  /**
   * @brief Gets the size of the VectImplGPU vector.
   *
   * @return int The size of the vector.
   */
  int size() const { return Dim; }

  /**
   * @brief Gets the device data of the VectImplGPU vector.
   *
   * @return const t_devcVect& The device data.
   */
  const t_devcVect &GetDeviceData() const { return deviceData_; }

  /**
   * @brief Gets the cuSPARSE dense vector descriptor.
   *
   * @return cusparseDnVecDescr_t The cuSPARSE dense vector descriptor.
   */
  cusparseDnVecDescr_t GetVecDescr() const { return vecDescr_; }

  /**
   * @brief Gets the cuSPARSE handle from the singleton.
   *
   * @return cusparseHandle_t The cuSPARSE handle.
   */
  cusparseHandle_t GetHandle() const {
    return CuSparseSingleton::getInstance().getHandle();
  }

private:
  /**
   * @brief Initializes cuSPARSE resources.
   */
  void InitializeCuSparse();
};

/**
 * @brief Overloaded multiplication operator for scalar multiplication.
 *
 * @param alpha Scalar value to multiply.
 * @param rhs VectImplGPU object to multiply.
 * @return VectImplGPU Result of the scalar multiplication.
 */
VectImplGPU operator*(const th_cplx &alpha, const VectImplGPU &rhs);