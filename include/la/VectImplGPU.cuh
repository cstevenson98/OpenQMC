//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by Conor Stevenson on 15/3/2025.
//

#pragma once

#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

#include "core/types.cuh"
#include "la/Vect.h"

/**
 * @brief A class representing the GPU implementation of vectors. Supports
 * vector algebra and uses complex numbers.
 */
class VectImplGPU {
public:
  /**
   * @brief Default constructor to initialize an empty VectImplGPU.
   */
  explicit VectImplGPU() = default;

  /**
   * @brief Constructor to initialize VectImplGPU with given size.
   *
   * @param size Size of the vector.
   */
  VectImplGPU(unsigned int size) : deviceData_(size, th_cplx(0.0, 0.0)) {}

  /**
   * @brief Constructor to initialize VectImplGPU with given data.
   *
   * @param in Input vector data.
   */
  explicit VectImplGPU(const t_hostVect &in) {
    // Convert std::complex to thrust::complex
    th_hostVect hostData(in.size());
    for (size_t i = 0; i < in.size(); ++i) {
      hostData[i] = th_cplx(in[i].real(), in[i].imag());
    }

    // Copy to device
    deviceData_ = hostData;
  }

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
   * @brief Adds two VectImplGPU objects.
   *
   * @param A Another VectImplGPU object to add.
   * @return VectImplGPU Result of the addition.
   */
  VectImplGPU Add(const VectImplGPU &A) const;

  /**
   * @brief Subtracts one VectImplGPU object from another.
   *
   * @param A Another VectImplGPU object to subtract.
   * @return VectImplGPU Result of the subtraction.
   */
  VectImplGPU Subtract(const VectImplGPU &A) const;

  /**
   * @brief Scales the VectImplGPU by a scalar value.
   *
   * @param alpha Scalar value to multiply.
   * @return VectImplGPU Result of the scalar multiplication.
   */
  VectImplGPU Scale(const th_cplx &alpha) const;

  /**
   * @brief Computes the dot product of two VectImplGPU objects.
   *
   * @param A Another VectImplGPU object.
   * @return double Dot product result.
   */
  std::complex<double> Dot(const VectImplGPU &A) const;

  /**
   * @brief Computes the norm of the VectImplGPU.
   *
   * @return double Norm of the vector.
   */
  double Norm() const;

  /**
   * @brief Overloaded addition operator for VectImplGPU objects.
   *
   * @param A Another VectImplGPU object to add.
   * @return VectImplGPU Result of the addition.
   */
  VectImplGPU operator+(const VectImplGPU &A) const;

  /**
   * @brief Overloaded subtraction operator for VectImplGPU objects.
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
   * @brief Gets the size of the VectImplGPU.
   *
   * @return int Size of the vector.
   */
  int size() const;

  /**
   * @brief Gets the device data pointer.
   *
   * @return th_cplx* Raw pointer to device data.
   */
  th_cplx *GetDeviceDataPtr() noexcept {
    return thrust::raw_pointer_cast(deviceData_.data());
  }

  /**
   * @brief Gets the device data.
   *
   * @return const t_devcVect& Reference to device data.
   */
  const t_devcVect &GetDeviceData() const { return deviceData_; }

private:
  t_devcVect deviceData_; ///< Vector data stored on device
};

/**
 * @brief Overloaded multiplication operator for scalar multiplication.
 *
 * @param alpha Scalar value to multiply.
 * @param rhs VectImplGPU object to multiply.
 * @return VectImplGPU Result of the scalar multiplication.
 */
VectImplGPU operator*(const th_cplx &alpha, const VectImplGPU &rhs);