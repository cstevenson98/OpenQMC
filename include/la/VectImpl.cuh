//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by Conor Stevenson on 15/3/2025.
//

#pragma once
#include "core/eigen_types.h"
#include "core/types.cuh"
#include "la/Vect.h"

/**
 * @brief A class representing the implementation of vectors. Supports vector
 * algebra and uses complex numbers.
 */
class VectImpl {
public:
  /**
   * @brief Default constructor to initialize an empty VectImpl.
   */
  explicit VectImpl() = default;

  /**
   * @brief Constructor to initialize VectImpl with given size.
   *
   * @param size Size of the vector.
   */
  VectImpl(unsigned int size) : Data(t_eigenVect::Zero(size)) {}

  /**
   * @brief Constructor to initialize VectImpl with given data.
   *
   * @param in Input vector data.
   */
  explicit VectImpl(t_hostVect &in) {
    Data.resize(in.size());
    for (size_t i = 0; i < in.size(); ++i) {
      Data(i) = in[i];
    }
  }

  /**
   * @brief Gets the host data of the VectImpl.
   *
   * @return const t_hostVect& Reference to the host data.
   */
  const t_hostVect GetHostData() const {
    static t_hostVect hostData;
    hostData.resize(Data.size());
    for (int i = 0; i < Data.size(); ++i) {
      hostData[i] = Data(i);
    }
    return hostData;
  }

  /**
   * @brief Computes the conjugate of the VectImpl.
   *
   * @return VectImpl Conjugated vector.
   */
  VectImpl Conj() const;

  /**
   * @brief Adds two VectImpl objects.
   *
   * @param A Another VectImpl object to add.
   * @return VectImpl Result of the addition.
   */
  VectImpl Add(const VectImpl &A) const;

  /**
   * @brief Subtracts one VectImpl object from another.
   *
   * @param A Another VectImpl object to subtract.
   * @return VectImpl Result of the subtraction.
   */
  VectImpl Subtract(const VectImpl &A) const;

  /**
   * @brief Scales the VectImpl by a scalar value.
   *
   * @param alpha Scalar value to multiply.
   * @return VectImpl Result of the scalar multiplication.
   */
  VectImpl Scale(const th_cplx &alpha) const;

  /**
   * @brief Computes the dot product of two VectImpl objects.
   *
   * @param A Another VectImpl object.
   * @return double Dot product result.
   */
  std::complex<double> Dot(const VectImpl &A) const;

  /**
   * @brief Computes the norm of the VectImpl.
   *
   * @return double Norm of the vector.
   */
  double Norm() const;

  /**
   * @brief Overloaded addition operator for VectImpl objects.
   *
   * @param A Another VectImpl object to add.
   * @return VectImpl Result of the addition.
   */
  VectImpl operator+(const VectImpl &A) const;

  /**
   * @brief Overloaded subtraction operator for VectImpl objects.
   *
   * @param A Another VectImpl object to subtract.
   * @return VectImpl Result of the subtraction.
   */
  VectImpl operator-(const VectImpl &A) const;

  /**
   * @brief Overloaded multiplication operator for scalar multiplication.
   *
   * @param alpha Scalar value to multiply.
   * @return VectImpl Result of the scalar multiplication.
   */
  VectImpl operator*(const th_cplx &alpha) const;

  /**
   * @brief Overloaded subscript operator to access vector elements.
   *
   * @param i Index of the element.
   * @return std::complex<double> Element at the specified position.
   */
  std::complex<double> operator[](unsigned int i) const;

  /**
   * @brief Prints the VectImpl.
   *
   * @param kind Type of data to print (real, imaginary, etc.).
   */
  void Print(unsigned int kind) const;

  /**
   * @brief Prints the real part of the VectImpl.
   */
  void PrintRe() const;

  /**
   * @brief Prints the imaginary part of the VectImpl.
   */
  void PrintIm() const;

  /**
   * @brief Prints the absolute value of the VectImpl.
   */
  void PrintAbs() const;

  /**
   * @brief Gets the data of the VectImpl.
   *
   * @return std::vector<std::complex<double>> Data of the vector.
   */
  std::vector<std::complex<double>> GetData() const;

  /**
   * @brief Sets the data of the VectImpl.
   *
   * @param data Data to set.
   */
  void SetData(const std::vector<std::complex<double>> &data);

  /**
   * @brief Gets the size of the VectImpl.
   *
   * @return int Size of the vector.
   */
  int size() const;

private:
  t_eigenVect Data; ///< Vector data using Eigen
};

/**
 * @brief Overloaded multiplication operator for scalar multiplication.
 *
 * @param alpha Scalar value to multiply.
 * @param rhs VectImpl object to multiply.
 * @return VectImpl Result of the scalar multiplication.
 */
VectImpl operator*(const th_cplx &alpha, const VectImpl &rhs);
