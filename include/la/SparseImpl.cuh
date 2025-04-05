//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by Conor Stevenson on 03/04/2022.
//

#pragma once

#include "core/types.h"
#include "la/Dense.h"
#include "la/Sparse.h"
#include "la/VectImpl.cuh"
#include <complex>
#include <vector>

/**
 * @brief A class representing the implementation of sparse matrices in COO
 * format. Supports matrix algebra and uses complex numbers.
 */
class SparseImpl {
public:
  int DimX;                   ///< Number of rows
  int DimY;                   ///< Number of columns
  std::vector<COOTuple> Data; ///< Matrix data stored in COO format

  /**
   * @brief Constructor to initialize SparseImpl matrix with given dimensions.
   *
   * @param dimX Number of rows.
   * @param dimY Number of columns.
   */
  SparseImpl(int dimX, int dimY) : DimX(dimX), DimY(dimY) {};

  /**
   * @brief Constructor to initialize SparseImpl matrix from a host matrix.
   *
   * @param in Host matrix to initialize from.
   */
  explicit SparseImpl(const t_hostMat &in);

  /**
   * @brief Scales the SparseImpl matrix by a scalar value.
   *
   * @param alpha Scalar value to multiply.
   * @return SparseImpl Result of the scalar multiplication.
   */
  SparseImpl Scale(const t_cplx &alpha) const;

  /**
   * @brief Adds two SparseImpl matrices.
   *
   * @param B Another SparseImpl object to add.
   * @return SparseImpl Result of the addition.
   */
  SparseImpl Add(const SparseImpl &B) const;

  /**
   * @brief Multiplies two SparseImpl matrices.
   *
   * @param A Another SparseImpl object to multiply.
   * @return SparseImpl Result of the multiplication.
   */
  SparseImpl RightMult(const SparseImpl &A) const;

  /**
   * @brief Transposes the SparseImpl matrix.
   *
   * @return SparseImpl Transposed matrix.
   */
  SparseImpl Transpose() const;

  /**
   * @brief Computes the Hermitian conjugate of the SparseImpl matrix.
   *
   * @return SparseImpl Hermitian conjugate matrix.
   */
  SparseImpl HermitianC() const;

  /**
   * @brief Converts the SparseImpl matrix to a Dense matrix.
   *
   * @return Dense Dense matrix.
   */
  Dense ToDense();

  /**
   * @brief Sorts the SparseImpl matrix data by row.
   */
  void SortByRow();

  /**
   * @brief Trims the SparseImpl matrix data.
   */
  void Trim();

  /**
   * @brief Multiplies the SparseImpl matrix by a vector.
   *
   * @param vect Vector to multiply.
   * @return Vect Result of the multiplication.
   */
  VectImpl VectMult(const VectImpl &vect) const;

  /**
   * @brief Gets the rows of the matrix in compressed row format.
   *
   * @return std::vector<CompressedRow> Vector of compressed rows.
   */
  std::vector<CompressedRow> GetRows() const;

  /**
   * @brief Gets the columns of the matrix in compressed column format.
   *
   * @return std::vector<CompressedRow> Vector of compressed columns.
   */
  std::vector<CompressedRow> GetCols() const;

  /**
   * @brief Overloaded addition operator for SparseImpl matrices.
   *
   * @param A Another SparseImpl object to add.
   * @return SparseImpl Result of the addition.
   */
  SparseImpl operator+(const SparseImpl &A) const;

  /**
   * @brief Overloaded subtraction operator for SparseImpl matrices.
   *
   * @param A Another SparseImpl object to subtract.
   * @return SparseImpl Result of the subtraction.
   */
  SparseImpl operator-(const SparseImpl &A) const;

  /**
   * @brief Overloaded multiplication operator for scalar multiplication.
   *
   * @param alpha Scalar value to multiply.
   * @return SparseImpl Result of the scalar multiplication.
   */
  SparseImpl operator*(const t_cplx &alpha) const;

  /**
   * @brief Overloaded multiplication operator for SparseImpl matrices.
   *
   * @param A Another SparseImpl object to multiply.
   * @return SparseImpl Result of the multiplication.
   */
  SparseImpl operator*(const SparseImpl &A) const;

  /**
   * @brief Overloaded element-wise multiplication operator for SparseImpl
   * matrices.
   *
   * @param A Another SparseImpl object to multiply element-wise.
   * @return SparseImpl Result of the element-wise multiplication.
   */
  SparseImpl operator%(const SparseImpl &A) const;

  /**
   * @brief Prints the SparseImpl matrix.
   */
  void Print() const;

  /**
   * @brief Prints the real part of the SparseImpl matrix.
   */
  void PrintRe() const;

  /**
   * @brief Prints the imaginary part of the SparseImpl matrix.
   */
  void PrintIm() const;

  /**
   * @brief Prints the absolute value of the SparseImpl matrix.
   */
  void PrintAbs() const;

  /**
   * @brief Gets the number of non-zero elements in the SparseImpl matrix.
   *
   * @return unsigned int Number of non-zero elements.
   */
  unsigned int NNZ() const;

  /**
   * @brief Gets the host data of the SparseImpl matrix.
   *
   * @return const std::vector<COOTuple>& Reference to the COO data.
   */
  const t_hostMat GetHostData() const;

  /**
   * @brief Gets a reference to a coefficient in the matrix.
   *
   * @param i Row index.
   * @param j Column index.
   * @return std::complex<double>& Reference to the coefficient.
   */
  std::complex<double> &CoeffRef(int i, int j);
};