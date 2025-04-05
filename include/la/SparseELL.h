//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by Conor Stevenson on 15/3/2025.
//

#pragma once

#include <memory>
#include <vector>

#include "core/types.h"

class SparseELLImpl;

/**
 * @brief Sparse matrix class using ELLPACK format.
 */
class SparseELL {
 public:
  /**
   * @brief Default constructor for SparseELL.
   */
  SparseELL() noexcept;

  /**
   * @brief Constructor to initialize SparseELL matrix with given dimensions.
   *
   * @param dimX Number of rows.
   * @param dimY Number of columns.
   * @param maxNnzPerRow Maximum number of non-zero elements per row.
   */
  SparseELL(int dimX, int dimY, int maxNnzPerRow);

  /**
   * @brief Destructor for SparseELL.
   */
  ~SparseELL() noexcept;

  /**
   * @brief Copy constructor for SparseELL.
   *
   * @param other Another SparseELL object to copy from.
   */
  SparseELL(const SparseELL &other) noexcept;

  /**
   * @brief Move constructor for SparseELL.
   *
   * @param other Another SparseELL object to move from.
   */
  SparseELL(SparseELL &&other) noexcept;

  /**
   * @brief Copy assignment operator for SparseELL.
   *
   * @param other Another SparseELL object to copy from.
   * @return SparseELL& Reference to the current object.
   */
  SparseELL &operator=(const SparseELL &other) noexcept;

  /**
   * @brief Adds two SparseELL matrices.
   *
   * @param A Another SparseELL object to add.
   * @return SparseELL Result of the addition.
   */
  SparseELL Add(const SparseELL &A) const;

  /**
   * @brief Multiplies two SparseELL matrices.
   *
   * @param A Another SparseELL object to multiply.
   * @return SparseELL Result of the multiplication.
   */
  SparseELL RightMult(const SparseELL &A) const;

  /**
   * @brief Scales the SparseELL matrix by a scalar value.
   *
   * @param alpha Scalar value to multiply.
   * @return SparseELL Result of the scalar multiplication.
   */
  SparseELL Scale(t_cplx alpha) const noexcept;

  /**
   * @brief Transposes the SparseELL matrix.
   *
   * @return SparseELL Transposed matrix.
   */
  SparseELL Transpose() const noexcept;

  /**
   * @brief Computes the Hermitian conjugate of the SparseELL matrix.
   *
   * @return SparseELL Hermitian conjugate matrix.
   */
  SparseELL HermitianC() const noexcept;

  /**
   * @brief Prints the SparseELL matrix.
   *
   * @param kind Type of data to print (real, imaginary, etc.).
   * @param prec Precision of the printed data.
   */
  void Print(unsigned int kind, unsigned int prec) const noexcept;

  /**
   * @brief Get the number of rows in the SparseELL matrix.
   *
   * @return int Number of rows.
   */
  int DimX() const;

  /**
   * @brief Get the number of columns in the SparseELL matrix.
   *
   * @return int Number of columns.
   */
  int DimY() const;

  /**
   * @brief Get the maximum number of non-zero elements per row.
   *
   * @return int Maximum number of non-zero elements per row.
   */
  int MaxNnzPerRow() const;

  /**
   * @brief Gets the host data of the SparseELL matrix.
   *
   * @return const std::vector<t_hostVect>& Reference to the host data.
   */
  const std::vector<t_hostVect> &GetHostData() const;

 private:
  std::unique_ptr<SparseELLImpl> pImpl;  ///< Pointer to implementation
  SparseELL(std::unique_ptr<SparseELLImpl> pImpl);  ///< Private constructor
};
