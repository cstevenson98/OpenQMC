//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by Conor Stevenson on 15/3/2025.
//

#pragma once

#include <vector>

#include "core/types.cuh"
#include "core/types.h"
#include "la/SparseELL.h"

class SparseELLImpl {
public:
  int DimX;         ///< Number of rows
  int DimY;         ///< Number of columns
  int MaxNnzPerRow; ///< Maximum number of non-zero elements per row
  std::vector<t_hostVect> CPUData; ///< Matrix data stored in a 2D vector
  std::vector<t_hostVect> CPUCol;  ///< Column indices stored in a 2D vector

  /**
   * @brief Default constructor for SparseELLImpl.
   */
  SparseELLImpl() noexcept;

  /**
   * @brief Destructor for SparseELLImpl.
   */
  ~SparseELLImpl() noexcept;

  /**
   * @brief Constructor to initialize SparseELLImpl matrix with given
   * dimensions.
   *
   * @param dimX Number of rows.
   * @param dimY Number of columns.
   * @param maxNnzPerRow Maximum number of non-zero elements per row.
   */
  SparseELLImpl(int dimX, int dimY, int maxNnzPerRow);

  /**
   * @brief Copy constructor for SparseELLImpl.
   *
   * @param other Another SparseELLImpl object to copy from.
   */
  SparseELLImpl(const SparseELLImpl &other) noexcept;

  /**
   * @brief Move constructor for SparseELLImpl.
   *
   * @param other Another SparseELLImpl object to move from.
   */
  SparseELLImpl(SparseELLImpl &&other) noexcept;

  /**
   * @brief Copy assignment operator for SparseELLImpl.
   *
   * @param other Another SparseELLImpl object to copy from.
   * @return SparseELLImpl& Reference to the current object.
   */
  SparseELLImpl &operator=(const SparseELLImpl &other) noexcept;

  /**
   * @brief Adds two SparseELLImpl matrices.
   *
   * @param A Another SparseELLImpl object to add.
   * @return SparseELLImpl Result of the addition.
   */
  SparseELLImpl Add(const SparseELLImpl &A) const;

  /**
   * @brief Multiplies two SparseELLImpl matrices.
   *
   * @param A Another SparseELLImpl object to multiply.
   * @return SparseELLImpl Result of the multiplication.
   */
  SparseELLImpl RightMult(const SparseELLImpl &A) const;

  /**
   * @brief Scales the SparseELLImpl matrix by a scalar value.
   *
   * @param alpha Scalar value to multiply.
   * @return SparseELLImpl Result of the scalar multiplication.
   */
  SparseELLImpl Scale(t_cplx alpha) const noexcept;

  /**
   * @brief Transposes the SparseELLImpl matrix.
   *
   * @return SparseELLImpl Transposed matrix.
   */
  SparseELLImpl Transpose() const noexcept;

  /**
   * @brief Computes the Hermitian conjugate of the SparseELLImpl matrix.
   *
   * @return SparseELLImpl Hermitian conjugate matrix.
   */
  SparseELLImpl HermitianC() const noexcept;

  /**
   * @brief Prints the SparseELLImpl matrix.
   *
   * @param kind Type of data to print (real, imaginary, etc.).
   * @param prec Precision of the printed data.
   */
  void Print(unsigned int kind, unsigned int prec) const noexcept;
};