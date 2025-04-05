//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by Conor Stevenson on 15/3/2025.
//

#include <vector>

#include "core/types.cuh"
#include "core/types.h"
#include "la/Dense.h"

class DenseImpl {
public:
  int DimX;          ///< Number of rows
  int DimY;          ///< Number of columns
  t_hostMat CPUData; ///< Matrix data stored in a 2D vector

  /**
   * @brief Default constructor for DenseImpl.
   */
  DenseImpl() noexcept;

  /**
   * @brief Destructor for DenseImpl.
   */
  ~DenseImpl() noexcept;

  /**
   * @brief Constructor to initialize DenseImpl matrix with given dimensions.
   *
   * @param dimX Number of rows.
   * @param dimY Number of columns.
   */
  DenseImpl(int dimX, int dimY);

  /**
   * @brief Copy constructor for DenseImpl.
   *
   * @param other Another DenseImpl object to copy from.
   */
  DenseImpl(const DenseImpl &other) noexcept;

  /**
   * @brief Move constructor for DenseImpl.
   *
   * @param other Another DenseImpl object to move from.
   */
  DenseImpl(DenseImpl &&other) noexcept;

  /**
   * @brief Constructor to initialize DenseImpl matrix with given data.
   *
   * @param in Input matrix data.
   */
  DenseImpl(t_hostMat &in) noexcept;

  /**
   * @brief Copy assignment operator for DenseImpl.
   *
   * @param other Another DenseImpl object to copy from.
   * @return DenseImpl& Reference to the current object.
   */
  DenseImpl &operator=(const DenseImpl &other) noexcept;

  /**
   * @brief Adds two DenseImpl matrices.
   *
   * @param A Another DenseImpl object to add.
   * @return DenseImpl Result of the addition.
   */
  DenseImpl Add(const DenseImpl &A) const;

  /**
   * @brief Multiplies two DenseImpl matrices.
   *
   * @param A Another DenseImpl object to multiply.
   * @return DenseImpl Result of the multiplication.
   */
  DenseImpl RightMult(const DenseImpl &A) const;

  /**
   * @brief Scales the DenseImpl matrix by a scalar value.
   *
   * @param alpha Scalar value to multiply.
   * @return DenseImpl Result of the scalar multiplication.
   */
  DenseImpl Scale(t_cplx alpha) const noexcept;

  /**
   * @brief Transposes the DenseImpl matrix.
   *
   * @return DenseImpl Transposed matrix.
   */
  DenseImpl Transpose() const noexcept;

  /**
   * @brief Computes the Hermitian conjugate of the DenseImpl matrix.
   *
   * @return DenseImpl Hermitian conjugate matrix.
   */
  DenseImpl HermitianC() const noexcept;

  /**
   * @brief Flattens the DenseImpl matrix data into a vector.
   *
   * @return t_hostVect Flattened data.
   */
  t_hostVect FlattenedData() const noexcept;

  /**
   * @brief Prints the DenseImpl matrix.
   *
   * @param kind Type of data to print (real, imaginary, etc.).
   * @param prec Precision of the printed data.
   */
  void Print(unsigned int kind, unsigned int prec) const noexcept;
};
