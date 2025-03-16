//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by Conor Stevenson on 03/04/2022.
//

#ifndef MAIN_DENSE_CUH
#define MAIN_DENSE_CUH

#include <complex>
#include <memory>
#include "core/types.h"

class DenseImpl;

/**
 * @brief A class representing dense matrices. Supports matrix algebra and uses
 * complex numbers.
 */
class Dense {
public:
/**
 * @brief Default constructor to initialize an empty Dense matrix.
 */
Dense() noexcept;

  /**
   * @brief Constructor to initialize Dense matrix with given dimensions.
   * 
   * @param dimX Number of rows.
   * @param dimY Number of columns.
   */
  Dense(int dimX, int dimY);

  /**
   * @brief Constructor to initialize Dense matrix with given data.
   * 
   * @param in Input matrix data.
   */
  explicit Dense(t_hostMat &in) noexcept;

  /**
   * @brief Destructor for Dense matrix.
   */
  ~Dense() noexcept;

  /**
   * @brief Copy constructor for Dense matrix.
   * 
   * @param other Another Dense object to copy from.
   */
  Dense(const Dense &other) noexcept;

  /**
   * @brief Move constructor for Dense matrix.
   * 
   * @param other Another Dense object to move from.
   */
  Dense(Dense &&other) noexcept;

  /**
   * @brief Copy assignment operator for Dense matrix.
   * 
   * @param other Another Dense object to copy from.
   * @return Dense& Reference to the current object.
   */
  Dense &operator=(const Dense &other) noexcept;

  /**
   * @brief Move assignment operator for Dense matrix.
   * 
   * @param other Another Dense object to move from.
   * @return Dense& Reference to the current object.
   */
  Dense &operator=(Dense &&other) noexcept;

  // at 
  std::complex<double> at(int col, int row) const noexcept;
  /**
   * @brief Get the number of rows in the Dense matrix.
   * 
   * @return int Number of rows.
   */
  int DimX() const;

  /**
   * @brief Get the number of columns in the Dense matrix.
   * 
   * @return int Number of columns.
   */
  int DimY() const;

  /**
   * @brief Get the data at a specific position in the Dense matrix.
   * 
   * @param col Column index.
   * @param row Row index.
   * @return std::complex<double>& Element at the specified position.
   */
  std::complex<double> &GetData(int col, int row) const;

  /**
   * @brief Get a reference to the data at a specific position in the Dense matrix.
   * 
   * @param col Column index.
   * @param row Row index.
   * @return std::complex<double>& Reference to the element at the specified position.
   */
  std::complex<double> &GetDataRef(int col, int row) const;

  /**
   * @brief Overloaded addition operator for Dense matrices.
   * 
   * @param A Another Dense object to add.
   * @return Dense Result of the addition.
   */
  Dense operator+(const Dense &A) const;

  /**
   * @brief Overloaded subtraction operator for Dense matrices.
   * 
   * @param A Another Dense object to subtract.
   * @return Dense Result of the subtraction.
   */
  Dense operator-(const Dense &A) const;

  /**
   * @brief Overloaded multiplication operator for scalar multiplication.
   * 
   * @param alpha Scalar value to multiply.
   * @return Dense Result of the scalar multiplication.
   */
  Dense operator*(const t_cplx &alpha) const noexcept;

  /**
   * @brief Overloaded multiplication operator for Dense matrices.
   * 
   * @param A Another Dense object to multiply.
   * @return Dense Result of the multiplication.
   */
  Dense operator*(const Dense &A) const;

  /**
   * @brief Overloaded subscript operator to access matrix elements.
   * 
   * @param col Column index.
   * @param row Row index.
   * @return std::complex<double> Element at the specified position.
   */
  std::complex<double> operator[](int col, int row);

  /**
   * @brief Transpose the Dense matrix.
   * 
   * @return Dense Transposed matrix.
   */
  Dense Transpose() const noexcept;


  /**
   * @brief Compute the Hermitian conjugate of the Dense matrix.
   * 
   * @return Dense Hermitian conjugate matrix.
   */
  Dense HermitianC() const noexcept;

  /**
   * @brief Flatten the Dense matrix data into a vector.
   * 
   * @return t_hostVect Flattened data.
   */
  t_hostVect FlattenedData() const noexcept;

  /**
   * @brief Print the Dense matrix.
   * 
   * @param kind Type of data to print (real, imaginary, etc.).
   * @param prec Precision of the printed data.
   */
  void Print(unsigned int kind = 0, unsigned int prec = 2) const;

private:
  std::unique_ptr<DenseImpl> pImpl;

  /**
   * @brief Constructor to initialize Dense matrix with a unique pointer to DenseImpl.
   * 
   * @param pImpl Unique pointer to DenseImpl.
   */
  Dense(std::unique_ptr<DenseImpl> pImpl) noexcept;
};

/**
 * @brief Overloaded multiplication operator for scalar multiplication.
 * 
 * @param alpha Scalar value to multiply.
 * @param rhs Dense object to multiply.
 * @return Dense Result of the scalar multiplication.
 */
Dense operator*(const std::complex<double> &alpha, const Dense &rhs) noexcept;

#endif // MAIN_DENSE_CUH
