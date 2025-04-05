//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by Conor Stevenson on 03/04/2022.
//

#ifndef MAIN_SPARSE_H
#define MAIN_SPARSE_H

#include <complex>
#include <memory>
#include <vector>

#include "core/types.h"

class SparseImpl;
class Dense;
class Vect;

/**
 * @brief A structure representing a tuple in COO (Coordinate) format.
 */
struct COOTuple {
  int Coords[2]{0, 0};  ///< Coordinates of the element
  t_cplx Val;           ///< Value of the element

  /**
   * @brief Constructor to initialize COOTuple with coordinates and value.
   *
   * @param x Row index.
   * @param y Column index.
   * @param val Value of the element.
   */
  COOTuple(int x, int y, t_cplx val) : Val(val) {
    Coords[0] = x;
    Coords[1] = y;
  };
};

/**
 * @brief A structure representing a compressed row in a sparse matrix.
 */
struct CompressedRow {
  int Index;                      ///< Index of the row
  std::vector<COOTuple> RowData;  ///< Data of the row

  /**
   * @brief Constructor to initialize CompressedRow with an index.
   *
   * @param index Index of the row.
   */
  explicit CompressedRow(int index) : Index(index){};

  /**
   * @brief Constructor to initialize CompressedRow with an index and row data.
   *
   * @param index Index of the row.
   * @param rowData Data of the row.
   */
  CompressedRow(int index, const std::vector<COOTuple>& rowData)
      : Index(index) {
    RowData = rowData;
  };
};

/**
 * @brief A class representing sparse matrices. Supports matrix algebra and uses
 * complex numbers.
 */
class Sparse {
 public:
  /**
   * @brief Constructor to initialize Sparse matrix with given dimensions.
   *
   * @param dimX Number of rows.
   * @param dimY Number of columns.
   */
  Sparse(int dimX, int dimY);

  /**
   * @brief Constructor to initialize Sparse matrix from a host matrix.
   *
   * @param in Host matrix to initialize from.
   */
  explicit Sparse(const t_hostMat& in);

  /**
   * @brief Destructor for Sparse matrix.
   */
  ~Sparse() noexcept;

  /**
   * @brief Copy constructor for Sparse matrix.
   *
   * @param other Another Sparse object to copy from.
   */
  Sparse(const Sparse& other) noexcept;

  /**
   * @brief Move constructor for Sparse matrix.
   *
   * @param other Another Sparse object to move from.
   */
  Sparse(Sparse&& other) noexcept;

  /**
   * @brief Copy assignment operator for Sparse matrix.
   *
   * @param other Another Sparse object to copy from.
   * @return Sparse& Reference to the current object.
   */
  Sparse& operator=(const Sparse& other) noexcept;

  /**
   * @brief Scales the Sparse matrix by a scalar value.
   *
   * @param alpha Scalar value to multiply.
   * @return Sparse Result of the scalar multiplication.
   */
  Sparse Scale(const t_cplx& alpha) const;

  /**
   * @brief Adds two Sparse matrices.
   *
   * @param B Another Sparse object to add.
   * @return Sparse Result of the addition.
   */
  Sparse Add(const Sparse& B) const;

  /**
   * @brief Multiplies two Sparse matrices.
   *
   * @param A Another Sparse object to multiply.
   * @return Sparse Result of the multiplication.
   */
  Sparse RightMult(const Sparse& A) const;

  /**
   * @brief Transposes the Sparse matrix.
   *
   * @return Sparse Transposed matrix.
   */
  Sparse Transpose() const;

  /**
   * @brief Computes the Hermitian conjugate of the Sparse matrix.
   *
   * @return Sparse Hermitian conjugate matrix.
   */
  Sparse HermitianC() const;

  /**
   * @brief Converts the Sparse matrix to a Dense matrix.
   *
   * @return Dense Dense matrix.
   */
  Dense ToDense();

  /**
   * @brief Trims the Sparse matrix data.
   */
  void Trim();

  /**
   * @brief Multiplies the Sparse matrix by a vector.
   *
   * @param vect Vector to multiply.
   * @return Vect Result of the multiplication.
   */
  Vect VectMult(const Vect& vect) const;

  /**
   * @brief Overloaded addition operator for Sparse matrices.
   *
   * @param A Another Sparse object to add.
   * @return Sparse Result of the addition.
   */
  Sparse operator+(const Sparse& A) const;

  /**
   * @brief Overloaded subtraction operator for Sparse matrices.
   *
   * @param A Another Sparse object to subtract.
   * @return Sparse Result of the subtraction.
   */
  Sparse operator-(const Sparse& A) const;

  /**
   * @brief Overloaded multiplication operator for scalar multiplication.
   *
   * @param alpha Scalar value to multiply.
   * @return Sparse Result of the scalar multiplication.
   */
  Sparse operator*(const t_cplx& alpha) const;

  /**
   * @brief Overloaded multiplication operator for Sparse matrices.
   *
   * @param A Another Sparse object to multiply.
   * @return Sparse Result of the multiplication.
   */
  Sparse operator*(const Sparse& A) const;

  /**
   * @brief Overloaded element-wise multiplication operator for Sparse matrices.
   *
   * @param A Another Sparse object to multiply element-wise.
   * @return Sparse Result of the element-wise multiplication.
   */
  Sparse operator%(const Sparse& A) const;

  /**
   * @brief Prints the Sparse matrix.
   */
  void Print() const;

  /**
   * @brief Prints the real part of the Sparse matrix.
   */
  void PrintRe() const;

  /**
   * @brief Prints the imaginary part of the Sparse matrix.
   */
  void PrintIm() const;

  /**
   * @brief Prints the absolute value of the Sparse matrix.
   */
  void PrintAbs() const;

  /**
   * @brief Gets the number of non-zero elements in the Sparse matrix.
   *
   * @return unsigned int Number of non-zero elements.
   */
  unsigned int NNZ() const;

  /**
   * @brief Gets the number of rows in the Sparse matrix.
   *
   * @return int Number of rows.
   */
  int DimX() const;

  /**
   * @brief Gets the number of columns in the Sparse matrix.
   *
   * @return int Number of columns.
   */
  int DimY() const;

  /**
   * @brief Gets the host data of the Sparse matrix in COO format.
   *
   * @return const t_hostMat, the host data.
   */
  const t_hostMat GetHostData() const;

  /**
   * @brief Gets a reference to a coefficient in the matrix.
   *
   * @param i Row index.
   * @param j Column index.
   * @return std::complex<double>& Reference to the coefficient.
   */
  std::complex<double>& CoeffRef(int i, int j);

 private:
  std::unique_ptr<SparseImpl> pImpl;
  Sparse(std::unique_ptr<SparseImpl> pImpl);
};

/**
 * @brief Overloaded multiplication operator for scalar multiplication.
 *
 * @param alpha Scalar value to multiply.
 * @param rhs Sparse object to multiply.
 * @return Sparse Result of the scalar multiplication.
 */
Sparse operator*(const std::complex<double>& alpha, const Sparse& rhs);

/**
 * @brief Converts a Dense matrix to a Sparse matrix in COO format.
 *
 * @param d Dense matrix to convert.
 * @return Sparse Sparse matrix in COO format.
 */
Sparse ToSparseCOO(const Dense& d);

/**
 * @brief Gets the rows of a Sparse matrix in COO format.
 *
 * @param s Sparse matrix.
 * @return std::vector<CompressedRow> Rows of the Sparse matrix.
 */
std::vector<CompressedRow> SparseRowsCOO(const Sparse& s);

/**
 * @brief Gets the columns of a Sparse matrix in COO format.
 *
 * @param s Sparse matrix.
 * @return std::vector<CompressedRow> Columns of the Sparse matrix.
 */
std::vector<CompressedRow> SparseColsCOO(const Sparse& s);

/**
 * @brief Computes the sum of two compressed rows.
 *
 * @param A First compressed row.
 * @param B Second compressed row.
 * @return CompressedRow Result of the sum.
 */
CompressedRow SparseVectorSum(const CompressedRow& A, const CompressedRow& B);

/**
 * @brief Computes the dot product of two compressed rows.
 *
 * @param A First compressed row.
 * @param B Second compressed row.
 * @return std::complex<double> Dot product result.
 */
std::complex<double> SparseDot(const CompressedRow& A, const CompressedRow& B);

#endif  // MAIN_SPARSE_H
