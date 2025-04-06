//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by Conor Stevenson on 15/3/2025.
//

#pragma once

#include <cusparse.h>
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "core/types.cuh"
#include "la/CuSparseSingleton.cuh"
#include "la/Sparse.h"
#include "la/VectImplGPU.cuh"
#include <complex>

/**
 * @brief A class representing the GPU implementation of sparse matrices using
 * Thrust and cuSPARSE. Supports matrix algebra and uses complex numbers.
 */
class SparseImplGPU {
public:
  int DimX; ///< Number of rows
  int DimY; ///< Number of columns

  // CSR format for cuSPARSE
  t_devcVectInt rowPtr; ///< Row pointers for CSR format
  t_devcVectInt colInd; ///< Column indices for CSR format
  t_devcVect values;    ///< Non-zero values for CSR format
  int nnz;              ///< Number of non-zero elements

  // cuSPARSE descriptor
  cusparseSpMatDescr_t matDescr_; // cuSPARSE matrix descriptor

  /**
   * @brief Constructor to initialize SparseImplGPU matrix with given
   * dimensions.
   *
   * @param dimX Number of rows.
   * @param dimY Number of columns.
   */
  SparseImplGPU(int dimX, int dimY);

  /**
   * @brief Destructor to clean up cuSPARSE resources.
   */
  ~SparseImplGPU();

  /**
   * @brief Constructor to initialize SparseImplGPU matrix from a host matrix.
   *
   * @param in Host matrix to initialize from.
   */
  explicit SparseImplGPU(const t_hostMat &in);

  /**
   * @brief Constructor to initialize SparseImplGPU matrix from a CPU
   * SparseImpl.
   *
   * @param cpuMatrix CPU sparse matrix to initialize from.
   */
  explicit SparseImplGPU(const class SparseImpl &cpuMatrix);

  /**
   * @brief Scales the SparseImplGPU matrix by a scalar value.
   *
   * @param alpha Scalar value to multiply.
   * @return SparseImplGPU Result of the scalar multiplication.
   */
  SparseImplGPU Scale(const th_cplx &alpha) const;

  /**
   * @brief Adds two SparseImplGPU matrices.
   *
   * @param B Another SparseImplGPU object to add.
   * @return SparseImplGPU Result of the addition.
   */
  SparseImplGPU Add(const SparseImplGPU &B) const;

  /**
   * @brief Multiplies two SparseImplGPU matrices.
   *
   * @param A Another SparseImplGPU object to multiply.
   * @return SparseImplGPU Result of the multiplication.
   */
  SparseImplGPU RightMult(const SparseImplGPU &A) const;

  /**
   * @brief Transposes the SparseImplGPU matrix.
   *
   * @return SparseImplGPU Transposed matrix.
   */
  SparseImplGPU Transpose() const;

  /**
   * @brief Computes the Hermitian conjugate of the SparseImplGPU matrix.
   *
   * @return SparseImplGPU Hermitian conjugate matrix.
   */
  SparseImplGPU HermitianC() const;

  /**
   * @brief Multiplies the SparseImplGPU matrix by a vector.
   *
   * @param vect Vector to multiply.
   * @return VectImplGPU Result of the multiplication.
   */
  VectImplGPU VectMult(const VectImplGPU &vect) const;

  /**
   * @brief Overloaded addition operator for SparseImplGPU matrices.
   *
   * @param A Another SparseImplGPU object to add.
   * @return SparseImplGPU Result of the addition.
   */
  SparseImplGPU operator+(const SparseImplGPU &A) const;

  /**
   * @brief Overloaded subtraction operator for SparseImplGPU matrices.
   *
   * @param A Another SparseImplGPU object to subtract.
   * @return SparseImplGPU Result of the subtraction.
   */
  SparseImplGPU operator-(const SparseImplGPU &A) const;

  /**
   * @brief Overloaded multiplication operator for scalar multiplication.
   *
   * @param alpha Scalar value to multiply.
   * @return SparseImplGPU Result of the scalar multiplication.
   */
  SparseImplGPU operator*(const th_cplx &alpha) const;

  /**
   * @brief Overloaded multiplication operator for SparseImplGPU matrices.
   *
   * @param A Another SparseImplGPU object to multiply.
   * @return SparseImplGPU Result of the multiplication.
   */
  SparseImplGPU operator*(const SparseImplGPU &A) const;

  /**
   * @brief Overloaded element-wise multiplication operator for SparseImplGPU
   * matrices.
   *
   * @param A Another SparseImplGPU object to multiply element-wise.
   * @return SparseImplGPU Result of the element-wise multiplication.
   */
  SparseImplGPU operator%(const SparseImplGPU &A) const;

  /**
   * @brief Gets the number of non-zero elements in the SparseImplGPU matrix.
   *
   * @return unsigned int Number of non-zero elements.
   */
  unsigned int NNZ() const;

  /**
   * @brief Gets the host data of the SparseImplGPU matrix.
   *
   * @return const t_hostMat The host data.
   */
  const t_hostMat GetHostData() const;

  /**
   * @brief Gets a reference to a coefficient in the matrix.
   *
   * @param i Row index.
   * @param j Column index.
   * @return std::complex<double> The coefficient.
   */
  std::complex<double> CoeffRef(int i, int j) const;

  /**
   * @brief Gets the raw pointer to the row pointers.
   *
   * @return const int* Raw pointer to row pointers.
   */
  const int *GetRowPtr() const {
    return thrust::raw_pointer_cast(rowPtr.data());
  }

  /**
   * @brief Gets the raw pointer to the column indices.
   *
   * @return const int* Raw pointer to column indices.
   */
  const int *GetColIndPtr() const {
    return thrust::raw_pointer_cast(colInd.data());
  }

  /**
   * @brief Gets the raw pointer to the values.
   *
   * @return const th_cplx* Raw pointer to values.
   */
  const th_cplx *GetValuesPtr() const {
    return thrust::raw_pointer_cast(values.data());
  }

  /**
   * @brief Gets the cuSPARSE matrix descriptor.
   *
   * @return cusparseSpMatDescr_t The cuSPARSE matrix descriptor.
   */
  cusparseSpMatDescr_t GetMatDescr() const { return matDescr_; }

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

  /**
   * @brief Converts from Eigen sparse matrix to CSR format.
   *
   * @param eigenMatrix Eigen sparse matrix to convert from.
   */
  void ConvertFromEigen(const class SparseImpl &eigenMatrix);

  thrust::device_vector<int> rowPtr_;
  thrust::device_vector<int> colInd_;
  thrust::device_vector<th_cplx> values_;
};

/**
 * @brief Overloaded multiplication operator for scalar multiplication.
 *
 * @param alpha Scalar value to multiply.
 * @param rhs SparseImplGPU object to multiply.
 * @return SparseImplGPU Result of the scalar multiplication.
 */
SparseImplGPU operator*(const th_cplx &alpha, const SparseImplGPU &rhs);