//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by conor on 11/04/2022.
//

#ifndef MAIN_SPARSEELL_CUH
#define MAIN_SPARSEELL_CUH

#include "la/Dense.h"
#include "la/DenseImpl.cuh"
#include "la/Sparse.cuh"
#include "la/Vect.h"

/**
 * @brief A class representing sparse matrices in ELLPACK format.
 */
struct SparseELL {
  int DimX; ///< Number of rows
  int DimY; ///< Number of columns
  int NNZ; ///< Number of non-zero elements
  int EntriesPerRow; ///< Number of entries per row

  DenseImpl Values; ///< Values of the non-zero elements
  DenseImpl Indices; ///< Column indices of the non-zero elements

  /**
   * @brief Default constructor to initialize an empty SparseELL matrix.
   */
  SparseELL() : DimX(0), DimY(0) {};

  /**
   * @brief Constructor to initialize SparseELL matrix with given dimensions and entries per row.
   * 
   * @param dimX Number of rows.
   * @param dimY Number of columns.
   * @param EntriesPerRow Number of entries per row.
   */
  SparseELL(int dimX, int dimY, int EntriesPerRow)
      : DimX(dimX), DimY(dimY), Values(dimX, EntriesPerRow),
        Indices(dimX, EntriesPerRow), EntriesPerRow(EntriesPerRow) {};

  /**
   * @brief Loads a Dense matrix into the SparseELL format.
   * 
   * @param A Dense matrix to load.
   */
  void LoadDense(const Dense &A);

  /**
   * @brief Converts the SparseELL matrix to a Dense matrix.
   * 
   * @return Dense Dense matrix.
   */
  Dense ToDense();

  /**
   * @brief Multiplies the SparseELL matrix by a vector.
   * 
   * @param vect Vector to multiply.
   * @return Vect Result of the multiplication.
   */
  Vect VectMult(const Vect &vect) const;
};

/**
 * @brief Converts a Sparse matrix to a SparseELL matrix.
 * 
 * @param A Sparse matrix to convert.
 * @return SparseELL SparseELL matrix.
 */
SparseELL ToSparseELL(const Sparse &A);

#endif // MAIN_SPARSEELL_CUH
