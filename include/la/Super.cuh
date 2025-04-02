//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by conor on 04/04/2022.
//

#ifndef MAIN_SUPER_CUH
#define MAIN_SUPER_CUH

#include "la/SparseImpl.cuh"

/**
 * @brief Computes the Kronecker product of two sparse matrices.
 *
 * @param A First sparse matrix.
 * @param B Second sparse matrix.
 * @return SparseImpl Resulting sparse matrix from the Kronecker product.
 */
SparseImpl Kronecker(const SparseImpl &A, const SparseImpl &B);

/**
 * @brief Computes the tensor product of a vector of sparse matrices.
 *
 * @param matrices Vector of sparse matrices.
 * @return SparseImpl Resulting sparse matrix from the tensor product.
 */
SparseImpl Tensor(const std::vector<SparseImpl> &matrices);

/**
 * @brief Converts two sparse matrices into a super matrix.
 *
 * @param A First sparse matrix.
 * @param B Second sparse matrix.
 * @return SparseImpl Resulting super sparse matrix.
 */
SparseImpl ToSuper(const SparseImpl &A, const SparseImpl &B);

#endif // MAIN_SUPER_CUH
