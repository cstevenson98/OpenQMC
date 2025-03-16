//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by conor on 04/04/2022.
//

#ifndef MAIN_SUPER_CUH
#define MAIN_SUPER_CUH

#include "la/Sparse.cuh"

/**
 * @brief Computes the Kronecker product of two sparse matrices.
 * 
 * @param A First sparse matrix.
 * @param B Second sparse matrix.
 * @return Sparse Resulting sparse matrix from the Kronecker product.
 */
Sparse Kronecker(const Sparse& A, const Sparse& B);

/**
 * @brief Computes the tensor product of a vector of sparse matrices.
 * 
 * @param matrices Vector of sparse matrices.
 * @return Sparse Resulting sparse matrix from the tensor product.
 */
Sparse Tensor(const std::vector<Sparse>& matrices);

/**
 * @brief Converts two sparse matrices into a super matrix.
 * 
 * @param A First sparse matrix.
 * @param B Second sparse matrix.
 * @return Sparse Resulting super sparse matrix.
 */
Sparse ToSuper(const Sparse& A, const Sparse& B);

#endif //MAIN_SUPER_CUH
