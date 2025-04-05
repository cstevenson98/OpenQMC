//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by Conor Stevenson on 15/3/2025.
//

#ifndef EIGEN_TYPES_H
#define EIGEN_TYPES_H

#include <Eigen/Dense>
#include <Eigen/Sparse>

// Eigen types
using t_eigenVect = Eigen::VectorXcd;
using t_eigenMat = Eigen::MatrixXcd;

// Define Eigen sparse matrix type with row-major storage
using t_eigenSparseMat =
    Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>;

#endif  // EIGEN_TYPES_H