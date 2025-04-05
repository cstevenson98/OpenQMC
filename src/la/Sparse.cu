//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by Conor Stevenson on 03/04/2022.
//

#include "la/Dense.h"
#include "la/DenseImpl.cuh"
#include "la/Sparse.h"
#include "la/SparseImpl.cuh"
#include "la/Super.cuh"
#include <vector>
//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by Conor Stevenson on 03/04/2022.
//

#include "la/Dense.h"
#include "la/Sparse.h"
#include "la/SparseImpl.cuh"
#include "la/Vect.h"
#include <memory>

Sparse::Sparse(int dimX, int dimY)
    : pImpl(std::make_unique<SparseImpl>(dimX, dimY)) {}

Sparse::Sparse(const t_hostMat &in) : pImpl(std::make_unique<SparseImpl>(in)) {}

Sparse::~Sparse() noexcept = default;

Sparse::Sparse(const Sparse &other) noexcept
    : pImpl(std::make_unique<SparseImpl>(*other.pImpl)) {}

Sparse::Sparse(Sparse &&other) noexcept : pImpl(std::move(other.pImpl)) {}

Sparse &Sparse::operator=(const Sparse &other) noexcept {
  if (this != &other) {
    pImpl = std::make_unique<SparseImpl>(*other.pImpl);
  }
  return *this;
}

Sparse Sparse::Scale(const t_cplx &alpha) const {
  return Sparse(std::make_unique<SparseImpl>(pImpl->Scale(alpha)));
}

Sparse Sparse::Add(const Sparse &B) const {
  return Sparse(std::make_unique<SparseImpl>(pImpl->Add(*B.pImpl)));
}

Sparse Sparse::RightMult(const Sparse &A) const {
  return Sparse(std::make_unique<SparseImpl>(pImpl->RightMult(*A.pImpl)));
}

Sparse Sparse::Transpose() const {
  return Sparse(std::make_unique<SparseImpl>(pImpl->Transpose()));
}

Sparse Sparse::HermitianC() const {
  return Sparse(std::make_unique<SparseImpl>(pImpl->HermitianC()));
}

Dense Sparse::ToDense() {
  return Dense(std::make_unique<DenseImpl>(pImpl->ToDense()));
}

// void Sparse::Trim() { pImpl->Trim(); }

Vect Sparse::VectMult(const Vect &vect) const {
  return Vect(std::make_unique<VectImpl>(pImpl->VectMult(*vect.pImpl)));
}

Sparse Sparse::operator+(const Sparse &A) const {
  return Sparse(std::make_unique<SparseImpl>(pImpl->operator+(*A.pImpl)));
}

Sparse Sparse::operator-(const Sparse &A) const {
  return Sparse(std::make_unique<SparseImpl>(pImpl->operator-(*A.pImpl)));
}

Sparse Sparse::operator*(const t_cplx &alpha) const {
  return Sparse(std::make_unique<SparseImpl>(pImpl->operator*(alpha)));
}

Sparse Sparse::operator*(const Sparse &A) const {
  return Sparse(std::make_unique<SparseImpl>(pImpl->operator*(*A.pImpl)));
}

Sparse Sparse::operator%(const Sparse &A) const {
  return Sparse(std::make_unique<SparseImpl>(pImpl->operator%(*A.pImpl)));
}

void Sparse::Print() const { pImpl->Print(); }

void Sparse::PrintRe() const { pImpl->PrintRe(); }

void Sparse::PrintIm() const { pImpl->PrintIm(); }

void Sparse::PrintAbs() const { pImpl->PrintAbs(); }

unsigned int Sparse::NNZ() const { return pImpl->NNZ(); }

int Sparse::DimX() const { return pImpl->DimX; }

int Sparse::DimY() const { return pImpl->DimY; }

const t_hostMat Sparse::GetHostData() const { return pImpl->GetHostData(); }

Sparse::Sparse(std::unique_ptr<SparseImpl> pImpl) : pImpl(std::move(pImpl)) {}

Sparse operator*(const std::complex<double> &alpha, const Sparse &rhs) {
  return rhs * alpha;
}

Sparse ToSparseCOO(const Dense &d) {
  // Get the dimensions of the dense matrix
  int dimX = d.DimX();
  int dimY = d.DimY();

  // Create a host matrix to store the data
  t_hostMat hostData(dimX, std::vector<std::complex<double>>(dimY));

  // Copy the data from the dense matrix to the host matrix
  for (int i = 0; i < dimX; ++i) {
    for (int j = 0; j < dimY; ++j) {
      hostData[i][j] = d.GetData(i, j);
    }
  }

  // Create a sparse matrix from the host matrix
  return Sparse(hostData);
}

std::vector<CompressedRow> SparseRowsCOO(const Sparse &s) {
  // Get the dimensions of the sparse matrix
  int dimX = s.DimX();
  int dimY = s.DimY();

  // Get the host data of the sparse matrix
  const t_hostMat hostData = s.GetHostData();

  // Create a vector of CompressedRow objects
  std::vector<CompressedRow> rows;

  // For each row in the matrix
  for (int i = 0; i < dimX; ++i) {
    // Create a CompressedRow object for the current row
    CompressedRow row(i);

    // For each column in the row
    for (int j = 0; j < dimY; ++j) {
      // If the element is non-zero, add it to the row data
      if (std::abs(hostData[i][j]) > 1e-10) {
        row.RowData.emplace_back(i, j, hostData[i][j]);
      }
    }

    // Add the row to the vector of rows
    rows.push_back(row);
  }

  return rows;
}

std::vector<CompressedRow> SparseColsCOO(const Sparse &s) {
  // Get the dimensions of the sparse matrix
  int dimX = s.DimX();
  int dimY = s.DimY();

  // Get the host data of the sparse matrix
  const t_hostMat hostData = s.GetHostData();

  // Create a vector of CompressedRow objects
  std::vector<CompressedRow> cols;

  // For each column in the matrix
  for (int j = 0; j < dimY; ++j) {
    // Create a CompressedRow object for the current column
    CompressedRow col(j);

    // For each row in the column
    for (int i = 0; i < dimX; ++i) {
      // If the element is non-zero, add it to the column data
      if (std::abs(hostData[i][j]) > 1e-10) {
        col.RowData.emplace_back(i, j, hostData[i][j]);
      }
    }

    // Add the column to the vector of columns
    cols.push_back(col);
  }

  return cols;
}
