//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by Conor Stevenson on 03/04/2022.
//

#include "la/Dense.h"
#include "la/DenseImpl.cuh"
#include "la/SparseImpl.cuh"
#include "la/Vect.h"
#include <Eigen/Sparse>

// Constructor that takes a t_hostMat
SparseImpl::SparseImpl(const t_hostMat &in)
    : DimX(in.size()), DimY(in[0].size()), CPUData(DimX, DimY) {
  // Convert the dense matrix to sparse format
  for (int i = 0; i < DimX; ++i) {
    for (int j = 0; j < DimY; ++j) {
      if (std::abs(in[i][j]) > 1e-10) { // Only add non-zero elements
        CPUData.insert(i, j) = in[i][j];
      }
    }
  }
  CPUData.makeCompressed();
}

SparseImpl SparseImpl::Add(const SparseImpl &B) const {
  SparseImpl out(DimX, DimY);
  out.CPUData = CPUData + B.CPUData;
  return out;
}

SparseImpl SparseImpl::RightMult(const SparseImpl &A) const {
  SparseImpl out(DimX, A.DimY);
  out.CPUData = CPUData * A.CPUData;
  return out;
}

SparseImpl SparseImpl::Transpose() const {
  SparseImpl out(DimY, DimX);
  out.CPUData = CPUData.transpose();
  return out;
}

SparseImpl SparseImpl::HermitianC() const {
  SparseImpl out(DimY, DimX);
  out.CPUData = CPUData.adjoint();
  return out;
}

DenseImpl SparseImpl::ToDense() {
  DenseImpl out(DimX, DimY);
  out.CPUData = CPUData.toDense();
  return out;
}

// Commented out as requested
/*
bool CompareCOORows(COOTuple L, COOTuple R) {
  if (L.Coords[0] == R.Coords[0]) {
    return L.Coords[1] < R.Coords[1];
  }
  return L.Coords[0] < R.Coords[0];
};

void SparseImpl::SortByRow() {
  std::sort(Data.begin(), Data.end(), CompareCOORows);
}

void SparseImpl::Trim() {}

std::vector<CompressedRow> SparseImpl::GetRows() const {
  std::vector<CompressedRow> sRows;
  std::vector<COOTuple> rowData;

  int index = 0;
  for (auto data : Data) {
    if (data.Coords[0] != index) {
      if (!rowData.empty()) {
        sRows.emplace_back(index, rowData);
        rowData.erase(rowData.begin(), rowData.end());
      }
      index = data.Coords[0];
    }
    rowData.emplace_back(data.Coords[0], data.Coords[1], data.Val);
  }

  if (!rowData.empty()) {
    sRows.emplace_back(index, rowData);
  }

  return sRows;
}

std::vector<CompressedRow> SparseImpl::GetCols() const {
  std::vector<CompressedRow> sCols;
  std::vector<COOTuple> colData;

  int index = 0;
  for (auto data : Data) {
    if (data.Coords[1] != index) {
      if (!colData.empty()) {
        sCols.emplace_back(index, colData);
        colData.erase(colData.begin(), colData.end());
      }
      index = data.Coords[1];
    }
    colData.emplace_back(data.Coords[0], data.Coords[1], data.Val);
  }

  if (!colData.empty()) {
    sCols.emplace_back(index, colData);
  }

  return sCols;
}

CompressedRow SparseVectorSum(const CompressedRow &rowA,
                              const CompressedRow &rowB) {
  unsigned int i = 0, j = 0;
  unsigned int I = rowA.RowData.size();
  unsigned int J = rowB.RowData.size();

  CompressedRow out(rowA.Index);
  while (true) {
    if (i > I - 1 || j > J - 1) {
      break;
    }

    if (rowA.RowData[i].Coords[1] == rowB.RowData[j].Coords[1]) {
      out.RowData.emplace_back(
          COOTuple(rowA.Index, rowA.RowData[i].Coords[1],
                   rowA.RowData[i].Val + rowB.RowData[j].Val));
      i++;
      j++;
      continue;
    }

    if (rowA.RowData[i].Coords[1] < rowB.RowData[j].Coords[1]) {
      out.RowData.emplace_back(rowA.RowData[i]);
      i++;
      continue;
    }

    if (rowA.RowData[i].Coords[1] > rowB.RowData[j].Coords[1]) {
      out.RowData.emplace_back(rowB.RowData[j]);
      j++;
      continue;
    }
  }

  while (i < I) {
    out.RowData.emplace_back(rowA.RowData[i]);
    i++;
  }

  while (j < J) {
    out.RowData.emplace_back(rowB.RowData[j]);
    j++;
  }

  return out;
}

std::complex<double> SparseDot(const CompressedRow &A, const CompressedRow &B) {
  unsigned int i = 0, j = 0;
  unsigned int I = A.RowData.size();
  unsigned int J = B.RowData.size();

  t_cplx out = 0;
  while (true) {
    if (i > I - 1 || j > J - 1) {
      break;
    }

    if (A.RowData[i].Coords[1] == B.RowData[j].Coords[1]) {
      out += A.RowData[i].Val * B.RowData[j].Val;
      i++;
      j++;
      continue;
    }

    if (A.RowData[i].Coords[1] < B.RowData[j].Coords[1]) {
      i++;
      continue;
    }

    if (A.RowData[i].Coords[1] > B.RowData[j].Coords[1]) {
      j++;
      continue;
    }
  }

  return out;
}
*/

VectImpl SparseImpl::VectMult(const VectImpl &vect) const {
  VectImpl out(DimX);

  // Convert vector to Eigen vector
  Eigen::VectorXcd eigenVect(vect.size());
  for (int i = 0; i < vect.size(); ++i) {
    eigenVect(i) = vect[i];
  }

  // Perform matrix-vector multiplication
  Eigen::VectorXcd result = CPUData * eigenVect;

  // Convert result to std::vector and set it in the output vector
  std::vector<std::complex<double>> resultData(DimX);
  for (int i = 0; i < DimX; ++i) {
    resultData[i] = result(i);
  }
  out.SetData(resultData);

  return out;
}

SparseImpl SparseImpl::operator+(const SparseImpl &A) const {
  return this->Add(A);
}

SparseImpl SparseImpl::operator-(const SparseImpl &A) const {
  return this->Add(A.Scale(-1));
}

SparseImpl SparseImpl::operator*(const t_cplx &alpha) const {
  return this->Scale(alpha);
}

SparseImpl operator*(const std::complex<double> &alpha, const SparseImpl &rhs) {
  return rhs * alpha;
}

SparseImpl SparseImpl::operator*(const SparseImpl &A) const {
  return this->RightMult(A);
}

SparseImpl SparseImpl::operator%(const SparseImpl &A) const {
  // Element-wise multiplication (Hadamard product)
  SparseImpl out(DimX, DimY);
  out.CPUData = CPUData.cwiseProduct(A.CPUData);
  return out;
}

unsigned int SparseImpl::NNZ() const { return CPUData.nonZeros(); }

SparseImpl SparseImpl::Scale(const t_cplx &alpha) const {
  SparseImpl out(DimX, DimY);
  out.CPUData = CPUData * alpha;
  return out;
}

void SparseImpl::Print() const {
  for (int i = 0; i < CPUData.outerSize(); ++i) {
    for (typename t_eigenSparseMat::InnerIterator it(CPUData, i); it; ++it) {
      std::cout << "(" << it.row() << "," << it.col() << "): " << it.value()
                << std::endl;
    }
  }
}

void SparseImpl::PrintRe() const {
  for (int i = 0; i < CPUData.outerSize(); ++i) {
    for (typename t_eigenSparseMat::InnerIterator it(CPUData, i); it; ++it) {
      std::cout << "(" << it.row() << "," << it.col()
                << "): " << it.value().real() << std::endl;
    }
  }
}

void SparseImpl::PrintIm() const {
  for (int i = 0; i < CPUData.outerSize(); ++i) {
    for (typename t_eigenSparseMat::InnerIterator it(CPUData, i); it; ++it) {
      std::cout << "(" << it.row() << "," << it.col()
                << "): " << it.value().imag() << std::endl;
    }
  }
}

void SparseImpl::PrintAbs() const {
  for (int i = 0; i < CPUData.outerSize(); ++i) {
    for (typename t_eigenSparseMat::InnerIterator it(CPUData, i); it; ++it) {
      std::cout << "(" << it.row() << "," << it.col()
                << "): " << std::abs(it.value()) << std::endl;
    }
  }
}

std::complex<double> &SparseImpl::CoeffRef(int i, int j) {
  // Eigen's coeffRef returns a reference to the coefficient
  return CPUData.coeffRef(i, j);
}

const t_hostMat SparseImpl::GetHostData() const {
  t_hostMat out;
  out.resize(DimX);
  for (int i = 0; i < DimX; ++i) {
    out[i].resize(DimY);
  }

  // Convert sparse matrix to dense matrix
  Eigen::MatrixXcd denseMat = CPUData.toDense();

  // Copy to t_hostMat
  for (int i = 0; i < DimX; ++i) {
    for (int j = 0; j < DimY; ++j) {
      out[i][j] = denseMat(i, j);
    }
  }

  return out;
}
