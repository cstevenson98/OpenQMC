//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by Conor Stevenson on 03/04/2022.
//

#include "la/Dense.h"
#include "la/SparseImpl.cuh"
#include "la/Vect.h"

SparseImpl SparseImpl::Add(const SparseImpl &B) const {
  SparseImpl out(B.DimX, B.DimY);

  int i = 0;
  int I = B.Data.size();
  for (auto elemA : Data) {
    while (i < I - 1 && B.Data[i].Coords[0] != elemA.Coords[0] &&
           B.Data[i].Coords[1] != elemA.Coords[1]) {
      out.Data.emplace_back(
          COOTuple(B.Data[i].Coords[0], B.Data[i].Coords[1], B.Data[i].Val));
      i++;
    }

    if (i < I && B.Data[i].Coords[0] == elemA.Coords[0] &&
        B.Data[i].Coords[1] == elemA.Coords[1]) {
      out.Data.emplace_back(COOTuple(elemA.Coords[0], elemA.Coords[1],
                                     elemA.Val + B.Data[i].Val));
    } else {
      out.Data.emplace_back(
          COOTuple(elemA.Coords[0], elemA.Coords[1], B.Data[i].Val));
    }
  }

  return out;
}

SparseImpl SparseImpl::RightMult(const SparseImpl &A) const {
  SparseImpl out(DimX, A.DimY);

  auto thisRows = this->GetRows();
  auto ACols = A.GetCols();

  for (auto &row : thisRows) {
    int i = row.Index;
    for (auto &col : ACols) {
      int j = col.Index;
      out.Data.emplace_back(i, j, SparseDot(row, col));
    }
  }

  return out;
}

SparseImpl SparseImpl::Transpose() const {
  SparseImpl out(DimX, DimY);
  out.Data.resize(Data.size(), COOTuple(0, 0, 0));

  for (int i = 0; i < Data.size(); i++) {
    out.Data[i].Coords[0] = Data[i].Coords[1];
    out.Data[i].Coords[1] = Data[i].Coords[0];
    out.Data[i].Val = Data[i].Val;
  }

  out.SortByRow();

  return out;
}

SparseImpl SparseImpl::HermitianC() const {
  SparseImpl out(DimX, DimY);
  out.Data.resize(Data.size(), COOTuple(0, 0, 0));

  for (int i = 0; i < Data.size(); i++) {
    out.Data[i].Coords[0] = Data[i].Coords[1];
    out.Data[i].Coords[1] = Data[i].Coords[0];
    out.Data[i].Val = conj(Data[i].Val);
  }

  out.SortByRow();

  return out;
}

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

SparseImpl ToSparseCOO(const Dense &d) {
  SparseImpl out(d.DimX(), d.DimY());

  for (int i = 0; i < d.DimX(); i++) {
    for (int j = 0; j < d.DimY(); j++) {
      if (d.GetData(i, j) != 0.) {
        out.Data.emplace_back(i, j, d.GetData(i, j));
      }
    }
  }

  return out;
}

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

Dense SparseImpl::ToDense() {
  Dense out(DimX, DimY);

  for (auto &elem : Data) {
    auto [i, j] = elem.Coords;
    out.GetDataRef(i, j) = elem.Val;
  }

  return out;
}

VectImpl SparseImpl::VectMult(const VectImpl &vect) const {
  VectImpl out(DimX);
  auto outData = out.GetData();
  auto vectData = vect.GetData();

  for (int i = 0; i < DimX; ++i) {
    outData[i] = 0;
    for (const auto &elem : Data) {
      if (elem.Coords[0] == i) {
        outData[i] += elem.Val * vectData[elem.Coords[1]];
      }
    }
  }

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
  return Kronecker(*this, A);
}

unsigned int SparseImpl::NNZ() const { return Data.size(); }
