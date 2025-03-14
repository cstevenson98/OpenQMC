//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by Conor Stevenson on 03/04/2022.
//

#include "la/Dense.h"
#include "la/Sparse.cuh"
#include "la/Super.cuh"
#include <vector>

Sparse Sparse::Scale(const th_cplx &alpha) const {
  Sparse out(DimX, DimY);
  out.Data.resize(Data.size(), COOTuple(0, 0, 0));

  for (int i = 0; i < Data.size(); ++i) {
    out.Data[i] =
        COOTuple(Data[i].Coords[0], Data[i].Coords[1], alpha * Data[i].Val);
  }
  return out;
}

Sparse Sparse::Add(const Sparse &B) const {
  Sparse out(B.DimX, B.DimY);

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

  //    auto rowsA = SparseRowsCOO(*this);
  //    if (rowsA.empty()) {
  //        return B;
  //    }
  //
  //    auto rowsB = SparseRowsCOO(B);
  //    if (rowsB.empty()) {
  //        return *this;
  //    }
  //
  //    unsigned int i = 0, j = 0;
  //    unsigned int I = rowsA.size();
  //    unsigned int J = rowsB.size();
  //
  //    while (true) {
  //        if (i > I-1 || j > J-1) {
  //            break;
  //        }
  //
  //        if (rowsA[i].Index == rowsB[j].Index) {
  //            auto sumRow = SparseVectorSum(rowsA[i], rowsB[j]);
  //            for (auto &elem : sumRow.RowData) {
  //                out.Data.emplace_back(elem.Coords[0], elem.Coords[1],
  //                elem.Val);
  //            }
  //            i++;
  //            j++;
  //            continue;
  //        }
  //
  //        if (rowsA[i].Index < rowsB[j].Index) {
  //            for (auto &elem : rowsA[i].RowData) {
  //                out.Data.emplace_back(elem.Coords[0], elem.Coords[1],
  //                elem.Val);
  //            }
  //            i++;
  //            continue;
  //        }
  //
  //        if (rowsA[i].Index > rowsB[j].Index) {
  //            for (auto &elem : rowsB[j].RowData) {
  //                out.Data.emplace_back(elem.Coords[0], elem.Coords[1],
  //                elem.Val);
  //            }
  //            j++;
  //            continue;
  //        }
  //
  //    }
  //
  //    // Add anything remaining
  //    while (i < I) {
  //        for (auto &elem : rowsA[i].RowData) {
  //            out.Data.emplace_back(elem.Coords[0], elem.Coords[1], elem.Val);
  //        }
  //        i++;
  //    }
  //    while (j < J) {
  //        for (auto &elem : rowsB[j].RowData) {
  //            out.Data.emplace_back(elem.Coords[0], elem.Coords[1], elem.Val);
  //        }
  //        j++;
  //    }
  //
  return out;
}

Sparse Sparse::RightMult(const Sparse &A) const {
  Sparse out(DimX, A.DimY);

  auto thisRows = SparseRowsCOO(*this);
  auto ACols = SparseColsCOO(A);

  for (auto &row : thisRows) {
    int i = row.Index;
    for (auto &col : ACols) {
      int j = col.Index;
      out.Data.emplace_back(i, j, SparseDot(row, col));
    }
  }

  return out;
}

Sparse Sparse::Transpose() const {
  Sparse out(DimX, DimY);
  out.Data.resize(Data.size(), COOTuple(0, 0, 0));

  for (int i = 0; i < Data.size(); i++) {
    out.Data[i].Coords[0] = Data[i].Coords[1];
    out.Data[i].Coords[1] = Data[i].Coords[0];
    out.Data[i].Val = Data[i].Val;
  }

  out.SortByRow();

  return out;
}

Sparse Sparse::HermitianC() const {
  Sparse out(DimX, DimY);
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

void Sparse::SortByRow() {
  std::sort(Data.begin(), Data.end(), CompareCOORows);
}

void Sparse::Trim() {}

Sparse ToSparseCOO(const Dense &d) {
  Sparse out(d.DimX(), d.DimY());

  for (int i = 0; i < d.DimX(); i++) {
    for (int j = 0; j < d.DimY(); j++) {
      if (d.GetData(i, j) != 0.) {
        out.Data.emplace_back(i, j, d.GetData(i, j));
      }
    }
  }

  return out;
}

std::vector<CompressedRow> SparseRowsCOO(const Sparse &s) {
  std::vector<CompressedRow> sRows;
  std::vector<COOTuple> rowData;

  int index = 0;
  for (auto data : s.Data) {
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

std::vector<CompressedRow> SparseColsCOO(const Sparse &s) {
  Sparse sT = s.Transpose();

  return SparseRowsCOO(sT);
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

  th_cplx out = 0;
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

Dense Sparse::ToDense() {
  Dense out(DimX, DimY);

  for (auto &elem : Data) {
    auto [i, j] = elem.Coords;
    out.GetDataRef(i, j) = elem.Val;
  }

  return out;
}

Vect Sparse::VectMult(const Vect &vect) const {
  std::vector<CompressedRow> sRows = SparseRowsCOO(*this);

  Vect out(vect.Data.size());
  for (const auto &row : sRows) {
    for (const auto rowElem : row.RowData) {
      out.Data[row.Index] += rowElem.Val * vect.Data[rowElem.Coords[1]];
    }
  }

  return out;
}

Sparse Sparse::operator+(const Sparse &A) const { return this->Add(A); }

Sparse Sparse::operator-(const Sparse &A) const {
  return this->Add(A.Scale(-1));
}

Sparse Sparse::operator*(const t_cplx &alpha) const {
  return this->Scale(alpha);
}

Sparse operator*(const std::complex<double> &alpha, const Sparse &rhs) {
  return rhs * alpha;
}

Sparse Sparse::operator*(const Sparse &A) const { return this->RightMult(A); }

Sparse Sparse::operator%(const Sparse &A) const { return Kronecker(*this, A); }

unsigned int Sparse::NNZ() const { return Data.size(); }
