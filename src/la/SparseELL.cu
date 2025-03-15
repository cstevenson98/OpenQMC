//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by conor on 11/04/2022.
//

#include "core/types.cuh"
#include "la/Sparse.cuh"
#include "la/SparseELL.cuh"

SparseELL ToSparseELL(const Sparse &A) {
  auto rows = SparseRowsCOO(A);

  // determine length of longest row
  unsigned int highestCount = 0;
  for (auto row : rows) {
    unsigned int len = row.RowData.size();
    if (len > highestCount) {
      highestCount = len;
    }
  }

  SparseELL out(A.DimX, A.DimY, highestCount);
  for (auto row : rows) {
    for (int i = 0; i < row.RowData.size(); i++) {
      // Change to denseimpls
      // out.Values.GetDataRef(row.Index, i) = row.RowData[i].Val;
      // out.Indices.GetDataRef(row.Index, i) = row.RowData[i].Coords[1];
    }

    // padding with '-1'
    for (int i = row.RowData.size(); i < highestCount; i++) {
      // out.Values.GetDataRef(row.Index, i) = 0;
      // out.Indices.GetDataRef(row.Index, i) = -1;
    }
  }

  return out;
}

// Vect SparseELL::VectMult(const Vect::Impl &vect) const {
//   Vect out(vect.Data.size());

//   for (int row = 0; row < vect.Data.size(); ++row) {
//     out[row] = 0;
//     for (int i = 0; i < EntriesPerRow; i++) {
//       int col = floor(Indices.GetData(row, i).real());
//       th_cplx val = Values.GetData(row, i);

//       if (col > -1)
//         out.Data[row] += val * vect.Data[col];
//     }
//   }
//   return out;
// }
