//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by conor on 04/04/2022.
//

#include "la/SparseImpl.cuh"
#include "la/Super.cuh"

SparseImpl Kronecker(const SparseImpl &A, const SparseImpl &B) {
  SparseImpl out(A.DimX * B.DimX, A.DimY * B.DimY);

  for (auto elemA : A.Data) {
    for (auto elemB : B.Data) {
      out.Data.emplace_back(B.DimX * elemA.Coords[0] + elemB.Coords[0],
                            B.DimY * elemA.Coords[1] + elemB.Coords[1],
                            elemA.Val * elemB.Val);
    }
  }

  return out;
}

SparseImpl Tensor(const std::vector<SparseImpl> &matrices) {
  SparseImpl out = matrices[0];

  for (int i = 1; i < matrices.size(); ++i) {
    out = Kronecker(out, matrices[i]);
  }

  return out;
}

SparseImpl ToSuper(const SparseImpl &A, const SparseImpl &B) {
  return Kronecker(B.Transpose(), A);
}
