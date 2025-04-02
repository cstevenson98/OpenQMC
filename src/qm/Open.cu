//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by conor on 16/04/22.
//

#include "core/types.h"
#include "la/SparseImpl.cuh"
#include "la/Super.cuh"
#include "qm/Spins.cuh"

SparseImpl Lindblad(const SparseImpl &A) {
  auto id = Identity(A.DimY);

  auto AdagA = A.HermitianC() * A;
  return ToSuper(A, A.HermitianC()) * t_cplx(2.0) - ToSuper(AdagA, id) -
         ToSuper(id, AdagA);
}

SparseImpl Lindblad(const SparseImpl &A, const SparseImpl &B) {
  auto id = Identity(A.DimY);
  auto BA = B * A;

  return ToSuper(A, B) * t_cplx(2.0) - ToSuper(BA, id) - ToSuper(id, BA);
}

SparseImpl SuperComm(const SparseImpl &A) {
  auto id = Identity(A.DimY);
  return ToSuper(A, id) - ToSuper(id, A);
}