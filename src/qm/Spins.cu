//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by conor on 05/04/2022.
//

#include "core/types.h"
#include "la/SparseImpl.cuh"
#include "qm/Spins.cuh"

SparseImpl Identity(unsigned int N) {
  SparseImpl out(N, N);
  for (int i = 0; i < N; ++i) {
    out.Data.emplace_back(i, i, 1);
  }

  return out;
}

Sparse SigmaX() {
  t_hostMat sX = {{0, 1}, {1, 0}};
  Dense out(sX);
  return ToSparseCOO(out);
}

Sparse SigmaY() {
  t_hostMat sY = {{0, {0, -1}}, {{0, 1}, 0}};
  Dense out(sY);
  return ToSparseCOO(out);
}

Sparse SigmaZ() {
  t_hostMat sZ = {{1, 0}, {0, -1}};
  Dense out(sZ);
  return ToSparseCOO(out);
}

Sparse SigmaPlus() {
  t_hostMat sP = {{0, 1}, {0, 0}};
  Dense out(sP);
  return ToSparseCOO(out);
}

Sparse SigmaMinus() {
  t_hostMat sM = {{0, 0}, {1, 0}};
  Dense out(sM);
  return ToSparseCOO(out);
}

Sparse SigmaX(unsigned int N, unsigned int j) {
  std::vector<Sparse> operators;

  for (int i = 0; i < N; ++i) {
    if (i == j) {
      // operators.emplace_back(SigmaX());
    } else {
      // operators.emplace_back(Identity(2));
    }
  }

  // return Tensor(operators);
  return Sparse(pow(2, N), pow(2, N));
}

Sparse SigmaY(unsigned int N, unsigned int j) {
  std::vector<Sparse> operators;

  for (int i = 0; i < N; ++i) {
    if (i == j) {
      // operators.emplace_back(SigmaY());
    } else {
      // operators.emplace_back(Identity(2));
    }
  }

  // return Tensor(operators);
  return Sparse(pow(2, N), pow(2, N));
}

Sparse SigmaZ(unsigned int N, unsigned int j) {
  std::vector<Sparse> operators;

  for (int i = 0; i < N; ++i) {
    if (i == j) {
      // operators.emplace_back(SigmaZ());
    } else {
      // operators.emplace_back(Identity(2));
    }
  }

  // return Tensor(operators);
  return Sparse(pow(2, N), pow(2, N));
}

Sparse SigmaPlus(unsigned int N, unsigned int j) {
  std::vector<Sparse> operators;

  for (int i = 0; i < N; ++i) {
    if (i == j) {
      // operators.emplace_back(SigmaPlus());
    } else {
      // operators.emplace_back(Identity(2));
    }
  }

  // return Tensor(operators);
  return Sparse(pow(2, N), pow(2, N));
}

Sparse SigmaMinus(unsigned int N, unsigned int j) {
  std::vector<Sparse> operators;

  for (int i = 0; i < N; ++i) {
    if (i == j) {
      // operators.emplace_back(SigmaMinus());
    } else {
      // operators.emplace_back(Identity(2));
    }
  }

  // return Tensor(operators);
  return Sparse(pow(2, N), pow(2, N));
}
