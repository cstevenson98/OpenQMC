//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by conor on 16/04/22.
//

#ifndef MAIN_OPEN_CUH
#define MAIN_OPEN_CUH

#include "la/Sparse.cuh"

Sparse Lindblad(const Sparse& A);
Sparse Lindblad(const Sparse& A, const Sparse& B);
Sparse SuperComm(const Sparse& A);

#endif //MAIN_OPEN_CUH
