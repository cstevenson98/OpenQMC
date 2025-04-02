//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by conor on 05/04/2022.
//

#ifndef MAIN_SPINS_CUH
#define MAIN_SPINS_CUH

#include "la/Sparse.h"

// Defines 2 level systems
SparseImpl Identity(unsigned int N);

// Single spin
Sparse SigmaX();
Sparse SigmaY();
Sparse SigmaZ();
Sparse SigmaPlus();
Sparse SigmaMinus();

// Many spins, on site j
Sparse SigmaX(unsigned int N, unsigned int j);
Sparse SigmaY(unsigned int N, unsigned int j);
Sparse SigmaZ(unsigned int N, unsigned int j);
Sparse SigmaPlus(unsigned int N, unsigned int j);
Sparse SigmaMinus(unsigned int N, unsigned int j);

#endif // MAIN_SPINS_CUH
