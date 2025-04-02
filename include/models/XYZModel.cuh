//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by conor on 05/04/2022.
//

#ifndef MAIN_XYZMODEL_CUH
#define MAIN_XYZMODEL_CUH

#include "la/Sparse.h"

// XYZ model of N coupled 2LS
struct XYZModel {
    unsigned int N;
    double g, Delta;
    double kappa;

    XYZModel(unsigned int N, double g, double Delta, double kappa)
            : N(N), g(g), Delta(Delta), kappa(kappa) { };

    // Hamiltonian
    Sparse H(bool PBC) const;

    // Derivative of density matrix,
    // with or without dissipator
    Sparse Dx(bool PBC) const;
};

#endif //MAIN_XYZMODEL_CUH
