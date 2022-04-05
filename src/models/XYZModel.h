//
// Created by conor on 05/04/2022.
//

#ifndef MAIN_XYZMODEL_H
#define MAIN_XYZMODEL_H

#include "../la/Sparse.h"

// XYZ model of N coupled 2LS
struct XYZModel {
    unsigned int N;
    double g, Delta;

    XYZModel(unsigned int N, double g, double Delta) : N(N), g(g), Delta(Delta) { };

    // Hamiltonian
    Sparse H(bool PBC) const;

    // Derivative of density matrix,
    // with or without dissipator
    static Sparse Dx(bool PBC);
};

#endif //MAIN_XYZMODEL_H
