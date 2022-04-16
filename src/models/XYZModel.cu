//
// Created by conor on 05/04/2022.
//

#include "XYZModel.cuh"
#include "../qm/Spins.cuh"

Sparse XYZModel::H(bool PBC = false) const {
    const int size = pow(N, 2);
    Sparse Hfield(size, size);

    for (int i = 0; i < N; i++) {
        Hfield = Hfield + g*SigmaZ(N, i);
    }
    Hfield.ToDense().PrintRe();

    Sparse Hop(size, size);
    for (int i = 0; i < N - 1; i++) {
        Hop = Hop + ((SigmaPlus(N, i) * SigmaMinus(N, i+1))
                  + (SigmaMinus(N, i) * SigmaPlus(N, i+1)));
    }
    Hop.ToDense().PrintRe();

    return Hfield + Hop;
}

Sparse XYZModel::Dx(bool PBC) {
    return {0, 0};
}
