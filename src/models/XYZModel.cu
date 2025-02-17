//
// Created by conor on 05/04/2022.
//

#include "models/XYZModel.cuh"
#include "qm/Spins.cuh"
#include "qm/Open.cuh"

Sparse XYZModel::H(bool PBC = false) const {
    const int size = pow(2, N);
    Sparse Hfield(size, size);

    for (int i = 0; i < N; i++) {
        Hfield = Hfield + g*SigmaZ(N, i);
    }

    Sparse Hop(size, size);
    for (int i = 0; i < N - 1; i++) {
        Hop = Hop + ((SigmaPlus(N, i) * SigmaMinus(N, i+1))
                  + (SigmaMinus(N, i) * SigmaPlus(N, i+1)));
    }

    return Hfield + Hop;
}

Sparse XYZModel::Dx(bool PBC) const {
    const int size = pow(4, N);

    Sparse Liouvillian(size, size);

    Liouvillian = Liouvillian + t_cplx(0, -1) * SuperComm(this->H(PBC));
    if (kappa != 0.) {
        for (int i = 0; i < N; i++) {
            Liouvillian = Liouvillian
                          + kappa / 2. * Lindblad(SigmaMinus(N, i), SigmaPlus(N, i));
        }
    }

    return Liouvillian;
}
