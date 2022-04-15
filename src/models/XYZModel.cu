//
// Created by conor on 05/04/2022.
//

#include "XYZModel.cuh"
#include "../qm/Spins.cuh"

Sparse XYZModel::H(bool PBC) const {
    const int size = pow(N, 2);
    Sparse out(size, size);

    Sparse term(size, size);
    for (int i = 0; i < N; ++i) {
        term = SigmaZ(N, i).Scale(g);
        term.ToDense().Print();
        out = out.Add(term);
    }

    for (int i = 0; i < N - 1; ++i) {
        term = SigmaPlus(N, i).RightMult(SigmaMinus(N, i+1));
        term = term.Add(SigmaMinus(N, i).RightMult(SigmaPlus(N, i+1)));
        term.ToDense().Print();
        out = out.Add(term);

        term = SigmaPlus(N, i).RightMult(SigmaPlus(N, i+1));
        term = term.Add(SigmaMinus(N, i).RightMult(SigmaMinus(N, i+1)));
        term = term.Scale(Delta);
        term.ToDense().Print();
        out = out.Add(term);
    }

    if (PBC) {
        term = term = SigmaPlus(N, N-1).RightMult(SigmaMinus(N, 0));
        term = term.Add(SigmaMinus(N, N-1).RightMult(SigmaPlus(N, 0)));
        term.ToDense().Print();
        out = out.Add(term);

        term = SigmaPlus(N, N-1).RightMult(SigmaPlus(N, 0));
        term = term.Add(SigmaMinus(N, N-1).RightMult(SigmaMinus(N, 0)));
        term = term.Scale(Delta);
        term.ToDense().Print();
        out = out.Add(term);
    }

    return out;
}

Sparse XYZModel::Dx(bool PBC) {
    return {0, 0};
}
