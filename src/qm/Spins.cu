////
//// Created by conor on 05/04/2022.
////

#include "Spins.cuh"
#include "../la/Super.cuh"

Sparse Identity(unsigned int N) {
    Sparse out(N, N);
    for (int i = 0; i < N; ++i) {
        out.Data.emplace_back(i, i, 1);
    }

    return out;
}

Sparse SigmaX() {
    Dense out(2, 2);

    vector<vector<complex<double>>> sX = {{0,1},{1,0}};
    out.Data = sX;

    return ToSparseCOO(out);
}

Sparse SigmaY() {
    Dense out(2, 2);

    vector<vector<complex<double>>> sY = {{0, {0,-1}}, {{0, 1}, 0}};
    out.Data = sY;

    return ToSparseCOO(out);
}

Sparse SigmaZ() {
    Dense out(2, 2);

    vector<vector<complex<double>>> sZ = {{1, 0}, {0, -1}};
    out.Data = sZ;

    return ToSparseCOO(out);
}

Sparse SigmaPlus() {
    Dense out(2, 2);

    vector<vector<complex<double>>> sP = {{0, 1}, {0, 0}};
    out.Data = sP;

    return ToSparseCOO(out);
}

Sparse SigmaMinus() {
    Dense out(2, 2);

    vector<vector<complex<double>>> sM = {{0, 0}, {1, 0}};
    out.Data = sM;

    return ToSparseCOO(out);
}

Sparse SigmaX(unsigned int N, unsigned int j) {
    vector<Sparse> operators;

    for (int i = 0; i < N; ++i) {
        if (i == j) {
            operators.emplace_back(SigmaX());
        } else {
            operators.emplace_back(Identity(2));
        }
    }

    return Tensor(operators);
}

Sparse SigmaY(unsigned int N, unsigned int j) {
    vector<Sparse> operators;

    for (int i = 0; i < N; ++i) {
        if (i == j) {
            operators.emplace_back(SigmaY());
        } else {
            operators.emplace_back(Identity(2));
        }
    }

    return Tensor(operators);
}

Sparse SigmaZ(unsigned int N, unsigned int j) {
    vector<Sparse> operators;

    for (int i = 0; i < N; ++i) {
        if (i == j) {
            operators.emplace_back(SigmaZ());
        } else {
            operators.emplace_back(Identity(2));
        }
    }

    return Tensor(operators);
}

Sparse SigmaPlus(unsigned int N, unsigned int j) {
    vector<Sparse> operators;

    for (int i = 0; i < N; ++i) {
        if (i == j) {
            operators.emplace_back(SigmaPlus());
        } else {
            operators.emplace_back(Identity(2));
        }
    }

    return Tensor(operators);
}

Sparse SigmaMinus(unsigned int N, unsigned int j) {
    vector<Sparse> operators;

    for (int i = 0; i < N; ++i) {
        if (i == j) {
            operators.emplace_back(SigmaMinus());
        } else {
            operators.emplace_back(Identity(2));
        }
    }

    return Tensor(operators);
}
