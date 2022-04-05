//
// Created by conor on 05/04/2022.
//

#include "Spins.h"
#include "../la/Super.h"

Sparse Identity(unsigned int N) {
    Sparse out(N, N);
    for (int i = 0; i < N; ++i) {
        out.Data.emplace_back(i, i, 1);
    }

    return out;
}

Sparse SigmaX() {
    Dense out(2, 2);
    out.Data = {{0, 1}, {1, 0}};
    return ToSparseCOO(out);
}

Sparse SigmaY() {
    Dense out(2, 2);
    out.Data = {{0, {0,-1}}, {{0, 1}, 0}};
    return ToSparseCOO(out);
}

Sparse SigmaZ() {
    Dense out(2, 2);
    out.Data = {{1, 0}, {0, -1}};
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
