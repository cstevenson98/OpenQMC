//
// Created by conor on 16/04/22.
//

#include "Open.cuh"
#include "../la/Super.cuh"
#include "Spins.cuh"

Sparse Lindblad(const Sparse& A, const Sparse& B) {
    auto id = Identity(A.DimY);
    auto BA = B * A;

    return 2 * ToSuper(A, B) - ToSuper(BA, id) - ToSuper(id, BA);
}

Sparse SuperComm(const Sparse& A) {
    auto id = Identity(A.DimY);
    return ToSuper(A, id) - ToSuper(id, A);
}