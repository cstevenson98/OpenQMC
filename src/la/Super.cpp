//
// Created by conor on 04/04/2022.
//

#include "Sparse.h"
#include "Super.h"

Sparse Kronecker(const Sparse& A, const Sparse& B) {
    Sparse out(A.DimX * B.DimX, A.DimY * B.DimY);

    for (auto elemA : A.Data) {
        for (auto elemB : B.Data) {
            out.Data.emplace_back(
                    B.DimX*elemA.Coords[0] + elemB.Coords[0],
                    B.DimY*elemA.Coords[1] + elemB.Coords[1],
                    elemA.Val * elemB.Val);
        }
    }

    return out;
}