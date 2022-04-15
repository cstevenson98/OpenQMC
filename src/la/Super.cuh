//
// Created by conor on 04/04/2022.
//

#ifndef MAIN_SUPER_CUH
#define MAIN_SUPER_CUH

#include "Sparse.cuh"

Sparse Kronecker(const Sparse& A, const Sparse& B);
Sparse Tensor(const vector<Sparse>& matrices);
Sparse ToSuper(const Sparse& A, const Sparse& B);

#endif //MAIN_SUPER_CUH
