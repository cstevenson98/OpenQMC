//
// Created by conor on 16/04/22.
//

#ifndef MAIN_OPEN_CUH
#define MAIN_OPEN_CUH

#include "../la/Sparse.cuh"

Sparse Lindblad(const Sparse& A, const Sparse& B);
Sparse SuperComm(const Sparse& A);

#endif //MAIN_OPEN_CUH
