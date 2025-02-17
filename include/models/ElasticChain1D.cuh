//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by conor on 04/04/2022.
//

#ifndef MAIN_ELASTICCHAIN1D_CUH
#define MAIN_ELASTICCHAIN1D_CUH


#include "la/Sparse.cuh"

class ElasticChain1D {
    int N;
    double w0, kappa, delta;

public:
    ElasticChain1D(int n, double w0, double kappa, double delta) : 
    N(n), w0(w0),kappa(kappa), delta(delta) { };

    Sparse Dx();
};


#endif //MAIN_ELASTICCHAIN1D_CUH
