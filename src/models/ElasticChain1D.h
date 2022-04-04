//
// Created by conor on 04/04/2022.
//

#ifndef MAIN_ELASTICCHAIN1D_H
#define MAIN_ELASTICCHAIN1D_H


#include "../la/Sparse.h"

class ElasticChain1D {
    int N;
    double kappa, delta;

public:
    ElasticChain1D(int n, double kappa, double delta) : N(n), kappa(kappa), delta(delta) { };

    Sparse Dx();
};


#endif //MAIN_ELASTICCHAIN1D_H
