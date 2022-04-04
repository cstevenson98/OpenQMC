//
// Created by conor on 04/04/2022.
//

#include "ElasticChain1D.h"

Sparse ElasticChain1D::Dx() {
    Sparse out(2*N, 2*N);

    for (int i = 0; i < 2*N; ++i) {
        for (int j = 0; j < 2*N; ++j) {
            if (i == j) {
                out.Data.emplace_back(i, j, -kappa);
            }

            if (j+1 == i && j%2 == 0) {
                out.Data.emplace_back(i, j, delta);
            }

            if (j == i+1 && j%2 == 0) {
                out.Data.emplace_back(i, j, -delta);
            }
        }
    }

    out.SortByRow();
    return out;
}
