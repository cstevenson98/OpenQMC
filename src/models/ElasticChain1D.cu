//
// Created by conor on 04/04/2022.
//

#include "models/ElasticChain1D.cuh"

Sparse ElasticChain1D::Dx() {
    Sparse out(2*N, 2*N);

    for (int i = 0; i < 2*N; ++i) {
        if (i%2 == 0) {
            out.Data.emplace_back(i, i+1, 1);
        }

        if (i%2 == 1) {
            if (i > 2) {
                out.Data.emplace_back(i, i-3, delta);
            }
//            else {
//                out.Data.emplace_back(i, 2*N-1, delta);
//            }

            out.Data.emplace_back(i, i-1, -w0);
            out.Data.emplace_back(i, i, -kappa);

            if (i < 2*N-1) {
                out.Data.emplace_back(i, i+1, delta);
            }
//            else {
//                out.Data.emplace_back(i, 0, delta);
//            }
        }
    }

    out.SortByRow();
    return out;
}
