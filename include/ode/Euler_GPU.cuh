//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by conor on 12/04/22.
//

#ifndef MAIN_EULER_GPU_CUH
#define MAIN_EULER_GPU_CUH

#include <thrust/device_vector.h>
#include <thrust/complex.h>

#include "ode/Integrator.cuh"
#include "la/SparseELL.cuh"

class Euler_GPU : public Integrator {
    th_hostVect    x, dx;
    double        t;
    SparseELL     M;
    double        Err, Tol;

    t_devcVect    D_M_Values;
    t_devcVectInt D_M_Indices;
    int           n_columns;
    t_devcVect    D_x;
    t_devcVect    D_dx;

public:
    Euler_GPU(th_hostVect& y0, double t0, SparseELL& M, double tol);

    void State(struct State &dst) override;
    double Step(double step) override;
};


#endif //MAIN_EULER_GPU_CUH
