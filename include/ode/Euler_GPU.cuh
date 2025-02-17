//
// Created by conor on 12/04/22.
//

#ifndef MAIN_EULER_GPU_CUH
#define MAIN_EULER_GPU_CUH

#include <thrust/device_vector.h>
#include <thrust/complex.h>

#include "ode/Integrator.cuh"
#include "la/SparseELL.cuh"

using t_cplx        = thrust::complex<double>;
using t_hostVect    = thrust::host_vector<thrust::complex<double>>;
using t_devcVect    = thrust::device_vector<thrust::complex<double>>;
using t_devcVectInt = thrust::device_vector<int>;

class Euler_GPU : public Integrator {
    t_hostVect    x, dx;
    double        t;
    SparseELL     M;
    double        Err, Tol;

    t_devcVect    D_M_Values;
    t_devcVectInt D_M_Indices;
    int           n_columns;
    t_devcVect    D_x;
    t_devcVect    D_dx;

public:
    Euler_GPU(t_hostVect& y0, double t0, SparseELL& M, double tol);

    void State(struct State &dst) override;
    double Step(double step) override;
};


#endif //MAIN_EULER_GPU_CUH
