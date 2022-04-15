//
// Created by conor on 12/04/22.
//

#ifndef MAIN_EULER_GPU_CUH
#define MAIN_EULER_GPU_CUH

#include <thrust/device_vector.h>
#include <thrust/complex.h>
#include <cmath>

#include "Integrator.cuh"
#include "../la/SparseELL.cuh"

class Euler_GPU : public Integrator {
    thrust::host_vector<thrust::complex<double> > x, dx;
    double t;
    SparseELL M;
    double Err, Tol;

    thrust::device_vector<thrust::complex<double> > D_M_Values;
    thrust::device_vector<int>                      D_M_Indices;
    int                                             n_columns;
    thrust::device_vector<thrust::complex<double> > D_x;
    thrust::device_vector<thrust::complex<double> > D_dx;

public:
    Euler_GPU(thrust::host_vector<thrust::complex<double> >& y0,
              double t0,
              SparseELL& M,
              double tol);

    void State(struct State &dst) override;
    double Step(double step) override;
};


#endif //MAIN_EULER_GPU_CUH
