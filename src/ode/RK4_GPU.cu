//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by conor on 16/04/22.
//
#include "ode/RK4_GPU.cuh"
#include "ode/GPU.cuh"

using t_cplx        = thrust::complex<double>;
using t_hostVect    = thrust::host_vector<thrust::complex<double>>;
using t_devcVect    = thrust::device_vector<thrust::complex<double>>;
using t_devcVectInt = thrust::device_vector<int>;


RK4_GPU::RK4_GPU(t_hostVect &y0, double t0, SparseELL &M, double tol)
    : k1(y0.size(), 0), k2(y0.size(), 0),
      k3(y0.size(), 0), k4(y0.size(), 0)

{
    D_M_Values  = M.Values.FlattenedData();
    D_M_Indices = M.Indices.FlattenedDataInt();
    n_columns   = M.EntriesPerRow;

    x = y0;
    thrust::copy(x.begin(), x.end(), D_x.begin());
}

void RK4_GPU::State(struct State &dst) {

}

double RK4_GPU::Step(double step) {
    t_cplx* D_x_ = thrust::raw_pointer_cast( D_x.data() );

    t_cplx* k1_ = thrust::raw_pointer_cast( k1.data() );
    t_cplx* k2_ = thrust::raw_pointer_cast( k2.data() );
    t_cplx* k3_ = thrust::raw_pointer_cast( k3.data() );
    t_cplx* k4_ = thrust::raw_pointer_cast( k4.data() );

    t_cplx* D_MValues_  = thrust::raw_pointer_cast( D_M_Values.data() );
    int*    D_MIndices_ = thrust::raw_pointer_cast( D_M_Indices.data() );

    return 0;
}

