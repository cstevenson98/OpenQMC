//
// Created by conor on 12/04/22.
//

#include <thrust/host_vector.h>

#include "Euler_GPU.cuh"
#include "Integrator.cuh"
#include "GPU.cuh"

using t_cplx        = thrust::complex<double>;
using t_hostVect    = thrust::host_vector<thrust::complex<double>>;
using t_devcVect    = thrust::device_vector<thrust::complex<double>>;
using t_devcVectInt = thrust::device_vector<int>;

Euler_GPU::Euler_GPU(t_hostVect& y0, double t0, SparseELL& M, double tol)
                     : Tol(tol), D_x(y0.size(), 0), D_dx(y0.size(), 0) {

    D_M_Values  = M.Values.FlattenedData();
    D_M_Indices = M.Indices.FlattenedDataInt();
    n_columns   = M.EntriesPerRow;

    x = y0;
    thrust::copy(x.begin(), x.end(), D_x.begin());
    thrust::copy(x.begin(), x.end(), D_dx.begin());

    t = t0;
    Err = 0;
}

void Euler_GPU::State(struct State &dst) {
    thrust::copy(D_x.begin(), D_x.end(), dst.Y.Data.begin());
    dst.E = Err;
    dst.T = t;
}

double Euler_GPU::Step(double step) {
    t_cplx* D_MValuesArray  = thrust::raw_pointer_cast( D_M_Values.data() );
    int*    D_MIndicesArray = thrust::raw_pointer_cast( D_M_Indices.data() );
    t_cplx* D_xArray        = thrust::raw_pointer_cast( D_x.data() );
    t_cplx* D_dxArray       = thrust::raw_pointer_cast( D_dx.data() );

    unsigned int numThreads = 256;
    unsigned int numBlocks = ceil(x.size()/256.);

    spmv_ell_kernel<<< numThreads, numBlocks >>>(
            x.size(), n_columns, D_MIndicesArray, D_MValuesArray, D_xArray, D_dxArray);

    saxpy_fast(step, D_dx, D_x);

    t += step;
    return step;
}
