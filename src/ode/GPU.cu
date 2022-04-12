//
// Created by conor on 12/04/22.
//

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>

#include "GPU.cuh"

struct xpy_functor {
    xpy_functor() = default;
    __host__ __device__
    thrust::complex<double> operator()(const thrust::complex<double>& x,
                                       const thrust::complex<double>& y) const {
        return x + y;
    }
};

struct saxpy_functor {
    const thrust::complex<double> a;

    saxpy_functor(thrust::complex<double> _a) : a(_a) {}

    __host__ __device__
    thrust::complex<double> operator()(const thrust::complex<double>& x,
                                       const thrust::complex<double>& y) const {
        return a * x + y;
    }
};

void xpy_fast(thrust::device_vector<thrust::complex<double>>& X,
              thrust::device_vector<thrust::complex<double>>& Y) {
    // Y <- X + Y
    thrust::transform(X.begin(), X.end(),
                      Y.begin(), Y.begin(),
                      xpy_functor());
}

void saxpy_fast(thrust::complex<double> A,
                thrust::device_vector<thrust::complex<double>>& X,
                thrust::device_vector<thrust::complex<double>>& Y) {
    // Y <- A * X + Y
    thrust::transform(X.begin(), X.end(),
                      Y.begin(), Y.begin(),
                      saxpy_functor(A));
}

__global__ void spmv_ell_kernel(const int num_cols_per_row,
                                const int *indices,
                                thrust::complex<double> *data,
                                thrust::complex<double> *x,
                                thrust::complex<double>       *y)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x ;
    for (int i = 0; i < num_cols_per_row; i++)
    {
        int                     col = indices[i + row * num_cols_per_row];
        thrust::complex<double> val = data   [i + row * num_cols_per_row];

        y[row] = y[row] + val * x[col];
    }
}
