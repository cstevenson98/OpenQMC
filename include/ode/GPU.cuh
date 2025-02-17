//
// Created by conor on 12/04/22.
//

#ifndef MAIN_GPU_CUH
#define MAIN_GPU_CUH

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>

#include "la/SparseELL.cuh"

using t_cplx        = thrust::complex<double>;
using t_hostVect    = thrust::host_vector<thrust::complex<double>>;
using t_devcVect    = thrust::device_vector<thrust::complex<double>>;
using t_devcVectInt = thrust::device_vector<int>;


void xpy_fast(t_devcVect& X, t_devcVect& Y);
void saxpy_fast(t_cplx A, t_devcVect& X, t_devcVect& Y);
void say_fast(t_cplx A, t_devcVect& X);

__global__ void spmv_ell_kernel(const int num_rows, const int num_cols_per_row,
                                const int *indices, t_cplx *data,
                                t_cplx *x, t_cplx *y);

void SPMV_ELL_CALL(const SparseELL& M, Vect& v);

#endif //MAIN_GPU_CUH
