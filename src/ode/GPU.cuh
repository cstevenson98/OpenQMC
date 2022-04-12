//
// Created by conor on 12/04/22.
//

#ifndef MAIN_GPU_CUH
#define MAIN_GPU_CUH

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>

void xpy_fast(thrust::device_vector<thrust::complex<double>>& X,
              thrust::device_vector<thrust::complex<double>>& Y);

void saxpy_fast(thrust::complex<double> A,
                thrust::device_vector<thrust::complex<double>>& X,
                thrust::device_vector<thrust::complex<double>>& Y);

__global__ void spmv_ell_kernel(int num_cols_per_row,
                                const int *indices,
                                thrust::complex<double> *data,
                                thrust::complex<double> *x,
                                thrust::complex<double> *y);

#endif //MAIN_GPU_CUH
