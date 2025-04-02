//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by conor on 12/04/22.
//

#ifndef MAIN_GPU_CUH
#define MAIN_GPU_CUH

#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "core/types.cuh"
#include "la/SparseELL.h"
#include "la/Vect.h"

void xpy_fast(t_devcVect &X, t_devcVect &Y);
void saxpy_fast(th_cplx A, t_devcVect &X, t_devcVect &Y);
void say_fast(th_cplx A, t_devcVect &X);

__global__ void spmv_ell_kernel(const int num_rows, const int num_cols_per_row,
                                const int *indices, th_cplx *data, th_cplx *x,
                                th_cplx *y);

void SPMV_ELL_CALL(const SparseELL &M, Vect &v);

#endif // MAIN_GPU_CUH
