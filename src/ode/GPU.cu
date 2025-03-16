//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by conor on 12/04/22.
//

#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "ode/GPU.cuh"

struct xpy_functor {
  xpy_functor() = default;

  __host__ __device__ th_cplx operator()(const th_cplx &x,
                                         const th_cplx &y) const {
    return x + y;
  }
};

struct saxpy_functor {
  const th_cplx a;

  explicit saxpy_functor(t_cplx _a) : a(_a) {}
  __host__ __device__ th_cplx operator()(const th_cplx &x,
                                         const th_cplx &y) const {
    return a * x + y;
  }
};

struct say_functor {
  const th_cplx a;

  explicit say_functor(th_cplx _a) : a(_a) {}
  __host__ __device__ th_cplx operator()(const th_cplx &x,
                                         const th_cplx &y) const {
    return a * y;
  }
};

void xpy_fast(t_devcVect &X, t_devcVect &Y) {
  // Y <- X + Y
  thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), xpy_functor());
}

void saxpy_fast(t_cplx A, t_devcVect &X, t_devcVect &Y) {
  // Y <- A * X + Y
  thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), saxpy_functor(A));
}

void say_fast(t_cplx A, t_devcVect &X) {
  // Y <- a Y
  thrust::transform(X.begin(), X.end(), X.begin(), X.begin(), say_functor(A));
}

void SPMV_ELL_CALL(const SparseELL &M, Vect &v) {
  t_devcVect D_M_Values;
  t_devcVectInt D_M_Indices;

  t_devcVect D_Vect;
  t_devcVect D_MdotVect;

  D_M_Values = M.Values.FlattenedData();
  // D_M_Indices = M.Indices.FlattenedDataInt();
  D_Vect = v.GetData();
  D_MdotVect = v.GetData();

  th_cplx *D_MValuesArray = thrust::raw_pointer_cast(D_M_Values.data());
  int *D_MIndicesArray = thrust::raw_pointer_cast(D_M_Indices.data());
  th_cplx *D_xArray = thrust::raw_pointer_cast(D_Vect.data());
  th_cplx *D_dxArray = thrust::raw_pointer_cast(D_MdotVect.data());

  unsigned int numThreads;
  unsigned int numBlocks;

  numThreads = 128;
  numBlocks = ceil(v.size() / 128.);

    spmv_ell_kernel<<< numThreads, numBlocks >>>(v.size(),
                                            M.EntriesPerRow,
                                            D_MIndicesArray,
                                            D_MValuesArray,
                                            D_xArray,
                                            D_dxArray);

  thrust::copy(D_MdotVect.begin(), D_MdotVect.end(), D_Vect.begin());
}

__global__ void spmv_ell_kernel(const int num_rows, const int num_cols_per_row,
                                const int *indices, th_cplx *data, th_cplx *x,
                                th_cplx *y) {
  int row = blockDim.x * blockIdx.x + threadIdx.x;
  if (row < num_rows) {
    th_cplx dot = 0;
    for (int n = 0; n < num_cols_per_row; n++) {
      int col = indices[n + row * num_cols_per_row];
      th_cplx val = data[n + row * num_cols_per_row];

      if (val != 0 && col > -1)
        dot += val * x[col];
    }

    y[row] = dot;
  }
}

struct square {
  __host__ __device__ double operator()(const t_cplx &x) const {
    return abs(x) * abs(x);
  }
};

double Norm(const t_devcVect &v) {
  square unary_op;
  thrust::plus<double> binary_op;

  return 0;
  // return std::sqrt( thrust::transform_reduce(v.begin(), v.end(), unary_op, 0,
  // binary_op) );
}