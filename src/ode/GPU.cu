//
// Created by conor on 12/04/22.
//

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>

#include "GPU.cuh"

using t_cplx        = thrust::complex<double>;
using t_hostVect    = thrust::host_vector<thrust::complex<double>>;
using t_devcVect    = thrust::device_vector<thrust::complex<double>>;
using t_devcVectInt = thrust::device_vector<int>;

struct xpy_functor {
    xpy_functor() = default;

    __host__ __device__
        t_cplx operator()(const t_cplx& x, const t_cplx& y) const {
            return x + y;
        }
};

struct saxpy_functor {
    const t_cplx a;

    explicit saxpy_functor(t_cplx _a) : a(_a) {}
    __host__ __device__
        t_cplx operator()(const t_cplx& x, const t_cplx& y) const {
            return a * x + y;
        }
};

struct say_functor {
    const t_cplx a;

    explicit say_functor(t_cplx _a) : a(_a) {}
    __host__ __device__
        t_cplx operator()(const t_cplx& x, const t_cplx& y) const {
            return a * y;
        }
};

void xpy_fast(t_devcVect& X, t_devcVect& Y) {
    // Y <- X + Y
    thrust::transform(X.begin(), X.end(),
                      Y.begin(), Y.begin(),
                      xpy_functor());
}

void saxpy_fast(t_cplx A, t_devcVect& X, t_devcVect& Y) {
    // Y <- A * X + Y
    thrust::transform(X.begin(), X.end(),
                      Y.begin(), Y.begin(),
                      saxpy_functor(A));
}

void say_fast(t_cplx A, t_devcVect& X) {
    // Y <- a Y
    thrust::transform(X.begin(), X.end(),
                      X.begin(), X.begin(),
                      say_functor(A));
}

void SPMV_ELL_CALL(const SparseELL& M, Vect& v) {
    t_devcVect    D_M_Values;
    t_devcVectInt D_M_Indices;

    t_devcVect    D_Vect;
    t_devcVect    D_MdotVect;

    D_M_Values  = M.Values.FlattenedData();
    D_M_Indices = M.Indices.FlattenedDataInt();
    D_Vect      = v.Data;
    D_MdotVect  = v.Data;

    t_cplx* D_MValuesArray  = thrust::raw_pointer_cast( D_M_Values.data() );
    int*    D_MIndicesArray = thrust::raw_pointer_cast( D_M_Indices.data() );
    t_cplx* D_xArray        = thrust::raw_pointer_cast( D_Vect.data() );
    t_cplx* D_dxArray       = thrust::raw_pointer_cast( D_MdotVect.data() );

    unsigned int numThreads;
    unsigned int numBlocks;

    numThreads = 128;
    numBlocks = ceil(v.Data.size()/128.);

    spmv_ell_kernel<<< numThreads, numBlocks >>>(v.Data.size(),
                                            M.EntriesPerRow,
                                            D_MIndicesArray,
                                            D_MValuesArray,
                                            D_xArray,
                                            D_dxArray);

    thrust::copy(D_MdotVect.begin(), D_MdotVect.end(), v.Data.begin());
}

__global__
    void spmv_ell_kernel(const int num_rows, const int num_cols_per_row,
                         const int *indices, t_cplx *data,
                         t_cplx *x, t_cplx *y)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < num_rows)
    {
        t_cplx dot = 0;
        for (int n = 0; n < num_cols_per_row; n++) {
            int col = indices[n + row * num_cols_per_row];
            t_cplx val = data[n + row * num_cols_per_row];

            if (val != 0 && col > -1)
                dot += val * x[col];
        }

        y[row] = dot;
    }
}

struct square
{
    __host__ __device__
    double operator()(const t_cplx& x) const {
        return abs(x) * abs(x);
    }
};

double Norm(const t_devcVect& v) {
    square               unary_op;
    thrust::plus<double> binary_op;

    return std::sqrt( thrust::transform_reduce(v.begin(), v.end(), unary_op, 0, binary_op) );
}