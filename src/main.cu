//
// Created by Conor Stevenson on 03/04/2022.
//

#include <iostream>
#include "cuComplex.h"
#include "qm/Spins.h"
#include "la/SparseELL.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>

#define N 6

__global__ void spmv_ell_kernel(const int num_cols_per_row, 
                                const int *indices, 
                                const thrust::complex<double> *data, 
                                const thrust::complex<double> *x, 
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

int main()
{
    thrust::device_vector<thrust::complex<double> > D_Mvals(pow(4., N), 1);
    thrust::device_vector<int>                      D_Mindices(pow(4., N), 1);
    thrust::device_vector<thrust::complex<double> > D_Xvals(pow(2., N), thrust::complex<double>(0.5));
    thrust::device_vector<thrust::complex<double> > D_Yvals(pow(2., N), 0);

    thrust::host_vector<thrust::complex<double> > Mvals = ToSparseELL(SigmaX(N, 1)).Values.FlattenedData();
    thrust::host_vector<int>                      Mindices = ToSparseELL(SigmaX(N, 1)).Indices.FlattenedDataInt();

    // copy all of H back to the beginning of D
    thrust::copy(Mvals.begin(),    Mvals.end(),    D_Mvals.begin());
    thrust::copy(Mindices.begin(), Mindices.end(), D_Mindices.begin());

    // print D 
    // for(int i = 0; i < Mvals.size(); i++)
    //     std::cout << "D[" << i << "] = " << D_Mvals[i] << std::endl;
    // for(int i = 0; i < Mindices.size(); i++)
    //     std::cout << "D[" << i << "] = " << D_Mindices[i] << std::endl;

    thrust::complex<double>* D_MvalsArray = thrust::raw_pointer_cast( D_Mvals.data() );
    int*                     D_MindicesArray = thrust::raw_pointer_cast( D_Mindices.data() );
    thrust::complex<double>* D_XvalsArray = thrust::raw_pointer_cast( D_Xvals.data() );
    thrust::complex<double>* D_YvalsArray = thrust::raw_pointer_cast( D_Yvals.data() );

    spmv_ell_kernel<<<pow(2., N), 1>>>(1, D_MindicesArray, D_MvalsArray, D_XvalsArray, D_YvalsArray);
    // thrust::copy(D_Mvals.begin(),    D_Mvals.end(),    Mvals.begin());

    // for(int i = 0; i < D_Yvals.size(); i++)
    //     std::cout << "D_Yvals[" << i << "] = " << D_Yvals[i] << std::endl;
    cudaDeviceSynchronize();

    return 0;
}
