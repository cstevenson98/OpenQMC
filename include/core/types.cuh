#include <thrust/complex.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using th_cplx        = thrust::complex<double>;
using th_hostVect    = thrust::host_vector<thrust::complex<double>>;
using t_devcVect    = thrust::device_vector<thrust::complex<double>>;
using t_devcVectInt = thrust::device_vector<int>;
