#include "utils/pimpl_test.h"
#include <memory>

#include <thrust/complex.h>
#include <thrust/host_vector.h>

using t_cplx = thrust::complex<double>;
using t_hostVect = thrust::host_vector<thrust::complex<double>>;

class MyClass::Impl {
public:
    Impl(int size) : data(size) {};
    ~Impl() = default;
private:
    t_hostVect data;
};

MyClass::MyClass(int size) : pimpl(std::make_unique<MyClass::Impl>(size)) {}
MyClass::~MyClass() = default;