//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by Conor Stevenson on 15/3/2025.
//

#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <vector>

#include "core/types.cuh"
#include "la/VectImpl.cuh"
#include "la/VectImplGPU.cuh"
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

// Functor for computing conjugate
struct conjugate_functor {
  __host__ __device__ th_cplx operator()(const th_cplx &x) const {
    return thrust::conj(x);
  }
};

// Functor for computing absolute value squared
struct abs_squared_functor {
  __host__ __device__ double operator()(const th_cplx &x) const {
    return thrust::norm(x);
  }
};

// Functor for computing dot product
struct dot_product_functor {
  __host__ __device__ th_cplx operator()(const th_cplx &x,
                                         const th_cplx &y) const {
    return thrust::conj(x) * y;
  }
};

// Constructor with dimension
VectImplGPU::VectImplGPU(int dim) : Dim(dim) {
  // Initialize device data with zeros
  deviceData_.resize(Dim, th_cplx(0.0, 0.0));

  // Initialize cuSPARSE
  InitializeCuSparse();
}

// Destructor
VectImplGPU::~VectImplGPU() {
  // Clean up cuSPARSE resources
  cusparseDestroyDnVec(vecDescr_);
}

// Initialize cuSPARSE resources
void VectImplGPU::InitializeCuSparse() {
  // Create dense vector descriptor
  cusparseCreateDnVec(&vecDescr_, Dim,
                      thrust::raw_pointer_cast(deviceData_.data()), CUDA_C_64F);
}

// Constructor from host vector
VectImplGPU::VectImplGPU(const t_hostVect &in) : Dim(in.size()) {
  // Convert std::complex to thrust::complex
  th_hostVect hostData(in.size());
  for (size_t i = 0; i < in.size(); ++i) {
    hostData[i] = th_cplx(in[i].real(), in[i].imag());
  }

  // Copy to device
  deviceData_ = hostData;

  // Initialize cuSPARSE
  InitializeCuSparse();
}

// Constructor from CPU VectImpl
VectImplGPU::VectImplGPU(const VectImpl &cpuVector) : Dim(cpuVector.size()) {
  // Convert std::complex to thrust::complex
  th_hostVect hostData(cpuVector.size());
  for (size_t i = 0; i < cpuVector.size(); ++i) {
    hostData[i] = th_cplx(cpuVector[i].real(), cpuVector[i].imag());
  }

  // Copy to device
  deviceData_ = hostData;

  // Initialize cuSPARSE
  InitializeCuSparse();
}

VectImplGPU VectImplGPU::Conj() const {
  VectImplGPU out(deviceData_.size());
  thrust::transform(deviceData_.begin(), deviceData_.end(),
                    out.deviceData_.begin(), conjugate_functor());
  return out;
}

VectImplGPU VectImplGPU::Add(const VectImplGPU &A) const {
  VectImplGPU out(deviceData_.size());
  thrust::transform(deviceData_.begin(), deviceData_.end(),
                    A.deviceData_.begin(), out.deviceData_.begin(),
                    thrust::plus<th_cplx>());
  return out;
}

VectImplGPU VectImplGPU::Subtract(const VectImplGPU &A) const {
  VectImplGPU out(deviceData_.size());
  thrust::transform(deviceData_.begin(), deviceData_.end(),
                    A.deviceData_.begin(), out.deviceData_.begin(),
                    thrust::minus<th_cplx>());
  return out;
}

VectImplGPU VectImplGPU::Scale(const th_cplx &alpha) const {
  VectImplGPU out(Dim);

  // Scale values
  thrust::transform(deviceData_.begin(), deviceData_.end(),
                    out.deviceData_.begin(),
                    [alpha] __device__(const th_cplx &x) { return alpha * x; });

  return out;
}

th_cplx VectImplGPU::Dot(const VectImplGPU &A) const {
  // Compute the dot product
  th_cplx result = thrust::inner_product(
      deviceData_.begin(), deviceData_.end(), A.deviceData_.begin(),
      th_cplx(0.0, 0.0), thrust::plus<th_cplx>(), dot_product_functor());

  return std::complex<double>(result.real(), result.imag());
}

double VectImplGPU::Norm() const {
  // Compute the sum of absolute values squared
  double sum = thrust::transform_reduce(deviceData_.begin(), deviceData_.end(),
                                        abs_squared_functor(), 0.0,
                                        thrust::plus<double>());

  // Return the square root
  return std::sqrt(sum);
}

VectImplGPU VectImplGPU::operator+(const VectImplGPU &A) const {
  return this->Add(A);
}

VectImplGPU VectImplGPU::operator-(const VectImplGPU &A) const {
  VectImplGPU out(Dim);

  // Subtract values
  thrust::transform(deviceData_.begin(), deviceData_.end(),
                    A.deviceData_.begin(), out.deviceData_.begin(),
                    thrust::minus<th_cplx>());

  return out;
}

VectImplGPU VectImplGPU::operator*(const th_cplx &alpha) const {
  return this->Scale(alpha);
}

std::complex<double> VectImplGPU::operator[](unsigned int i) const {
  // Copy the element to host
  th_hostVect hostData(1);
  thrust::copy(deviceData_.begin() + i, deviceData_.begin() + i + 1,
               hostData.begin());

  // Convert to std::complex
  return std::complex<double>(hostData[0].real(), hostData[0].imag());
}

void VectImplGPU::Print(unsigned int kind) const {
  // Copy data to host
  th_hostVect hostData = deviceData_;

  std::string s;
  std::stringstream stream;
  stream.setf(std::ios::fixed);
  stream.precision(2);

  stream << " Vector [" << deviceData_.size() << " x " << 1
         << "]:" << std::endl;
  for (int i = 0; i < deviceData_.size(); ++i) {
    stream << "   ";
    std::string spaceCharRe = !std::signbit(hostData[i].real()) ? " " : "";
    std::string spaceCharIm = !std::signbit(hostData[i].imag()) ? " " : "";
    std::string spaceCharAbs =
        !std::signbit(hostData[i].imag()) ? " + " : " - ";

    switch (kind) {
    case 0: // re + im
      stream << spaceCharRe << hostData[i].real() << spaceCharAbs
             << std::abs(hostData[i].imag()) << "i  ";
      break;
    case 1: // re
      stream << spaceCharRe << hostData[i].real() << " ";
      break;
    case 2: // im
      stream << spaceCharIm << hostData[i].imag() << "i  ";
      break;
    case 3: // abs
      stream << " "
             << std::sqrt(hostData[i].real() * hostData[i].real() +
                          hostData[i].imag() * hostData[i].imag());
      break;
    default:
      stream << "[e]";
    }
    stream << std::endl;
  }

  s = stream.str();
  std::cout << s << std::endl;
}

void VectImplGPU::PrintRe() const { this->Print(1); }

void VectImplGPU::PrintIm() const { this->Print(2); }

void VectImplGPU::PrintAbs() const { this->Print(3); }

std::vector<std::complex<double>> VectImplGPU::GetData() const {
  // Copy from device to host
  th_hostVect hostData = deviceData_;
  // Convert to std::vector<std::complex<double>>
  std::vector<std::complex<double>> out(hostData.size());
  for (size_t i = 0; i < hostData.size(); ++i) {
    out[i] = std::complex<double>(hostData[i].real(), hostData[i].imag());
  }

  return out;
}

void VectImplGPU::SetData(const std::vector<std::complex<double>> &data) {
  deviceData_ = data;
}

void VectImplGPU::CopyFromHost(const t_hostVect &hostData) {
  deviceData_ = hostData;
}

t_hostVect VectImplGPU::GetHostData() const {
  t_hostVect hostData(deviceData_.size());

  // copy to hostData
  thrust::copy(deviceData_.begin(), deviceData_.end(), hostData.begin());

  return hostData;
}

VectImplGPU operator*(const th_cplx &alpha, const VectImplGPU &rhs) {
  return rhs * alpha;
}

void VectImplGPU::ConjInPlace(VectImplGPU &output) const {
  // Ensure output has correct size
  if (output.size() != deviceData_.size()) {
    output = VectImplGPU(deviceData_.size());
  }
  thrust::transform(deviceData_.begin(), deviceData_.end(),
                    output.deviceData_.begin(), conjugate_functor());
}

void VectImplGPU::AddInPlace(const VectImplGPU &A, VectImplGPU &output) const {
  // Ensure output has correct size
  if (output.size() != deviceData_.size()) {
    output = VectImplGPU(deviceData_.size());
  }
  thrust::transform(deviceData_.begin(), deviceData_.end(),
                    A.deviceData_.begin(), output.deviceData_.begin(),
                    thrust::plus<th_cplx>());
}

void VectImplGPU::SubtractInPlace(const VectImplGPU &A,
                                  VectImplGPU &output) const {
  // Ensure output has correct size
  if (output.size() != deviceData_.size()) {
    output = VectImplGPU(deviceData_.size());
  }
  thrust::transform(deviceData_.begin(), deviceData_.end(),
                    A.deviceData_.begin(), output.deviceData_.begin(),
                    thrust::minus<th_cplx>());
}

void VectImplGPU::ScaleInPlace(const th_cplx &alpha,
                               VectImplGPU &output) const {
  // Ensure output has correct size
  if (output.size() != deviceData_.size()) {
    output = VectImplGPU(deviceData_.size());
  }

  // Create a functor for scaling
  struct scale_functor {
    const th_cplx alpha;
    scale_functor(const th_cplx &a) : alpha(a) {}
    __host__ __device__ th_cplx operator()(const th_cplx &x) const {
      return alpha * x;
    }
  };

  thrust::transform(deviceData_.begin(), deviceData_.end(),
                    output.deviceData_.begin(), scale_functor(alpha));
}