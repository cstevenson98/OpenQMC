#include "la/VectImpl.cuh"
#include "la/VectImplGPU.cuh"

void VectImpl::SetData(const std::vector<std::complex<double>> &data) {
  // Resize the Eigen vector if needed
  if (Data.size() != data.size()) {
    Data.resize(data.size());
  }

  // Copy data from std::vector to Eigen vector
  for (size_t i = 0; i < data.size(); ++i) {
    Data(i) = data[i];
  }
}