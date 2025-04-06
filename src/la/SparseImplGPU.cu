//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by Conor Stevenson on 15/3/2025.
//

#include "la/SparseImpl.cuh"
#include "la/SparseImplGPU.cuh"
#include <cusparse.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#define CUSPARSE_CHECK(call)                                                   \
  do {                                                                         \
    cusparseStatus_t status = call;                                            \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
      printf("cuSPARSE error at %s:%d: %d\n", __FILE__, __LINE__, status);     \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// Constructor with dimensions
SparseImplGPU::SparseImplGPU(int dimX, int dimY)
    : DimX(dimX), DimY(dimY), nnz(0), spmvBufferSize_(0) {
  // Initialize row pointers with zeros
  rowPtr.resize(DimX + 1, 0);

  // Initialize cuSPARSE
  InitializeCuSparse();
}

// Destructor
SparseImplGPU::~SparseImplGPU() {
  // Clean up cuSPARSE resources
  cusparseDestroySpMat(matDescr_);
}

// Initialize cuSPARSE resources
void SparseImplGPU::InitializeCuSparse() {
  // Create matrix descriptor
  CUSPARSE_CHECK(cusparseCreateCsr(&matDescr_,
                                   DimX,                // rows
                                   DimY,                // cols
                                   values.size(),       // nnz
                                   rowPtr.data().get(), // row offsets array
                                   colInd.data().get(), // column indices array
                                   values.data().get(), // values array
                                   CUSPARSE_INDEX_32I,  // row offsets type
                                   CUSPARSE_INDEX_32I,  // column indices type
                                   CUSPARSE_INDEX_BASE_ZERO, // index base
                                   CUDA_C_64F)); // values type (complex double)
}

// Constructor from host matrix
SparseImplGPU::SparseImplGPU(const t_hostMat &in)
    : DimX(in.size()), DimY(in[0].size()), nnz(0), spmvBufferSize_(0) {
  // Count non-zero elements
  for (int i = 0; i < DimX; ++i) {
    for (int j = 0; j < DimY; ++j) {
      if (std::abs(in[i][j]) > 1e-10) {
        nnz++;
      }
    }
  }

  // Allocate memory for CSR format
  rowPtr.resize(DimX + 1, 0);
  colInd.resize(nnz);
  values.resize(nnz);

  // Convert to CSR format
  int idx = 0;
  for (int i = 0; i < DimX; ++i) {
    rowPtr[i] = idx;
    for (int j = 0; j < DimY; ++j) {
      if (std::abs(in[i][j]) > 1e-10) {
        colInd[idx] = j;
        values[idx] = th_cplx(in[i][j].real(), in[i][j].imag());
        idx++;
      }
    }
  }
  rowPtr[DimX] = nnz;

  // Initialize cuSPARSE
  InitializeCuSparse();
}

// Constructor from CPU SparseImpl
SparseImplGPU::SparseImplGPU(const SparseImpl &cpuMatrix)
    : DimX(cpuMatrix.DimX), DimY(cpuMatrix.DimY),
      nnz(cpuMatrix.CPUData.nonZeros()), spmvBufferSize_(0) {
  // Convert from Eigen sparse matrix to CSR format
  ConvertFromEigen(cpuMatrix);

  // Initialize cuSPARSE
  InitializeCuSparse();
}

// Convert from Eigen sparse matrix to CSR format
void SparseImplGPU::ConvertFromEigen(const SparseImpl &eigenMatrix) {
  // Get the number of non-zero elements
  nnz = eigenMatrix.CPUData.nonZeros();

  // Allocate memory for CSR format
  rowPtr.resize(DimX + 1, 0);
  colInd.resize(nnz);
  values.resize(nnz);

  // Convert to CSR format
  int idx = 0;
  for (int i = 0; i < DimX; ++i) {
    rowPtr[i] = idx;
    for (typename t_eigenSparseMat::InnerIterator it(eigenMatrix.CPUData, i);
         it; ++it) {
      colInd[idx] = it.col();
      values[idx] = th_cplx(it.value().real(), it.value().imag());
      idx++;
    }
  }
  rowPtr[DimX] = nnz;
}

// Scale the matrix by a scalar
SparseImplGPU SparseImplGPU::Scale(const th_cplx &alpha) const {
  SparseImplGPU out(DimX, DimY);

  // Copy row pointers and column indices
  out.rowPtr = rowPtr;
  out.colInd = colInd;
  out.nnz = nnz;

  // Scale values
  out.values.resize(nnz);
  thrust::transform(values.begin(), values.end(), out.values.begin(),
                    [alpha] __device__(const th_cplx &x) { return alpha * x; });

  return out;
}

// Add two matrices
SparseImplGPU SparseImplGPU::Add(const SparseImplGPU &B) const {
  // Create output matrix
  SparseImplGPU out(DimX, DimY);

  // Allocate memory for CSR format
  out.rowPtr.resize(DimX + 1, 0);

  // Count non-zero elements in the result
  const int *d_rowPtrA = GetRowPtr();
  const int *d_colIndA = GetColIndPtr();
  const th_cplx *d_valuesA = GetValuesPtr();

  const int *d_rowPtrB = B.GetRowPtr();
  const int *d_colIndB = B.GetColIndPtr();
  const th_cplx *d_valuesB = B.GetValuesPtr();

  // Use cuSPARSE to compute the number of non-zero elements
  int *d_nnzTotalDevHostPtr = nullptr;
  cudaMalloc(&d_nnzTotalDevHostPtr, sizeof(int));

  // Note: We need to use the correct cuSPARSE function for complex matrices
  // For simplicity, we'll use a placeholder here
  // In a real implementation, you would use the appropriate cuSPARSE function
  // such as cusparseZcsrgeam2 for complex matrices

  // For now, we'll use a simple approach to estimate the number of non-zero
  // elements
  out.nnz = nnz + B.nnz; // This is an upper bound

  // Allocate memory for the result
  out.colInd.resize(out.nnz);
  out.values.resize(out.nnz);

  // Use cuSPARSE to compute the sum
  th_cplx alpha = th_cplx(1.0, 0.0);
  th_cplx beta = th_cplx(1.0, 0.0);

  // Note: We need to use the correct cuSPARSE function for complex matrices
  // For simplicity, we'll use a placeholder here
  // In a real implementation, you would use the appropriate cuSPARSE function
  // such as cusparseZcsrgeam2 for complex matrices

  return out;
}

// Multiply two matrices
SparseImplGPU SparseImplGPU::RightMult(const SparseImplGPU &other) const {
  SparseImplGPU out(DimX, other.DimY);

  // Get raw pointers for matrix A
  const int *rowPtrA = GetRowPtr();
  const int *colIndA = GetColIndPtr();
  const th_cplx *valuesA = GetValuesPtr();

  // Get raw pointers for matrix B (this)
  const int *rowPtrB = other.GetRowPtr();
  const int *colIndB = other.GetColIndPtr();
  const th_cplx *valuesB = other.GetValuesPtr();

  // Compute the number of non-zero elements in the result
  size_t bufferSize_ = 0;
  cusparseSpGEMMDescr_t spgemmDesc_;
  CUSPARSE_CHECK(cusparseSpGEMM_createDescr(&spgemmDesc_));

  // Create alpha and beta values
  th_cplx alpha_(1.0, 0.0);
  th_cplx beta_(0.0, 0.0);

  // Get buffer size
  CUSPARSE_CHECK(cusparseSpGEMM_workEstimation(
      GetHandle(), CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_, matDescr_, other.matDescr_,
      &beta_, out.matDescr_, CUDA_C_64F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc_,
      &bufferSize_, nullptr));

  // Ensure the buffer is allocated with sufficient size
  EnsureSpMVBuffer(bufferSize_);

  // Get buffer size
  CUSPARSE_CHECK(cusparseSpGEMM_workEstimation(
      GetHandle(), CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_, matDescr_, other.matDescr_,
      &beta_, out.matDescr_, CUDA_C_64F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc_,
      &bufferSize_, thrust::raw_pointer_cast(spmvBuffer_.data())));

  size_t bufferSize2_ = 0;
  // Compute the number of non-zero elements
  CUSPARSE_CHECK(cusparseSpGEMM_compute(
      GetHandle(), CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_, matDescr_, other.matDescr_,
      &beta_, out.matDescr_, CUDA_C_64F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc_,
      &bufferSize2_, nullptr));

  other.EnsureSpMVBuffer(bufferSize2_);

  // Compute the number of non-zero elements
  CUSPARSE_CHECK(cusparseSpGEMM_compute(
      GetHandle(), CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_, matDescr_, other.matDescr_,
      &beta_, out.matDescr_, CUDA_C_64F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc_,
      &bufferSize2_, thrust::raw_pointer_cast(other.spmvBuffer_.data())));

  // Get the number of non-zero elements
  int64_t rows_, cols_, nnz_;
  CUSPARSE_CHECK(cusparseSpMatGetSize(out.matDescr_, &rows_, &cols_, &nnz_));
  out.nnz = static_cast<int>(nnz_);

  // Allocate memory for the result
  out.colInd.resize(out.nnz);
  out.values.resize(out.nnz);

  // Get the row pointers
  int *d_rowPtrC_ = thrust::raw_pointer_cast(out.rowPtr.data());
  int *d_colIndC_ = thrust::raw_pointer_cast(out.colInd.data());
  th_cplx *d_valuesC_ = thrust::raw_pointer_cast(out.values.data());

  // Update the matrix C with the actual data
  CUSPARSE_CHECK(cusparseCsrSetPointers(out.matDescr_, d_rowPtrC_, d_colIndC_,
                                        d_valuesC_));

  // Copy the result
  CUSPARSE_CHECK(cusparseSpGEMM_copy(
      GetHandle(), CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_, matDescr_, other.matDescr_,
      &beta_, out.matDescr_, CUDA_C_64F, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc_));

  // Clean up
  CUSPARSE_CHECK(cusparseSpGEMM_destroyDescr(spgemmDesc_));

  return out;
}

// Transpose the matrix
SparseImplGPU SparseImplGPU::Transpose() const {
  // Create output matrix
  SparseImplGPU out(DimY, DimX);

  // Allocate memory for CSR format
  out.rowPtr.resize(DimY + 1, 0);
  out.colInd.resize(nnz);
  out.values.resize(nnz);
  out.nnz = nnz;

  // Get raw pointers
  const int *d_rowPtrA = GetRowPtr();
  const int *d_colIndA = GetColIndPtr();
  const th_cplx *d_valuesA = GetValuesPtr();

  // todo: implement

  return out;
}

// Compute the Hermitian conjugate
SparseImplGPU SparseImplGPU::HermitianC() const {
  // Create output matrix
  SparseImplGPU out(DimY, DimX);

  // Allocate memory for CSR format
  out.rowPtr.resize(DimY + 1, 0);
  out.colInd.resize(nnz);
  out.values.resize(nnz);
  out.nnz = nnz;

  // Get raw pointers
  const int *d_rowPtrA = GetRowPtr();
  const int *d_colIndA = GetColIndPtr();
  const th_cplx *d_valuesA = GetValuesPtr();

  // todo: implement

  thrust::transform(
      out.values.begin(), out.values.end(), out.values.begin(),
      [] __device__(const th_cplx &x) { return thrust::conj(x); });

  return out;
}

// Multiply the matrix by a vector
VectImplGPU SparseImplGPU::VectMult(const VectImplGPU &vect) const {
  // Create output vector
  VectImplGPU out(DimX);

  // Get raw pointers
  const int *d_rowPtrA_ = GetRowPtr();
  const int *d_colIndA_ = GetColIndPtr();
  const th_cplx *d_valuesA_ = GetValuesPtr();
  const th_cplx *d_x_ = thrust::raw_pointer_cast(vect.GetDeviceData().data());

  // Create alpha and beta values
  th_cplx alpha_(1.0, 0.0);
  th_cplx beta_(0.0, 0.0);

  // Get buffer size for SpMV
  size_t bufferSize_ = 0;
  CUSPARSE_CHECK(cusparseSpMV_bufferSize(
      GetHandle(), CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_, matDescr_,
      vect.vecDescr_, &beta_, out.vecDescr_, CUDA_C_64F,
      CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize_));

  // Ensure the buffer is allocated with sufficient size
  EnsureSpMVBuffer(bufferSize_);

  // Execute SpMV using the persistent buffer
  CUSPARSE_CHECK(cusparseSpMV(
      GetHandle(), CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_, matDescr_,
      vect.vecDescr_, &beta_, out.vecDescr_, CUDA_C_64F,
      CUSPARSE_SPMV_ALG_DEFAULT, thrust::raw_pointer_cast(spmvBuffer_.data())));

  return out;
}

// Ensure SpMV buffer is allocated with sufficient size
void SparseImplGPU::EnsureSpMVBuffer(size_t requiredSize) const {
  // If the buffer is already allocated and large enough, do nothing
  if (!spmvBuffer_.empty() && spmvBufferSize_ >= requiredSize) {
    return;
  }

  // Resize the buffer to the required size
  spmvBuffer_.resize(requiredSize);
  spmvBufferSize_ = requiredSize;
}

// Free the SpMV buffer
void SparseImplGPU::FreeSpMVBuffer() const {
  // Clear the buffer
  spmvBuffer_.clear();
  spmvBufferSize_ = 0;
}

// Multiply the matrix by a vector and store result in output
void SparseImplGPU::VectMultInPlace(const VectImplGPU &vect,
                                    VectImplGPU &output) const {
  // Ensure output has correct size
  if (output.size() != DimX) {
    output = VectImplGPU(DimX);
  }

  // Get raw pointers
  const int *d_rowPtrA_ = GetRowPtr();
  const int *d_colIndA_ = GetColIndPtr();
  const th_cplx *d_valuesA_ = GetValuesPtr();
  const th_cplx *d_x_ = thrust::raw_pointer_cast(vect.GetDeviceData().data());
  const th_cplx *d_y_ = thrust::raw_pointer_cast(output.GetDeviceData().data());

  // Create alpha and beta values
  th_cplx alpha_(1.0, 0.0);
  th_cplx beta_(0.0, 0.0);

  // Get buffer size for SpMV
  size_t bufferSize_ = 0;
  CUSPARSE_CHECK(cusparseSpMV_bufferSize(
      GetHandle(), CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_, matDescr_,
      vect.vecDescr_, &beta_, output.vecDescr_, CUDA_C_64F,
      CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize_));

  // Ensure the buffer is allocated with sufficient size
  EnsureSpMVBuffer(bufferSize_);

  // Execute SpMV using the persistent buffer
  CUSPARSE_CHECK(cusparseSpMV(
      GetHandle(), CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha_, matDescr_,
      vect.vecDescr_, &beta_, output.vecDescr_, CUDA_C_64F,
      CUSPARSE_SPMV_ALG_DEFAULT, thrust::raw_pointer_cast(spmvBuffer_.data())));
}

// Overloaded addition operator
SparseImplGPU SparseImplGPU::operator+(const SparseImplGPU &A) const {
  return this->Add(A);
}

// Overloaded subtraction operator
SparseImplGPU SparseImplGPU::operator-(const SparseImplGPU &A) const {
  return this->Add(A.Scale(th_cplx(-1.0, 0.0)));
}

// Overloaded multiplication operator for scalar multiplication
SparseImplGPU SparseImplGPU::operator*(const th_cplx &alpha) const {
  return this->Scale(alpha);
}

// Overloaded multiplication operator for matrix multiplication
SparseImplGPU SparseImplGPU::operator*(const SparseImplGPU &A) const {
  return this->RightMult(A);
}

// Overloaded element-wise multiplication operator
SparseImplGPU SparseImplGPU::operator%(const SparseImplGPU &A) const {
  // Create output matrix
  SparseImplGPU out(DimX, DimY);

  // Allocate memory for CSR format
  out.rowPtr.resize(DimX + 1, 0);

  // Get raw pointers
  const int *d_rowPtrA = GetRowPtr();
  const int *d_colIndA = GetColIndPtr();
  const th_cplx *d_valuesA = GetValuesPtr();

  const int *d_rowPtrB = A.GetRowPtr();
  const int *d_colIndB = A.GetColIndPtr();
  const th_cplx *d_valuesB = A.GetValuesPtr();

  // Use cuSPARSE to compute the number of non-zero elements
  int *d_nnzTotalDevHostPtr = nullptr;
  cudaMalloc(&d_nnzTotalDevHostPtr, sizeof(int));

  // Note: We need to use the correct cuSPARSE function for complex matrices
  // For simplicity, we'll use a placeholder here
  // In a real implementation, you would use the appropriate cuSPARSE function
  // such as cusparseZcsrgemm2 for complex matrices

  // For now, we'll use a simple approach to estimate the number of non-zero
  // elements
  out.nnz = std::min(nnz, A.nnz); // This is an estimate

  // Allocate memory for the result
  out.colInd.resize(out.nnz);
  out.values.resize(out.nnz);

  // Use cuSPARSE to compute the element-wise product
  th_cplx alpha = th_cplx(1.0, 0.0);
  th_cplx beta = th_cplx(0.0, 0.0);

  // Note: We need to use the correct cuSPARSE function for complex matrices
  // For simplicity, we'll use a placeholder here
  // In a real implementation, you would use the appropriate cuSPARSE function
  // such as cusparseZcsrgemm2 for complex matrices

  // Initialize cuSPARSE
  out.InitializeCuSparse();

  return out;
}

// Get the number of non-zero elements
unsigned int SparseImplGPU::NNZ() const { return nnz; }

// Get the host data
const t_hostMat SparseImplGPU::GetHostData() const {
  t_hostMat out;
  out.resize(DimX);
  for (int i = 0; i < DimX; ++i) {
    out[i].resize(DimY, std::complex<double>(0.0, 0.0));
  }

  // Copy data to host
  std::vector<int> hostRowPtr(DimX + 1);
  std::vector<int> hostColInd(nnz);
  std::vector<th_cplx> hostValues(nnz);

  thrust::copy(rowPtr.begin(), rowPtr.end(), hostRowPtr.begin());
  thrust::copy(colInd.begin(), colInd.end(), hostColInd.begin());
  thrust::copy(values.begin(), values.end(), hostValues.begin());

  // Convert to dense format
  for (int i = 0; i < DimX; ++i) {
    for (int j = hostRowPtr[i]; j < hostRowPtr[i + 1]; ++j) {
      int col = hostColInd[j];
      out[i][col] =
          std::complex<double>(hostValues[j].real(), hostValues[j].imag());
    }
  }

  return out;
}

// Get a coefficient
std::complex<double> SparseImplGPU::CoeffRef(int i, int j) const {
  // Copy data to host
  std::vector<int> hostRowPtr(DimX + 1);
  std::vector<int> hostColInd(nnz);
  std::vector<th_cplx> hostValues(nnz);

  thrust::copy(rowPtr.begin(), rowPtr.end(), hostRowPtr.begin());
  thrust::copy(colInd.begin(), colInd.end(), hostColInd.begin());
  thrust::copy(values.begin(), values.end(), hostValues.begin());

  // Find the coefficient
  for (int k = hostRowPtr[i]; k < hostRowPtr[i + 1]; ++k) {
    if (hostColInd[k] == j) {
      return std::complex<double>(hostValues[k].real(), hostValues[k].imag());
    }
  }

  return std::complex<double>(0.0, 0.0);
}

// Overloaded multiplication operator for scalar multiplication
SparseImplGPU operator*(const th_cplx &alpha, const SparseImplGPU &rhs) {
  return rhs * alpha;
}