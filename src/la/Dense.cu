//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by Conor Stevenson on 03/04/2022.
//

#include "core/types.cuh"
#include "core/types.h"
#include "la/Dense.h"
#include "la/DenseImpl.cuh"

#include <cassert>
#include <complex>
#include <iostream>
#include <memory>
#include <vector>

// DenseImpl constructor
DenseImpl::DenseImpl() noexcept : DimX(0), DimY(0) {}

// DenseImpl destructor
DenseImpl::~DenseImpl() noexcept = default;

// DenseImpl constructor
DenseImpl::DenseImpl(int dimX, int dimY) : DimX(dimX), DimY(dimY) {
  if (DimX < 0 || DimY < 0) {
    throw std::invalid_argument("Invalid dimensions for Dense matrix.");
  }
  CPUData.resize(DimX);
  for (int i = 0; i < DimX; ++i) {
    CPUData[i].resize(DimY);
  }
}

// DenseImpl copy constructor
DenseImpl::DenseImpl(const DenseImpl &other) noexcept
    : DimX(other.DimX), DimY(other.DimY), CPUData(other.CPUData) {}

// DenseImpl move constructor
DenseImpl::DenseImpl(DenseImpl &&other) noexcept
    : DimX(other.DimX), DimY(other.DimY), CPUData(std::move(other.CPUData)) {}

// DenseImpl constructor
DenseImpl::DenseImpl(t_hostMat &in) noexcept
    : DimX(in.size()), DimY(in[0].size()) {
  CPUData = in;
}

// Copy assignment operator
DenseImpl &DenseImpl::operator=(const DenseImpl &other) noexcept {
  if (this == &other) {
    return *this;
  }

  DimX = other.DimX;
  DimY = other.DimY;
  CPUData = other.CPUData;

  return *this;
}

DenseImpl DenseImpl::Add(const DenseImpl &A) const {
  // if dimensions don't match, throw
  if (DimX != A.DimX || DimY != A.DimY) {
    throw std::invalid_argument("Dimensions do not match for Dense matrix.");
  }

  DenseImpl out(DimX, DimY);
  for (int i = 0; i < DimX; ++i) {
    for (int j = 0; j < DimY; ++j) {
      out.CPUData[i][j] = CPUData[i][j] + A.CPUData[i][j];
    }
  }
  return out;
}

DenseImpl DenseImpl::RightMult(const DenseImpl &A) const {
  // if dimensions don't match, throw
  if (DimY != A.DimX) {
    throw std::invalid_argument("Dimensions do not match for Dense matrix.");
  }

  DenseImpl out(DimX, A.DimY);
  for (int i = 0; i < DimX; ++i) {
    for (int j = 0; j < DimY; ++j) {
      t_cplx sum = 0;
      for (int k = 0; k < DimY; ++k) {
        sum += CPUData[i][k] * A.CPUData[k][j];
      }
      out.CPUData[i][j] = sum;
    }
  }

  return out;
}

DenseImpl DenseImpl::Scale(t_cplx alpha) const noexcept {
  DenseImpl out(DimX, DimY);

  for (int i = 0; i < out.CPUData.size(); ++i) {
    for (int j = 0; j < out.CPUData[0].size(); ++j) {
      out.CPUData[i][j] = alpha * CPUData[i][j];
    }
  }

  return out;
}

DenseImpl DenseImpl::Transpose() const noexcept {
  DenseImpl out(DimY, DimX);

  for (int i = 0; i < DimY; ++i) {
    for (int j = 0; j < DimX; ++j) {
      out.CPUData[i][j] = CPUData[j][i];
    }
  }

  return out;
}

DenseImpl DenseImpl::HermitianC() const noexcept {
  DenseImpl out(DimY, DimX);

  for (int i = 0; i < DimY; ++i) {
    for (int j = 0; j < DimX; ++j) {
      out.CPUData[i][j] = conj(CPUData[j][i]);
    }
  }

  return out;
}

t_hostVect DenseImpl::FlattenedData() const noexcept {
  t_hostVect out;
  out.resize(DimX * DimY);

  for (int i = 0; i < DimX; i++) {
    for (int j = 0; j < DimY; j++) {
      out[j + i * DimY] = CPUData[i][j];
    }
  }

  return out;
}

void DenseImpl::Print(unsigned int kind, unsigned int prec) const noexcept {
  std::string s;
  std::stringstream stream;
  stream.setf(std::ios::fixed);
  stream.precision(prec);

  stream << " Matrix [" << DimX << " x " << DimY << "]:" << std::endl;
  for (const auto &X : CPUData) {
    stream << "   ";
    for (auto Y : X) {
      std::string spaceCharRe = !std::signbit(Y.real()) ? " " : "";
      std::string spaceCharIm = !std::signbit(Y.imag()) ? " " : "";
      std::string spaceCharAbs = !std::signbit(Y.imag()) ? " + " : "-";

      switch (kind) {
      case 0: // re + im
        stream << spaceCharRe << Y.real() << spaceCharAbs << abs(Y.imag())
               << "i  ";
        break;
      case 1: // re
        stream << spaceCharRe << Y.real() << " ";
        break;
      case 2: // im
        stream << spaceCharIm << Y.imag() << "i  ";
        break;
      case 3: // abs
        stream << " " << abs(Y);
        break;
      default:
        stream << "[e]";
      }
    }
    stream << std::endl;
  }

  s = stream.str();

  std::cout << s << std::endl;
}

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
// Definition of Dense
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

// Default constructor
Dense::Dense() noexcept : pImpl(std::make_unique<DenseImpl>()) {}

// Dense constructor
Dense::Dense(int dimX, int dimY)
    : pImpl(std::make_unique<DenseImpl>(dimX, dimY)) {
  // throw if dimensions are invalid
  if (dimX < 0 || dimY < 0) {
    throw std::invalid_argument("Invalid dimensions for Dense matrix.");
  }
}

// Dense constructor
Dense::Dense(t_hostMat &in) noexcept : pImpl(std::make_unique<DenseImpl>(in)) {
  pImpl->CPUData = in;
}

// Destructor
Dense::~Dense() noexcept = default;

/**
 * @brief Dense copy constructor
 */
Dense::Dense(const Dense &other) noexcept
    : pImpl(std::make_unique<DenseImpl>(*other.pImpl)) {}

/*
 * @brief Dense move constructor
 */
Dense::Dense(Dense &&other) noexcept : pImpl(std::move(other.pImpl)) {}

/**
 * @brief Dense non-empty constructor
 */
Dense::Dense(std::unique_ptr<DenseImpl> impl) noexcept
    : pImpl(std::move(impl)) {}

// Copy assignment operator
Dense &Dense::operator=(const Dense &other) noexcept {
  if (this == &other) {
    return *this;
  }

  pImpl = std::make_unique<DenseImpl>(*other.pImpl);

  return *this;
}

/**
 * @brief Get the number of rows in the Dense matrix.
 *
 * @return int Number of rows.
 */

int Dense::DimX() const { return pImpl->DimX; }

/**
 * @brief Get the number of columns in the Dense matrix.
 *
 * @return int Number of columns.
 */
int Dense::DimY() const { return pImpl->DimY; }

const t_hostMat &Dense::GetHostData() const { return pImpl->CPUData; }

/**
 * @brief Get the data at a specific position in the Dense matrix.
 *
 * @param i Row index.
 * @param j Column index.
 * @return std::complex<double> Element at the specified position.
 */
std::complex<double> &Dense::GetData(int i, int j) const {
  // if out of bounds, throw
  if (i < 0 || i >= pImpl->DimX || j < 0 || j >= pImpl->DimY) {
    throw std::out_of_range("Index out of bounds for Dense matrix.");
  }
  return pImpl->CPUData[i][j];
}

// at access, which will never throw
std::complex<double> Dense::at(int i, int j) const noexcept {
  // make sure in range
  if (i < 0 || i >= pImpl->DimX || j < 0 || j >= pImpl->DimY) {
    return {-1337., -1337.};
  }
  return pImpl->CPUData[i][j];
}

/**
 * @brief Get a reference to the data at a specific position in the Dense
 * matrix.
 *
 * @param i Row index.
 * @param j Column index.
 * @return std::complex<double>& Reference to the element at the specified
 * position.
 */
std::complex<double> &Dense::GetDataRef(int i, int j) const {
  return pImpl->CPUData[i][j];
}

/**
 * @brief Overloaded subscript operator to access matrix elements.
 *
 * @param col Column index.
 * @param row Row index.
 * @return std::complex<double> Element at the specified position.
 */
std::complex<double> Dense::operator[](int col, int row) {
  if (pImpl != nullptr) {
    return pImpl->CPUData[col][row];
  }
  return -1337.;
}

/**
 * @brief Move assignment operator.
 *
 * @param other Another Dense object to move from.
 * @return Dense& Reference to the current object.
 */
Dense &Dense::operator=(Dense &&other) noexcept {
  if (this == &other) {
    return *this;
  }

  pImpl = std::move(other.pImpl);

  return *this;
}

/**
 * @brief Overloaded addition operator for Dense matrices.
 *
 * @param A Another Dense object to add.
 * @return Dense Result of the addition.
 */
Dense Dense::operator+(const Dense &A) const {
  return Dense(std::make_unique<DenseImpl>(pImpl->Add(*A.pImpl)));
}

/**
 * @brief Overloaded multiplication operator for Dense matrices.
 *
 * @param A Another Dense object to multiply.
 * @return Dense Result of the multiplication.
 */
Dense Dense::operator*(const Dense &A) const {
  return Dense(std::make_unique<DenseImpl>(pImpl->RightMult(*A.pImpl)));
}

/**
 * @brief Overloaded subtraction operator for Dense matrices.
 *
 * @param A Another Dense object to subtract.
 * @return Dense Result of the subtraction.
 */
Dense Dense::operator-(const Dense &A) const {
  return this->operator+(A.operator*(t_cplx(-1)));
}

/**
 * @brief Overloaded multiplication operator for scalar multiplication.
 *
 * @param alpha Scalar value to multiply.
 * @return Dense Result of the scalar multiplication.
 */
Dense Dense::operator*(const t_cplx &alpha) const noexcept {
  return Dense(std::make_unique<DenseImpl>(pImpl->Scale(alpha)));
}

/**
 * @brief Overloaded multiplication operator for scalar multiplication.
 *
 * @param alpha Scalar value to multiply.
 * @param rhs Dense object to multiply.
 * @return Dense Result of the scalar multiplication.
 */
Dense operator*(const t_cplx &alpha, const Dense &rhs) noexcept {
  return rhs * alpha;
}

/**
 * @brief Transpose the Dense matrix.
 *
 * @return Dense Transposed matrix.
 */
Dense Dense::Transpose() const noexcept {
  return Dense(std::make_unique<DenseImpl>(pImpl->Transpose()));
}

/**
 * @brief Compute the Hermitian conjugate of the Dense matrix.
 *
 * @return Dense Hermitian conjugate matrix.
 */
Dense Dense::HermitianC() const noexcept {
  return Dense(std::make_unique<DenseImpl>(pImpl->HermitianC()));
}

/**
 * @brief Flatten the Dense matrix data into a vector.
 *
 * @return t_hostVect Flattened data.
 */
t_hostVect Dense::FlattenedData() const noexcept {
  return pImpl->FlattenedData();
}

/**
 * @brief Print the Dense matrix.
 *
 * @param kind Type of data to print (real, imaginary, etc.).
 * @param prec Precision of the printed data.
 */
void Dense::Print(unsigned int kind, unsigned int prec) const {
  pImpl->Print(kind, prec);
}
