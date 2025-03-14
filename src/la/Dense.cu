//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by Conor Stevenson on 03/04/2022.
//

#include "la/Dense.h"
#include <cassert>
#include <complex>
#include <iostream>
#include <memory>
#include <vector>

using t_cplx = std::complex<double>;
using t_hostVect = std::vector<std::complex<double>>;
using t_hostVectInt = std::vector<int>;

class Dense::DenseImpl {
public:
  int DimX;                        ///< Number of rows
  int DimY;                        ///< Number of columns
  std::vector<t_hostVect> CPUData; ///< Matrix data stored in a 2D vector

  /**
   * @brief Default constructor for DenseImpl.
   */
  DenseImpl() = default;

  /**
   * @brief Destructor for DenseImpl.
   */
  ~DenseImpl() = default;

  /**
   * @brief Constructor to initialize DenseImpl matrix with given dimensions.
   *
   * @param dimX Number of rows.
   * @param dimY Number of columns.
   */
  DenseImpl(int dimX, int dimY) : DimX(dimX), DimY(dimY) {
    CPUData.resize(dimX, std::vector<std::complex<double>>(dimY));
  }

  /**
   * @brief Copy constructor for DenseImpl.
   *
   * @param other Another DenseImpl object to copy from.
   */
  DenseImpl(const DenseImpl &other)
      : DimX(other.DimX), DimY(other.DimY), CPUData(other.CPUData) {}

  /**
   * @brief Move constructor for DenseImpl.
   *
   * @param other Another DenseImpl object to move from.
   */
  DenseImpl(DenseImpl &&other) noexcept
      : DimX(other.DimX), DimY(other.DimY), CPUData(std::move(other.CPUData)) {}

  /**
   * @brief Constructor to initialize DenseImpl matrix with given data.
   *
   * @param in Input matrix data.
   */
  DenseImpl(t_hostMat &in) : CPUData(in), DimX(in.size()), DimY(in[0].size()) {}

  /**
   * @brief Copy assignment operator for DenseImpl.
    *
    * @param other Another DenseImpl object to copy from.
    * @return DenseImpl& Reference to the current object.
    */
  DenseImpl &operator=(const DenseImpl &other) {
    if (this == &other) {
      return *this;
    }

    DimX = other.DimX;
    DimY = other.DimY;
    CPUData = other.CPUData;

    return *this;
  }

  /**
   * @brief Adds two DenseImpl matrices.
   *
   * @param A Another DenseImpl object to add.
   * @return DenseImpl Result of the addition.
   */
  DenseImpl Add(const DenseImpl &A) const;

  /**
   * @brief Multiplies two DenseImpl matrices.
   *
   * @param A Another DenseImpl object to multiply.
   * @return DenseImpl Result of the multiplication.
   */
  DenseImpl RightMult(const DenseImpl &A) const;

  /**
   * @brief Scales the DenseImpl matrix by a scalar value.
   *
   * @param alpha Scalar value to multiply.
   * @return DenseImpl Result of the scalar multiplication.
   */
  DenseImpl Scale(t_cplx alpha) const;

  /**
   * @brief Transposes the DenseImpl matrix.
   *
   * @return DenseImpl Transposed matrix.
   */
  DenseImpl Transpose() const;

  /**
   * @brief Computes the Hermitian conjugate of the DenseImpl matrix.
   *
   * @return DenseImpl Hermitian conjugate matrix.
   */
  DenseImpl HermitianC() const;

  /**
   * @brief Flattens the DenseImpl matrix data into a vector.
   *
   * @return t_hostVect Flattened data.
   */
  t_hostVect FlattenedData() const;

  /**
   * @brief Flattens the DenseImpl matrix data into a vector of integers.
   *
   * @return t_hostVectInt Flattened data as integers.
   */
  t_hostVectInt FlattenedDataInt() const;

  /**
   * @brief Prints the DenseImpl matrix.
   *
   * @param kind Type of data to print (real, imaginary, etc.).
   * @param prec Precision of the printed data.
   */
  void Print(unsigned int kind, unsigned int prec) const;
};

Dense::DenseImpl Dense::DenseImpl::Add(const Dense::DenseImpl &A) const {
  std::cout << "DenseImpl Add" << std::endl;
  assert(DimX == A.DimX && DimY == A.DimY);

  Dense::DenseImpl out(DimX, DimY);
  for (int i = 0; i < DimX; ++i) {
    for (int j = 0; j < DimY; ++j) {
      out.CPUData[i][j] = CPUData[i][j] + A.CPUData[i][j];
    }
  }
  std::cout << "DenseImpl Add returning..." << std::endl;
  return out;
}

Dense::DenseImpl Dense::DenseImpl::RightMult(const Dense::DenseImpl &A) const {
  assert(DimY == A.DimX);

  Dense::DenseImpl out(DimX, A.DimY);
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

Dense::DenseImpl Dense::DenseImpl::Scale(t_cplx alpha) const {
  Dense::DenseImpl out(DimX, DimY);

  for (int i = 0; i < out.CPUData.size(); ++i) {
    for (int j = 0; j < out.CPUData[0].size(); ++j) {
      out.CPUData[i][j] = alpha * CPUData[i][j];
    }
  }

  return out;
}

Dense::DenseImpl Dense::DenseImpl::Transpose() const {
  Dense::DenseImpl out(DimY, DimX);

  for (int i = 0; i < DimY; ++i) {
    for (int j = 0; j < DimX; ++j) {
      out.CPUData[i][j] = CPUData[j][i];
    }
  }

  return out;
}

Dense::DenseImpl Dense::DenseImpl::HermitianC() const {
  Dense::DenseImpl out(DimY, DimX);

  for (int i = 0; i < DimY; ++i) {
    for (int j = 0; j < DimX; ++j) {
      out.CPUData[i][j] = conj(CPUData[j][i]);
    }
  }

  return out;
}

t_hostVect Dense::DenseImpl::FlattenedData() const {
  t_hostVect out;
  out.resize(DimX * DimY);

  for (int i = 0; i < DimX; i++) {
    for (int j = 0; j < DimY; j++) {
      out[j + i * DimY] = CPUData[i][j];
    }
  }

  return out;
}

t_hostVectInt Dense::DenseImpl::FlattenedDataInt() const {
  std::vector<int> out;

  out.resize(DimX * DimY);

  for (int i = 0; i < DimX; i++) {
    for (int j = 0; j < DimY; j++) {
      out[j + i * DimY] = round(abs(CPUData[i][j]));
    }
  }

  return out;
}

void Dense::DenseImpl::Print(unsigned int kind, unsigned int prec) const {
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

// Destructor
Dense::~Dense() = default;

/**
 * @brief Dense copy constructor
 */
Dense::Dense(const Dense &other)
    : pImpl(std::make_unique<DenseImpl>(*other.pImpl)) {}

/*
 * @brief Dense move constructor
 */
Dense::Dense(Dense &&other) noexcept : pImpl(std::move(other.pImpl)) {}

/*
 * @brief Dense non-empty constructor
 */
Dense::Dense(t_hostMat &in) : pImpl(std::make_unique<DenseImpl>(in)) {}

/**
 * @brief Dense non-empty constructor
 */
Dense::Dense(std::unique_ptr<DenseImpl> impl) : pImpl(std::move(impl)) {}

// Copy assignment operator
Dense &Dense::operator=(const Dense &other) {
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

/**
 * @brief Get the data at a specific position in the Dense matrix.
 *
 * @param i Row index.
 * @param j Column index.
 * @return std::complex<double> Element at the specified position.
 */
std::complex<double> &Dense::GetData(int i, int j) const {
  return pImpl->CPUData[i][j];
}

/**
 * @brief Get a reference to the data at a specific position in the Dense matrix.
 *
 * @param i Row index.
 * @param j Column index.
 * @return std::complex<double>& Reference to the element at the specified position.
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
Dense Dense::operator*(const t_cplx &alpha) const {
  return Dense(std::make_unique<DenseImpl>(pImpl->Scale(alpha)));
}

/**
 * @brief Overloaded multiplication operator for scalar multiplication.
 *
 * @param alpha Scalar value to multiply.
 * @param rhs Dense object to multiply.
 * @return Dense Result of the scalar multiplication.
 */
Dense operator*(const t_cplx &alpha, const Dense &rhs) { return rhs * alpha; }

/**
 * @brief Transpose the Dense matrix.
 *
 * @return Dense Transposed matrix.
 */
Dense Dense::Transpose() const {
  return Dense(std::make_unique<DenseImpl>(pImpl->Transpose()));
}

/**
 * @brief Compute the Hermitian conjugate of the Dense matrix.
 *
 * @return Dense Hermitian conjugate matrix.
 */
Dense Dense::HermitianC() const {
  return Dense(std::make_unique<DenseImpl>(pImpl->HermitianC()));
}

/**
 * @brief Flatten the Dense matrix data into a vector.
 *
 * @return t_hostVect Flattened data.
 */
t_hostVect Dense::FlattenedData() const { return pImpl->FlattenedData(); }

/**
 * @brief Flatten the Dense matrix data into a vector of integers.
 *
 * @return t_hostVectInt Flattened data as integers.
 */
t_hostVectInt Dense::FlattenedDataInt() const {
  return pImpl->FlattenedDataInt();
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
