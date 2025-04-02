//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by Conor Stevenson on 15/3/2025.
//

#include "core/types.cuh"
#include "core/types.h"
#include "la/SparseELL.h"
#include "la/SparseELLImpl.cuh"

#include <cassert>
#include <iostream>
#include <memory>
#include <vector>

// SparseELLImpl constructor
SparseELLImpl::SparseELLImpl() noexcept : DimX(0), DimY(0), MaxNnzPerRow(0) {}

// SparseELLImpl destructor
SparseELLImpl::~SparseELLImpl() noexcept = default;

// SparseELLImpl constructor
SparseELLImpl::SparseELLImpl(int dimX, int dimY, int maxNnzPerRow)
    : DimX(dimX), DimY(dimY), MaxNnzPerRow(maxNnzPerRow) {
  if (DimX < 0 || DimY < 0 || MaxNnzPerRow < 0) {
    throw std::invalid_argument("Invalid dimensions for SparseELL matrix.");
  }
  CPUData.resize(DimX);
  CPUCol.resize(DimX);
  for (int i = 0; i < DimX; ++i) {
    CPUData[i].resize(MaxNnzPerRow);
    CPUCol[i].resize(MaxNnzPerRow);
  }
}

// SparseELLImpl copy constructor
SparseELLImpl::SparseELLImpl(const SparseELLImpl &other) noexcept
    : DimX(other.DimX), DimY(other.DimY), MaxNnzPerRow(other.MaxNnzPerRow),
      CPUData(other.CPUData), CPUCol(other.CPUCol) {}

// SparseELLImpl move constructor
SparseELLImpl::SparseELLImpl(SparseELLImpl &&other) noexcept
    : DimX(other.DimX), DimY(other.DimY), MaxNnzPerRow(other.MaxNnzPerRow),
      CPUData(std::move(other.CPUData)), CPUCol(std::move(other.CPUCol)) {}

// Copy assignment operator
SparseELLImpl &SparseELLImpl::operator=(const SparseELLImpl &other) noexcept {
  if (this == &other) {
    return *this;
  }

  DimX = other.DimX;
  DimY = other.DimY;
  MaxNnzPerRow = other.MaxNnzPerRow;
  CPUData = other.CPUData;
  CPUCol = other.CPUCol;

  return *this;
}

SparseELLImpl SparseELLImpl::Add(const SparseELLImpl &A) const {
  if (DimX != A.DimX || DimY != A.DimY) {
    throw std::invalid_argument(
        "Dimensions do not match for SparseELL matrix.");
  }

  SparseELLImpl out(DimX, DimY, std::max(MaxNnzPerRow, A.MaxNnzPerRow));
  // TODO: Implement sparse matrix addition
  return out;
}

SparseELLImpl SparseELLImpl::RightMult(const SparseELLImpl &A) const {
  if (DimY != A.DimX) {
    throw std::invalid_argument(
        "Dimensions do not match for SparseELL matrix.");
  }

  SparseELLImpl out(DimX, A.DimY, std::max(MaxNnzPerRow, A.MaxNnzPerRow));
  // TODO: Implement sparse matrix multiplication
  return out;
}

SparseELLImpl SparseELLImpl::Scale(t_cplx alpha) const noexcept {
  SparseELLImpl out(DimX, DimY, MaxNnzPerRow);
  for (int i = 0; i < DimX; ++i) {
    for (int j = 0; j < MaxNnzPerRow; ++j) {
      out.CPUData[i][j] = alpha * CPUData[i][j];
      out.CPUCol[i][j] = CPUCol[i][j];
    }
  }
  return out;
}

SparseELLImpl SparseELLImpl::Transpose() const noexcept {
  SparseELLImpl out(DimY, DimX, MaxNnzPerRow);
  // TODO: Implement sparse matrix transpose
  return out;
}

SparseELLImpl SparseELLImpl::HermitianC() const noexcept {
  SparseELLImpl out(DimY, DimX, MaxNnzPerRow);
  // TODO: Implement sparse matrix Hermitian conjugate
  return out;
}

void SparseELLImpl::Print(unsigned int kind, unsigned int prec) const noexcept {
  std::string s;
  std::stringstream stream;
  stream.setf(std::ios::fixed);
  stream.precision(prec);

  stream << " SparseELL Matrix [" << DimX << " x " << DimY
         << "] (MaxNnzPerRow: " << MaxNnzPerRow << "):" << std::endl;
  for (int i = 0; i < DimX; ++i) {
    stream << "   Row " << i << ": ";
    for (int j = 0; j < MaxNnzPerRow; ++j) {
      if (std::abs(CPUCol[i][j]) >= 0) { // Only print non-zero elements
        std::string spaceCharRe =
            !std::signbit(CPUData[i][j].real()) ? " " : "";
        std::string spaceCharIm =
            !std::signbit(CPUData[i][j].imag()) ? " " : "";
        std::string spaceCharAbs =
            !std::signbit(CPUData[i][j].imag()) ? " + " : "-";

        stream << "(" << CPUCol[i][j] << ") ";
        switch (kind) {
        case 0: // re + im
          stream << spaceCharRe << CPUData[i][j].real() << spaceCharAbs
                 << abs(CPUData[i][j].imag()) << "i  ";
          break;
        case 1: // re
          stream << spaceCharRe << CPUData[i][j].real() << " ";
          break;
        case 2: // im
          stream << spaceCharIm << CPUData[i][j].imag() << "i  ";
          break;
        case 3: // abs
          stream << " " << abs(CPUData[i][j]);
          break;
        default:
          stream << "[e]";
        }
      }
    }
    stream << std::endl;
  }

  s = stream.str();
  std::cout << s << std::endl;
}

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
// Definition of SparseELL
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

// Default constructor
SparseELL::SparseELL() noexcept : pImpl(std::make_unique<SparseELLImpl>()) {}
SparseELL::SparseELL(std::unique_ptr<SparseELLImpl> pImpl)
    : pImpl(std::move(pImpl)) {}

// SparseELL constructor
SparseELL::SparseELL(int dimX, int dimY, int maxNnzPerRow)
    : pImpl(std::make_unique<SparseELLImpl>(dimX, dimY, maxNnzPerRow)) {
  if (dimX < 0 || dimY < 0 || maxNnzPerRow < 0) {
    throw std::invalid_argument("Invalid dimensions for SparseELL matrix.");
  }
}

// Destructor
SparseELL::~SparseELL() noexcept = default;

// Copy constructor
SparseELL::SparseELL(const SparseELL &other) noexcept
    : pImpl(std::make_unique<SparseELLImpl>(*other.pImpl)) {}

// Move constructor
SparseELL::SparseELL(SparseELL &&other) noexcept
    : pImpl(std::move(other.pImpl)) {}

// Copy assignment operator
SparseELL &SparseELL::operator=(const SparseELL &other) noexcept {
  if (this == &other) {
    return *this;
  }

  pImpl = std::make_unique<SparseELLImpl>(*other.pImpl);
  return *this;
}

SparseELL SparseELL::Add(const SparseELL &A) const {
  return SparseELL(std::make_unique<SparseELLImpl>(pImpl->Add(*A.pImpl)));
}

SparseELL SparseELL::RightMult(const SparseELL &A) const {
  return SparseELL(std::make_unique<SparseELLImpl>(pImpl->RightMult(*A.pImpl)));
}

SparseELL SparseELL::Scale(t_cplx alpha) const noexcept {
  return SparseELL(std::make_unique<SparseELLImpl>(pImpl->Scale(alpha)));
}

SparseELL SparseELL::Transpose() const noexcept {
  return SparseELL(std::make_unique<SparseELLImpl>(pImpl->Transpose()));
}

SparseELL SparseELL::HermitianC() const noexcept {
  return SparseELL(std::make_unique<SparseELLImpl>(pImpl->HermitianC()));
}

void SparseELL::Print(unsigned int kind, unsigned int prec) const noexcept {
  pImpl->Print(kind, prec);
}

int SparseELL::DimX() const { return pImpl->DimX; }

int SparseELL::DimY() const { return pImpl->DimY; }

int SparseELL::MaxNnzPerRow() const { return pImpl->MaxNnzPerRow; }