//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by conor on 11/04/2022.
//

#include "core/types.cuh"
#include "la/Sparse.h"
#include "la/SparseELL.h"
#include "la/SparseELLImpl.cuh"

// SparseELL ToSparseELL(const Sparse &A) {
//   auto rows = SparseRowsCOO(A);

//   // determine length of longest row
//   unsigned int highestCount = 0;
//   for (auto row : rows) {
//     unsigned int len = row.RowData.size();
//     if (len > highestCount) {
//       highestCount = len;
//     }
//   }

//   SparseELL out(A.DimX, A.DimY, highestCount);
//   for (auto row : rows) {
//     for (int i = 0; i < row.RowData.size(); i++) {
//       // Change to denseimpls
//       // out.Values.GetDataRef(row.Index, i) = row.RowData[i].Val;
//       // out.Indices.GetDataRef(row.Index, i) = row.RowData[i].Coords[1];
//     }

//     // padding with '-1'
//     for (int i = row.RowData.size(); i < highestCount; i++) {
//       // out.Values.GetDataRef(row.Index, i) = 0;
//       // out.Indices.GetDataRef(row.Index, i) = -1;
//     }
//   }

//   return out;
// }

// TODO: Figure out a way to let underlying pimpls communicate with each other
//  in order to implement this function.

// Vect SparseELL::VectMult(const Vect::Impl &vect) const {
//   Vect out(vect.Data.size());

//   for (int row = 0; row < vect.Data.size(); ++row) {
//     out[row] = 0;
//     for (int i = 0; i < EntriesPerRow; i++) {
//       int col = floor(Indices.GetData(row, i).real());
//       th_cplx val = Values.GetData(row, i);

//       if (col > -1)
//         out.Data[row] += val * vect.Data[col];
//     }
//   }
//   return out;
// }
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

const std::vector<t_hostVect> &SparseELL::GetHostData() const {
  return pImpl->CPUData;
}
