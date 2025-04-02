//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by Conor Stevenson on 03/04/2022.
//

#include "la/Dense.h"
#include "la/Sparse.h"
#include "la/SparseImpl.cuh"
#include "la/Super.cuh"
#include <vector>
//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by Conor Stevenson on 03/04/2022.
//

#include "la/Dense.h"
#include "la/Sparse.h"
#include "la/SparseImpl.cuh"
#include "la/Vect.h"
#include <memory>

Sparse::Sparse(int dimX, int dimY)
    : pImpl(std::make_unique<SparseImpl>(dimX, dimY)) {}

Sparse::~Sparse() noexcept = default;

Sparse::Sparse(const Sparse &other) noexcept
    : pImpl(std::make_unique<SparseImpl>(*other.pImpl)) {}

Sparse::Sparse(Sparse &&other) noexcept : pImpl(std::move(other.pImpl)) {}

Sparse &Sparse::operator=(const Sparse &other) noexcept {
  if (this != &other) {
    pImpl = std::make_unique<SparseImpl>(*other.pImpl);
  }
  return *this;
}

Sparse Sparse::Scale(const t_cplx &alpha) const {
  return Sparse(std::make_unique<SparseImpl>(pImpl->Scale(alpha)));
}

Sparse Sparse::Add(const Sparse &B) const {
  return Sparse(std::make_unique<SparseImpl>(pImpl->Add(*B.pImpl)));
}

Sparse Sparse::RightMult(const Sparse &A) const {
  return Sparse(std::make_unique<SparseImpl>(pImpl->RightMult(*A.pImpl)));
}

Sparse Sparse::Transpose() const {
  return Sparse(std::make_unique<SparseImpl>(pImpl->Transpose()));
}

Sparse Sparse::HermitianC() const {
  return Sparse(std::make_unique<SparseImpl>(pImpl->HermitianC()));
}

Dense Sparse::ToDense() { return pImpl->ToDense(); }

void Sparse::SortByRow() { pImpl->SortByRow(); }

void Sparse::Trim() { pImpl->Trim(); }

Vect Sparse::VectMult(const Vect &vect) const {
  return Vect(std::make_unique<VectImpl>(pImpl->VectMult(*vect.pImpl)));
}

Sparse Sparse::operator+(const Sparse &A) const {
  return Sparse(std::make_unique<SparseImpl>(pImpl->operator+(*A.pImpl)));
}

Sparse Sparse::operator-(const Sparse &A) const {
  return Sparse(std::make_unique<SparseImpl>(pImpl->operator-(*A.pImpl)));
}

Sparse Sparse::operator*(const t_cplx &alpha) const {
  return Sparse(std::make_unique<SparseImpl>(pImpl->operator*(alpha)));
}

Sparse Sparse::operator*(const Sparse &A) const {
  return Sparse(std::make_unique<SparseImpl>(pImpl->operator*(*A.pImpl)));
}

Sparse Sparse::operator%(const Sparse &A) const {
  return Sparse(std::make_unique<SparseImpl>(pImpl->operator%(*A.pImpl)));
}

void Sparse::Print() const { pImpl->Print(); }

void Sparse::PrintRe() const { pImpl->PrintRe(); }

void Sparse::PrintIm() const { pImpl->PrintIm(); }

void Sparse::PrintAbs() const { pImpl->PrintAbs(); }

unsigned int Sparse::NNZ() const { return pImpl->NNZ(); }

const std::vector<COOTuple> &Sparse::GetHostData() const {
  return pImpl->GetHostData();
}

std::complex<double> &Sparse::CoeffRef(int i, int j) {
  return pImpl->CoeffRef(i, j);
}

Sparse::Sparse(std::unique_ptr<SparseImpl> pImpl) : pImpl(std::move(pImpl)) {}

Sparse operator*(const std::complex<double> &alpha, const Sparse &rhs) {
  return rhs * alpha;
}
