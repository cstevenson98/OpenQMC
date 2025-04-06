//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by Conor Stevenson on 03/04/2022.
//
#include <complex>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

#include "core/types.cuh"
#include "la/Vect.h"
#include "la/VectImpl.cuh"

VectImpl VectImpl::Conj() const {
  VectImpl out(Data.size());
  out.Data = Data.conjugate();
  return out;
}

VectImpl VectImpl::Add(const VectImpl &A) const {
  VectImpl out(Data.size());
  out.Data = Data + A.Data;
  return out;
}

VectImpl VectImpl::Subtract(const VectImpl &A) const {
  VectImpl out(Data.size());
  out.Data = Data - A.Data;
  return out;
}

VectImpl VectImpl::Scale(const th_cplx &alpha) const {
  VectImpl out(Data.size());
  out.Data = Data * alpha;
  return out;
}

std::complex<double> VectImpl::Dot(const VectImpl &A) const {
  return Data.dot(A.Data);
}

double VectImpl::Norm() const { return Data.norm(); }

VectImpl VectImpl::operator+(const VectImpl &A) const { return this->Add(A); }

VectImpl VectImpl::operator-(const VectImpl &A) const {
  return this->Subtract(A);
}

VectImpl VectImpl::operator*(const th_cplx &alpha) const {
  return this->Scale(alpha);
}

std::complex<double> VectImpl::operator[](unsigned int i) const {
  return this->Data(i);
}

void VectImpl::Print(unsigned int kind) const {
  std::string s;
  std::stringstream stream;
  stream.setf(std::ios::fixed);
  stream.precision(2);

  stream << " Vector [" << Data.size() << " x " << 1 << "]:" << std::endl;
  for (int i = 0; i < Data.size(); ++i) {
    stream << "   ";
    std::string spaceCharRe = !std::signbit(Data(i).real()) ? " " : "";
    std::string spaceCharIm = !std::signbit(Data(i).imag()) ? " " : "";
    std::string spaceCharAbs = !std::signbit(Data(i).imag()) ? " + " : " - ";

    switch (kind) {
    case 0: // re + im
      stream << spaceCharRe << Data(i).real() << spaceCharAbs
             << abs(Data(i).imag()) << "i  ";
      break;
    case 1: // re
      stream << spaceCharRe << Data(i).real() << " ";
      break;
    case 2: // im
      stream << spaceCharIm << Data(i).imag() << "i  ";
      break;
    case 3: // abs
      stream << " " << abs(Data(i));
      break;
    default:
      stream << "[e]";
    }
    stream << std::endl;
  }

  s = stream.str();

  std::cout << s << std::endl;
}

void VectImpl::PrintRe() const { this->Print(1); }

void VectImpl::PrintIm() const { this->Print(2); }

void VectImpl::PrintAbs() const { this->Print(3); }

int VectImpl::size() const { return Data.size(); }

std::vector<std::complex<double>> VectImpl::GetData() const {
  // convert Eigen vector to std::vector
  std::vector<std::complex<double>> out;
  out.resize(Data.size());
  for (int i = 0; i < Data.size(); ++i) {
    out[i] = Data(i);
  }
  return out;
}

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

/////////////////////////////////////////////////////////
/// Vect
/////////////////////////////////////////////////////////

Vect::Vect(unsigned int size)
    : pImpl(std::make_unique<VectImpl>(VectImpl(size))) {}

Vect::Vect(t_hostVect &in) : pImpl(std::make_unique<VectImpl>(in)) {}

Vect::Vect(std::unique_ptr<VectImpl> pImpl) : pImpl(std::move(pImpl)) {}

Vect Vect::Conj() const {
  Vect out;
  out.pImpl = std::make_unique<VectImpl>(pImpl->Conj());
  return out;
}

Vect Vect::operator+(const Vect &A) const {
  return Vect(std::make_unique<VectImpl>(pImpl->Add(*A.pImpl)));
}

Vect Vect::operator-(const Vect &A) const {
  return Vect(std::make_unique<VectImpl>(pImpl->Subtract(*A.pImpl)));
}

Vect Vect::operator*(const t_cplx &alpha) const {
  return Vect(std::make_unique<VectImpl>(pImpl->Scale(alpha)));
}

Vect operator*(const t_cplx &alpha, const Vect &rhs) { return rhs * alpha; }

std::complex<double> Vect::operator[](unsigned int i) const {
  return pImpl->operator[](i);
}

int Vect::size() const { return pImpl->size(); }

std::complex<double> Vect::Dot(const Vect &A) const {
  return pImpl->Dot(*A.pImpl);
}

double Vect::Norm() const { return pImpl->Norm(); }

void Vect::Print(unsigned int kind) const { pImpl->Print(kind); }

void Vect::PrintRe() const { pImpl->PrintRe(); }

void Vect::PrintIm() const { pImpl->PrintIm(); }

void Vect::PrintAbs() const { pImpl->PrintAbs(); }

std::vector<std::complex<double>> Vect::GetData() const {
  return pImpl->GetData();
}

const t_hostVect Vect::GetHostData() const { return pImpl->GetHostData(); }

// Destructor
Vect::~Vect() = default;

// Copy constructor
Vect::Vect(const Vect &rhs) : pImpl(std::make_unique<VectImpl>(*rhs.pImpl)) {}

// Copy assignment
Vect &Vect::operator=(const Vect &rhs) {
  if (this != &rhs) {
    pImpl = std::make_unique<VectImpl>(*rhs.pImpl);
  }

  return *this;
}

Vect::Vect(Vect &&rhs) noexcept = default;

Vect &Vect::operator=(Vect &&rhs) noexcept = default;

Vect::Vect() : pImpl(std::make_unique<VectImpl>()) {}
