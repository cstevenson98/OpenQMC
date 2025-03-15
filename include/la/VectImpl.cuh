//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by Conor Stevenson on 15/3/2025.
//

#include "core/types.cuh"
#include "la/Vect.h"

class VectImpl {
public:
  explicit VectImpl() = default;
  VectImpl(unsigned int size) { Data.resize(size); }
  explicit VectImpl(t_hostVect &in) { Data = in; }

  VectImpl Conj() const;
  VectImpl Add(const VectImpl &A) const;
  VectImpl Subtract(const VectImpl &A) const;
  VectImpl Scale(const th_cplx &alpha) const;
  double Dot(const VectImpl &A) const;
  double Norm() const;
  VectImpl operator+(const VectImpl &A) const;
  VectImpl operator-(const VectImpl &A) const;
  VectImpl operator*(const th_cplx &alpha) const;
  std::complex<double> operator[](unsigned int i) const;
  void Print(unsigned int kind) const;
  void PrintRe() const;
  void PrintIm() const;
  void PrintAbs() const;
  std::vector<std::complex<double>> GetData() const;

private:
  th_hostVect Data;
};

VectImpl operator*(const th_cplx &alpha, const VectImpl &rhs);
