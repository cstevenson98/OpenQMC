//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by Conor Stevenson on 03/04/2022.
//

#ifndef MAIN_DENSE_CUH
#define MAIN_DENSE_CUH

#include <complex>
#include <memory>
#include <vector>

using t_cplx = std::complex<double>;
using t_hostVect = std::vector<std::complex<double>>;
using t_hostMat = std::vector<std::vector<std::complex<double>>>;
using t_hostVectInt = std::vector<int>;

class Dense {
public:
  Dense(int dimX, int dimY); //
  ~Dense();

  Dense operator+(const Dense &A) const;
  Dense operator-(const Dense &A) const;
  Dense operator*(const t_cplx &alpha) const;
  Dense operator*(const Dense &A) const;
  Dense operator%(const Dense &A) const;

  Dense Transpose() const;
  Dense HermitianC() const;
  t_hostVect FlattenedData() const;
  t_hostVectInt FlattenedDataInt() const;

  void Print(unsigned int kind = 0, unsigned int prec = 2) const;

private:
  class DenseImpl;
  std::unique_ptr<DenseImpl> pImpl;

  Dense(std::unique_ptr<DenseImpl> &pImpl);
};


Dense operator*(const std::complex<double> &alpha, const Dense &rhs);

#endif // MAIN_DENSE_CUH
