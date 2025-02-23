//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by Conor Stevenson on 03/04/2022.
//

#include <cassert>
// #include <cmath>
#include "la/Dense.h"
#include <complex>
#include <memory>
#include <vector>
// #include "la/Sparse.cuh"

using t_cplx = std::complex<double>;
using t_hostVect = std::vector<std::complex<double>>;
using t_hostVectInt = std::vector<int>;

class Dense::DenseImpl {
public:
  int DimX;
  int DimY;
  std::vector<std::vector<std::complex<double>>> Data;

  DenseImpl(int dimX, int dimY) : DimX(dimX), DimY(dimY) {
    Data.resize(dimX, std::vector<std::complex<double>>(dimY));
  }
  ~DenseImpl() = default;
  DenseImpl Add(const DenseImpl &A) const;
  DenseImpl RightMult(const DenseImpl &A) const;
  DenseImpl Scale(t_cplx alpha) const;
  DenseImpl Transpose() const;
  DenseImpl HermitianC() const;
  t_hostVect FlattenedData() const;
  t_hostVectInt FlattenedDataInt() const;
  // void Print(unsigned int kind, unsigned int prec) const;
  // void PrintRe(unsigned int prec) const;
  // void PrintIm(unsigned int prec) const;
  // void PrintAbs(unsigned int prec) const;
};

Dense::Dense(int dimX, int dimY)
    : pImpl(std::make_unique<DenseImpl>(dimX, dimY)) {}
Dense::Dense(std::unique_ptr<DenseImpl> &pImpl) : pImpl(std::move(pImpl)) {}
Dense::~Dense() = default;

Dense Dense::operator+(const Dense &A) const {
  auto impl = std::make_unique<DenseImpl>(pImpl->Add(*A.pImpl));
  return Dense(impl); //
}

Dense Dense::operator*(const Dense &A) const {
  auto impl = std::make_unique<DenseImpl>(pImpl->RightMult(*A.pImpl));
  return Dense(impl);
}

Dense Dense::operator*(const t_cplx &alpha) const {
  auto impl = std::make_unique<DenseImpl>(pImpl->Scale(alpha));
  return Dense(impl);
}

Dense operator*(const t_cplx &alpha, const Dense &rhs) { return rhs * alpha; }

Dense Dense::operator-(const Dense &A) const {
  return this->operator+(A.operator*(t_cplx(-1)));
}

// Dense::matrix_row Dense::matrix_row::operator[] (unsigned int x)  {
//     return Dense::matrix_row(pImpl->Data[x]);
// }

// Dense Dense::operator % (const Dense &A) const {
//     return {0, 0};
// }

Dense Dense::Transpose() const {
  auto impl = std::make_unique<DenseImpl>(pImpl->Transpose());
  return Dense(impl);
}

Dense Dense::HermitianC() const {
  auto impl = std::make_unique<DenseImpl>(pImpl->HermitianC());
  return Dense(impl);
}

t_hostVect Dense::FlattenedData() const { return pImpl->FlattenedData(); }

t_hostVectInt Dense::FlattenedDataInt() const {
  return pImpl->FlattenedDataInt();
}

// void Dense::Print(unsigned int kind, unsigned int prec) const {
//   pImpl->Print(kind, prec);
// }

// void Dense::PrintRe(unsigned int prec) const {
//     pImpl->PrintRe(prec);
// }

// void Dense::PrintIm(unsigned int prec) const {
//     pImpl->PrintIm(prec);
// }

// void Dense::PrintAbs(unsigned int prec) const {
//     pImpl->PrintAbs(prec);
// }

Dense::DenseImpl Dense::DenseImpl::Add(const Dense::DenseImpl &A) const {
  assert(DimX == A.DimX && DimY == A.DimY);

  Dense::DenseImpl out(DimX, DimY);
  for (int i = 0; i < DimX; ++i) {
    for (int j = 0; j < DimY; ++j) {
      out.Data[i][j] = Data[i][j] + A.Data[i][j];
    }
  }
  return out;
}

Dense::DenseImpl Dense::DenseImpl::RightMult(const Dense::DenseImpl &A) const {
  assert(DimY == A.DimX);

  Dense::DenseImpl out(DimX, A.DimY);
  for (int i = 0; i < DimX; ++i) {
    for (int j = 0; j < DimY; ++j) {
      t_cplx sum = 0;
      for (int k = 0; k < DimY; ++k) {
        sum += Data[i][k] * A.Data[k][j];
      }
      out.Data[i][j] = sum;
    }
  }

  return out;
}

Dense::DenseImpl Dense::DenseImpl::Scale(t_cplx alpha) const {
  Dense::DenseImpl out(DimX, DimY);

  for (int i = 0; i < out.Data.size(); ++i) {
    for (int j = 0; j < out.Data[0].size(); ++j) {
      out.Data[i][j] = alpha * Data[i][j];
    }
  }

  return out;
}

Dense::DenseImpl Dense::DenseImpl::Transpose() const {
  Dense::DenseImpl out(DimY, DimX);

  for (int i = 0; i < DimY; ++i) {
    for (int j = 0; j < DimX; ++j) {
      out.Data[i][j] = Data[j][i];
    }
  }

  return out;
}

Dense::DenseImpl Dense::DenseImpl::HermitianC() const {
  Dense::DenseImpl out(DimY, DimX);

  for (int i = 0; i < DimY; ++i) {
    for (int j = 0; j < DimX; ++j) {
      out.Data[i][j] = conj(Data[j][i]);
    }
  }

  return out;
}

t_hostVect Dense::DenseImpl::FlattenedData() const {
  t_hostVect out;
  out.resize(DimX * DimY);

  for (int i = 0; i < DimX; i++) {
    for (int j = 0; j < DimY; j++) {
      out[j + i * DimY] = Data[i][j];
    }
  }

  return out;
}

t_hostVectInt Dense::DenseImpl::FlattenedDataInt() const {
  std::vector<int> out;

  out.resize(DimX * DimY);

  for (int i = 0; i < DimX; i++) {
    for (int j = 0; j < DimY; j++) {
      out[j + i * DimY] = round(abs(Data[i][j]));
    }
  }

  return out;
}

// void Dense::DenseImpl::Print(unsigned int kind, unsigned int prec) const {
//     std::string s;
//     std::stringstream stream;
//     stream.setf(std::ios::fixed);
//     stream.precision(prec);

//     stream << " Matrix [" << DimX << " x " << DimY << "]:" << std::endl;
//     for (const auto& X : Data) {
//         stream << "   ";
//         for (auto Y : X) {
//             std::string spaceCharRe = !std::signbit(Y.real()) ? " " : "";
//             std::string spaceCharIm = !std::signbit(Y.imag()) ? " " : "";
//             std::string spaceCharAbs = !std::signbit(Y.imag()) ? " + " : "
//             -";

//             switch (kind) {
//                 case 0: // re + im
//                     stream << spaceCharRe << Y.real() << spaceCharAbs <<
//                     abs(Y.imag()) << "i  "; break;
//                 case 1: // re
//                     stream << spaceCharRe << Y.real() << " ";
//                     break;
//                 case 2: // im
//                     stream << spaceCharIm << Y.imag() << "i  ";
//                     break;
//                 case 3: // abs
//                     stream << " " << abs(Y);
//                     break;
//                 default:
//                     stream << "[e]";
//             }
//         }
//         stream << std::endl;
//     }

//     s = stream.str();

//     std::cout.imbue(std::locale(std::cout.getloc(), new SignPadding));
//     std::cout << s << std::endl;
// }

// Dense Dense::operator - (const Dense &A) const {
//     return this->Add(A.Scale(-1));
// }

// Dense Dense::operator * (const t_cplx &alpha) const {
//     return this->Scale(alpha);
// }

// Dense operator * (const t_cplx &alpha, const Dense& rhs) {
//     return rhs*alpha;
// }

// Dense Dense::operator * (const Dense &A) const {
//     return this->RightMult(A);
// }

// Dense Dense::operator % (const Dense &A) const {
//     return {0, 0};
// }
