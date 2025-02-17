//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by Conor Stevenson on 03/04/2022.
//

#include <iostream>
#include <cassert>
#include <cmath>
#include <complex>

#include "la/Dense.cuh"
#include "la/Sparse.cuh"
#include "utils/SignPadding.cuh"

using t_cplx = thrust::complex<double>;
using t_hostVect = thrust::host_vector<thrust::complex<double>>;
using t_hostVectInt = thrust::host_vector<int>;

Dense::Dense(int dimX, int dimY) : DimX(dimX), DimY(dimY) {
    Data.resize(dimX, std::vector<std::complex<double> >(dimY));
}

Dense Dense::Add(const Dense& A) const {
    assert(DimX == A.DimX && DimY == A.DimY);

    Dense out(DimX, DimY);
    for (int i = 0; i < DimX; ++i) {
        for (int j = 0; j < DimY; ++j) {
            out.Data[i][j] = Data[i][j] + A.Data[i][j];
        }
    }

    return out;
}

Dense Dense::RightMult(const Dense& A) const {
    assert(DimY == A.DimX);

    Dense out(DimX, A.DimY);
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

Dense Dense::Scale(t_cplx alpha) const {
    Dense out(DimX, DimY);

    for (int i = 0; i < out.Data.size(); ++i) {
        for (int j = 0; j < out.Data[0].size(); ++j) {
            out.Data[i][j] = alpha * Data[i][j];
        }
    }

    return out;
}

Dense Dense::Transpose() const {
    Dense out(DimY, DimX);

    for (int i = 0; i < DimY; ++i) {
        for (int j = 0; j < DimX; ++j) {
            out.Data[i][j] = Data[j][i];
        }
    }

    return out;
}

Dense Dense::HermitianC() const {
    Dense out(DimY, DimX);

    for (int i = 0; i < DimY; ++i) {
        for (int j = 0; j < DimX; ++j) {
            out.Data[i][j] = conj(Data[j][i]);
        }
    }

    return out;
}

t_hostVect Dense::FlattenedData() const {
    t_hostVect out;
    out.resize(DimX * DimY);

    for (int i = 0; i < DimX; i++) {
        for (int j = 0; j < DimY; j++) {
            out[j + i * DimY] = Data[i][j];
        }
    }
    
    return out;
}

t_hostVectInt Dense::FlattenedDataInt() const {
    std::vector<int> out;
    
    out.resize(DimX * DimY);

    for (int i = 0; i < DimX; i++) {
        for (int j = 0; j < DimY; j++) {
            out[j + i * DimY] = round(abs(Data[i][j]));
        }
    }
    
    return out;
}

void Dense::Print(unsigned int kind, unsigned int prec) const {
    std::string s;
    std::stringstream stream;
    stream.setf(std::ios::fixed);
    stream.precision(prec);

    stream << " Matrix [" << DimX << " x " << DimY << "]:" << std::endl;
    for (const auto& X : Data) {
        stream << "   ";
        for (auto Y : X) {
            std::string spaceCharRe = !std::signbit(Y.real()) ? " " : "";
            std::string spaceCharIm = !std::signbit(Y.imag()) ? " " : "";
            std::string spaceCharAbs = !std::signbit(Y.imag()) ? " + " : " - ";

            switch (kind) {
                case 0: // re + im
                    stream << spaceCharRe << Y.real() << spaceCharAbs << abs(Y.imag()) << "i  ";
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

    std::cout.imbue(std::locale(std::cout.getloc(), new SignPadding));
    std::cout << s << std::endl;
}

void Dense::PrintRe(unsigned int prec) const {
    this->Print(1, prec);
}

void Dense::PrintIm(unsigned int prec) const {
    this->Print(2, prec);
}

void Dense::PrintAbs(unsigned int prec) const {
    this->Print(3, prec);
}

Dense Dense::operator + (const Dense &A) const {
    return this->Add(A);
}

Dense Dense::operator - (const Dense &A) const {
    return this->Add(A.Scale(-1));
}

Dense Dense::operator * (const t_cplx &alpha) const {
    return this->Scale(alpha);
}

Dense operator * (const t_cplx &alpha, const Dense& rhs) {
    return rhs*alpha;
}

Dense Dense::operator * (const Dense &A) const {
    return this->RightMult(A);
}

Dense Dense::operator % (const Dense &A) const {
    return {0, 0};
}
