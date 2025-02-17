//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by Conor Stevenson on 03/04/2022.
//

#ifndef MAIN_DENSE_CUH
#define MAIN_DENSE_CUH

#include <complex>
#include <thrust/complex.h>
#include <thrust/host_vector.h>

using t_cplx = thrust::complex<double>;
using t_hostVect = thrust::host_vector<thrust::complex<double>>;
using t_hostMat = thrust::host_vector<thrust::host_vector<thrust::complex<double>>>;
using t_hostVectInt = thrust::host_vector<int>;

struct Dense {
    int DimX;
    int DimY;
    t_hostMat Data;

    Dense() : DimX(0), DimY(0) { };
    Dense(int dimX, int dimY);

    struct matrix_row {
        t_hostVect& row;
        explicit matrix_row(t_hostVect& r) : row(r) { }

        std::complex<double> operator[](int y) {
            return row[y];
        }
    };

    matrix_row operator[] (unsigned int x) {
        return matrix_row(Data[x]);
    }

    Dense Add(const Dense &A) const;
    Dense RightMult(const Dense &A) const;
    Dense Scale(t_cplx alpha) const;
    Dense Transpose() const;
    Dense HermitianC() const;
    t_hostVect FlattenedData() const;
    t_hostVectInt FlattenedDataInt() const;

    Dense operator + (const Dense& A) const;
    Dense operator - (const Dense& A) const;
    Dense operator * (const t_cplx& alpha) const;
    Dense operator * (const Dense& A) const;
    Dense operator % (const Dense& A) const;

    void Print(unsigned int kind = 0, unsigned int prec = 2) const;
    void PrintRe(unsigned int prec) const;
    void PrintIm(unsigned int prec) const;
    void PrintAbs(unsigned int prec) const;
};

Dense operator * (const std::complex<double>& alpha, const Dense& rhs);

#endif //MAIN_DENSE_CUH
