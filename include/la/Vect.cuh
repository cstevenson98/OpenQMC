//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by Conor Stevenson on 03/04/2022.
//

#ifndef MAIN_VECT_CUH
#define MAIN_VECT_CUH

#include <complex>
#include <thrust/complex.h>
#include <thrust/host_vector.h>

using t_cplx = thrust::complex<double>;
using t_hostVect = thrust::host_vector<thrust::complex<double>>;

struct Vect {
    explicit Vect() = default;
    explicit Vect(unsigned int N);
    explicit Vect(t_hostVect &in);

    t_hostVect Data;

    Vect Add(const Vect& A) const;
    Vect Subtract(const Vect& A) const;
    Vect Scale(const t_cplx& alpha) const;
    Vect AddScaledVect(const t_cplx& alpha, const Vect& A) const;
    Vect Conj() const;
    Vect operator + (const Vect& A) const;
    Vect operator - (const Vect& A) const;
    Vect operator * (const t_cplx& alpha) const;
    std::complex<double> operator [] (unsigned int i) const;

    double Dot(const Vect& A) const;
    double Norm() const;

    void Print(unsigned int kind) const;
    void PrintRe() const;
    void PrintIm() const;
    void PrintAbs() const;
};

// Non-member binary operators
Vect operator * (const t_cplx& alpha, const Vect& rhs);

#endif //MAIN_VECT_CUH
