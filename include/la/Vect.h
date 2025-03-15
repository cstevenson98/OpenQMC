//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by Conor Stevenson on 03/04/2022.
//

#ifndef MAIN_VECT_CUH
#define MAIN_VECT_CUH

#include <complex>

#include "core/types.h"
#include <memory>

class VectImpl;
class Vect {
public:
    explicit Vect();
    explicit Vect(unsigned int N);
    explicit Vect(t_hostVect &in);
    ~Vect();
    Vect(const Vect &other);
    Vect(Vect &&other) noexcept;
    Vect &operator=(const Vect &other);
    Vect &operator=(Vect &&other) noexcept;

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
    int size() const;

    std::vector<std::complex<double>> GetData() const;

private:
    std::unique_ptr<VectImpl> pImpl;
};

// Non-member binary operators
Vect operator * (const t_cplx& alpha, const Vect& rhs);

// Non-member binary operators

#endif //MAIN_VECT_CUH
