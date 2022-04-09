//
// Created by Conor Stevenson on 03/04/2022.
//

#ifndef MAIN_VECT_H
#define MAIN_VECT_H

#include <complex>
#include <vector>

using namespace std;

struct Vect {
    explicit Vect() = default;
    explicit Vect(unsigned int N);
    explicit Vect(vector<complex<double>> &in);

    vector<complex<double> > Data;

    Vect Add(const Vect& A) const;
    Vect Subtract(const Vect& A) const;
    Vect Scale(const complex<double>& alpha) const;
    Vect AddScaledVect(const complex<double>& alpha, const Vect& A) const;
    Vect Conj() const;
    Vect operator + (const Vect& A) const;
    Vect operator - (const Vect& A) const;
    Vect operator * (const complex<double>& alpha) const;
    complex<double> operator [] (unsigned int i) const;

    double Dot(const Vect& A) const;
    double Norm() const;

    void Print() const;
    void PrintRe() const;
    void PrintIm() const;
    void PrintAbs() const;
};

// Non-member binary operators
Vect operator * (const complex<double>& alpha, const Vect& rhs);

#endif //MAIN_VECT_H
