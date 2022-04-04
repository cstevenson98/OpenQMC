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

    Vect Add(Vect& A);
    Vect Subtract(Vect& A);
    Vect Scale(complex<double> alpha);
    Vect AddScaledVect(complex<double> alpha, Vect& A);

    double Norm();
};

#endif //MAIN_VECT_H
