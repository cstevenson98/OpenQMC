//
// Created by Conor Stevenson on 03/04/2022.
//

#ifndef MAIN_VECT_H
#define MAIN_VECT_H

#include <complex>
#include <vector>

using namespace std;

struct vect {
    explicit vect(unsigned int N);
    explicit vect(vector<complex<double>> &in);

    vector<complex<double> > Data;

    vect Add(vect& A);
    vect Subtract(vect& A);
    vect Scale(complex<double> alpha);
    vect AddScaledVect(complex<double> alpha, vect& A);
};

#endif //MAIN_VECT_H
