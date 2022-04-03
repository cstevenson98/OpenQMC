//
// Created by Conor Stevenson on 03/04/2022.
//

#include "vect.h"

vect vect::Add(vect &A) {
    vect out(Data.size());
    out.Data.resize(Data.size());

    for (int i = 0; i < A.Data.size(); ++i) {
        out.Data[i] = Data[i] + A.Data[i];
    }

    return out;
}

vect vect::Subtract(vect &A) {
    vect out(Data.size());
    out.Data.resize(Data.size());

    for (int i = 0; i < A.Data.size(); ++i) {
        out.Data[i] = Data[i] - A.Data[i];
    }

    return out;
}

vect vect::Scale(complex<double> alpha) {
    vect out(Data.size());
    out.Data.resize(Data.size());

    for (int i = 0; i < Data.size(); ++i) {
        out.Data[i] = alpha * Data[i];
    }

    return out;
}

vect vect::AddScaledVect(complex<double> alpha, vect &A) {
    vect out(Data.size());
    out.Data.resize(Data.size());

    for (int i = 0; i < A.Data.size(); ++i) {
        out.Data[i] = Data[i] + alpha * A.Data[i];
    }

    return out;
}

vect::vect(unsigned int size) {
    Data.resize(size);
}

vect::vect(vector<complex<double> > &in) {
    Data = in;
}