//
// Created by Conor Stevenson on 03/04/2022.
//

#include "Vect.h"

Vect Vect::Add(Vect &A) {
    Vect out(Data.size());
    out.Data.resize(Data.size());

    for (int i = 0; i < A.Data.size(); ++i) {
        out.Data[i] = Data[i] + A.Data[i];
    }

    return out;
}

Vect Vect::Subtract(Vect &A) {
    Vect out(Data.size());
    out.Data.resize(Data.size());

    for (int i = 0; i < A.Data.size(); ++i) {
        out.Data[i] = Data[i] - A.Data[i];
    }

    return out;
}

Vect Vect::Scale(complex<double> alpha) {
    Vect out(Data.size());
    out.Data.resize(Data.size());

    for (int i = 0; i < Data.size(); ++i) {
        out.Data[i] = alpha * Data[i];
    }

    return out;
}

Vect Vect::AddScaledVect(complex<double> alpha, Vect &A) {
    Vect out(Data.size());
    out.Data.resize(Data.size());

    for (int i = 0; i < A.Data.size(); ++i) {
        out.Data[i] = Data[i] + alpha * A.Data[i];
    }

    return out;
}

Vect::Vect(unsigned int size) {
    Data.resize(size);
}

Vect::Vect(vector<complex<double> > &in) {
    Data = in;
}

double Vect::Norm() {
    double out = 0;

    for (auto elem : Data) {
        out += abs(elem);
    }
    return sqrt(out);
}
