//
// Created by Conor Stevenson on 03/04/2022.
//

#include "Vect.h"

Vect Vect::Add(const Vect &A) const {
    Vect out(Data.size());
    out.Data.resize(Data.size());

    for (int i = 0; i < A.Data.size(); ++i) {
        out.Data[i] = Data[i] + A.Data[i];
    }

    return out;
}

Vect Vect::Subtract(const Vect &A) const {
    Vect out(Data.size());
    out.Data.resize(Data.size());

    for (int i = 0; i < A.Data.size(); ++i) {
        out.Data[i] = Data[i] - A.Data[i];
    }

    return out;
}

Vect Vect::Scale(const complex<double>& alpha) const {
    Vect out(Data.size());
    out.Data.resize(Data.size());

    for (int i = 0; i < Data.size(); ++i) {
        out.Data[i] = alpha * Data[i];
    }

    return out;
}

Vect Vect::AddScaledVect(const complex<double>& alpha, const Vect &A) const {
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

double Vect::Dot(const Vect& A) const {
    double dot = 0;
    for (unsigned int i = 0; i < this->Data.size(); ++i) {
        dot += real(conj(this->Data[i]) * A.Data[i]);
    }

    return dot;
}

double Vect::Norm() const {
    double out = 0;
    for (auto elem : Data) {
        out += abs(elem);
    }

    return sqrt(out);
}

Vect Vect::operator+(const Vect &A) const {
    return this->Add(A);
}

Vect Vect::operator-(const Vect &A) const {
    return this->Subtract(A);
}

Vect Vect::operator*(const complex<double>& alpha) const {
    return this->Scale(alpha);
}

Vect operator*(const complex<double>& alpha, const Vect& rhs) {
    return rhs*alpha;
}

complex<double> Vect::operator[](unsigned int i) const {
    return this->Data.at(i);
}
