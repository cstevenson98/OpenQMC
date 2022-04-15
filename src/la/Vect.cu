//
// Created by Conor Stevenson on 03/04/2022.
//
#include <iostream>

#include "Vect.cuh"
#include "../utils/SignPadding.cuh"
#include <thrust/complex.h>
#include <thrust/host_vector.h>

using namespace std;
using t_cplx = thrust::complex<double>;
using t_hostVect = thrust::host_vector<thrust::complex<double>>;

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

Vect Vect::Scale(const t_cplx& alpha) const {
    Vect out(Data.size());
    out.Data.resize(Data.size());

    for (int i = 0; i < Data.size(); ++i) {
        out.Data[i] = alpha * Data[i];
    }

    return out;
}

Vect Vect::AddScaledVect(const t_cplx& alpha, const Vect &A) const {
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

Vect::Vect(t_hostVect &in) {
    Data = in;
}

double Vect::Dot(const Vect& A) const {
    double dot = 0;
    for (unsigned int i = 0; i < this->Data.size(); ++i) {
        dot += (conj(this->Data[i]) * A.Data[i]).real();
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

Vect Vect::operator*(const t_cplx& alpha) const {
    return this->Scale(alpha);
}

Vect operator*(const t_cplx& alpha, const Vect& rhs) {
    return rhs*alpha;
}

complex<double> Vect::operator[](unsigned int i) const {
    return this->Data[i];
}

void Vect::Print(unsigned int kind) const {
    string s;
    stringstream stream;
    stream.setf(ios::fixed);
    stream.precision(2);

    stream << " Vector [" << Data.size() << " x " << 1 << "]:" << endl;
    for (const auto& X : Data) {
        stream << "   ";
            string spaceCharRe = !std::signbit(X.real()) ? " " : "";
            string spaceCharIm = !std::signbit(X.imag()) ? " " : "";
            string spaceCharAbs = !std::signbit(X.imag()) ? " + " : " - ";

            switch (kind) {
                case 0: // re + im
                    stream << spaceCharRe << X.real() << spaceCharAbs << abs(X.imag()) << "i  ";
                    break;
                case 1: // re
                    stream << spaceCharRe << X.real() << " ";
                    break;
                case 2: // im
                    stream << spaceCharIm << X.imag() << "i  ";
                    break;
                case 3: // abs
                    stream << " " << abs(X);
                    break;
                default:
                    stream << "[e]";
            }
        stream << endl;
    }

    s = stream.str();

    cout.imbue(locale(cout.getloc(), new SignPadding));
    cout << s << endl;
}

void Vect::PrintRe() const {
    this->Print(1);
}

void Vect::PrintIm() const {
    this->Print(2);
}

void Vect::PrintAbs() const {
    this->Print(3);
}
