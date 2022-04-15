//
// Created by Conor Stevenson on 03/04/2022.
//

#include <iostream>
#include <cassert>
#include <cmath>

#include "Dense.cuh"
#include "Sparse.cuh"
#include "../utils/SignPadding.cuh"

using namespace std;

Dense::Dense(int dimX, int dimY) : DimX(dimX), DimY(dimY) {
    Data.resize(dimX, vector<complex<double> >(dimY));
}

Dense Dense::Add(const Dense& A) const {
    assert(DimX == A.DimX && DimY == A.DimY);

    Dense out(DimX, DimY);
    for (int i = 0; i < DimX; ++i) {
        for (int j = 0; j < DimY; ++j) {
            out.Data[i][j] = Data[i][j] + A.Data[i][j];
        }
    }

    return out;
}

Dense Dense::RightMult(const Dense& A) const {
    assert(DimY == A.DimX);

    Dense out(DimX, A.DimY);
    for (int i = 0; i < DimX; ++i) {
        for (int j = 0; j < DimY; ++j) {
            complex<double> sum = 0;
            for (int k = 0; k < DimY; ++k) {
                sum += Data[i][k] * A.Data[k][j];
            }
            out.Data[i][j] = sum;
        }
    }

    return out;
}

Dense Dense::Scale(complex<double> alpha) const {
    Dense out(DimX, DimY);

    for (int i = 0; i < out.Data.size(); ++i) {
        for (int j = 0; j < out.Data[0].size(); ++j) {
            out.Data[i][j] = alpha * Data[i][j];
        }
    }

    return out;
}

Dense Dense::Transpose() const {
    Dense out(DimY, DimX);

    for (int i = 0; i < DimY; ++i) {
        for (int j = 0; j < DimX; ++j) {
            out.Data[i][j] = Data[j][i];
        }
    }

    return out;
}

Dense Dense::HermitianC() const {
    Dense out(DimY, DimX);

    for (int i = 0; i < DimY; ++i) {
        for (int j = 0; j < DimX; ++j) {
            out.Data[i][j] = conj(Data[j][i]);
        }
    }

    return out;
}

vector<complex<double> > Dense::FlattenedData() const {
    vector<complex<double> > out;
    out.resize(DimX * DimY);

    for (int i = 0; i < DimX; i++) {
        for (int j = 0; j < DimY; j++) {
            out[i + j * DimX] = Data[i][j];
        }
    }
    
    return out;
}

vector<int> Dense::FlattenedDataInt() const {
    vector<int> out;
    
    out.resize(DimX * DimY);

    for (int i = 0; i < DimX; i++) {
        for (int j = 0; j < DimY; j++) {
            out[i + j * DimX] = round(abs(Data[i][j]));
        }
    }
    
    return out;
}

void Dense::Print(unsigned int kind) const {
    string s;
    stringstream stream;
    stream.setf(ios::fixed);
    stream.precision(2);

    stream << " Matrix [" << DimX << " x " << DimY << "]:" << endl;
    for (const auto& X : Data) {
        stream << "   ";
        for (auto Y : X) {
            string spaceCharRe = !std::signbit(Y.real()) ? " " : "";
            string spaceCharIm = !std::signbit(Y.imag()) ? " " : "";
            string spaceCharAbs = !std::signbit(Y.imag()) ? " + " : " - ";

            switch (kind) {
                case 0: // re + im
                    stream << spaceCharRe << Y.real() << spaceCharAbs << abs(Y.imag()) << "i  ";
                    break;
                case 1: // re
                    stream << spaceCharRe << Y.real() << " ";
                    break;
                case 2: // im
                    stream << spaceCharIm << Y.imag() << "i  ";
                    break;
                case 3: // abs
                    stream << " " << abs(Y);
                    break;
                default:
                    stream << "[e]";
            }
        }
        stream << endl;
    }

    s = stream.str();

    cout.imbue(locale(cout.getloc(), new SignPadding));
    cout << s << endl;
}

void Dense::PrintRe() const {
    this->Print(1);
}

void Dense::PrintIm() const {
    this->Print(2);
}

void Dense::PrintAbs() const {
    this->Print(3);
}

Dense Dense::operator + (const Dense &A) const {
    return this->Add(A);
}

Dense Dense::operator - (const Dense &A) const {
    return this->Add(A.Scale(-1));
}

Dense Dense::operator * (const complex<double> &alpha) const {
    return this->Scale(alpha);
}

Dense operator * (const complex<double> &alpha, const Dense& rhs) {
    return rhs*alpha;
}

Dense Dense::operator * (const Dense &A) const {
    return this->RightMult(A);
}

Dense Dense::operator % (const Dense &A) const {
    return {0, 0};
}
