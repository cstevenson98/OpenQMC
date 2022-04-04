//
// Created by Conor Stevenson on 03/04/2022.
//

#include <iostream>
#include "Dense.h"

using namespace std;

Dense::Dense(int dimX, int dimY) : DimX(dimX), DimY(dimY) {
    Data.resize(dimX, vector<complex<double> >(dimY));
}

Dense Dense::Add(const Dense& A) {
    assert(DimX == A.DimX && DimY == A.DimY);

    Dense out(DimX, DimY);
    for (int i = 0; i < DimX; ++i) {
        for (int j = 0; j < DimY; ++j) {
            out.Data[i][j] = Data[i][j] + A.Data[i][j];
        }
    }

    return out;
}

Dense Dense::RightMult(const Dense& A) {
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

Dense Dense::Scale(complex<double> alpha) {
    Dense out(DimX, DimY);

    for (int i = 0; i < out.Data.size(); ++i) {
        for (int j = 0; j < out.Data[0].size(); ++j) {
            out.Data[i][j] = alpha * Data[i][j];
        }
    }

    return out;
}

Dense Dense::Transpose() {
    Dense out(DimY, DimX);

    for (int i = 0; i < DimY; ++i) {
        for (int j = 0; j < DimX; ++j) {
            out.Data[i][j] = Data[j][i];
        }
    }

    return out;
}

Dense Dense::HermitianC() {
    Dense out(DimY, DimX);

    for (int i = 0; i < DimY; ++i) {
        for (int j = 0; j < DimX; ++j) {
            out.Data[i][j] = conj(Data[j][i]);
        }
    }

    return out;
}

void Dense::Print() {
    string s;
    stringstream stream;
    stream.setf(ios::fixed);
    stream.precision(2);

    stream << " Matrix [" << DimX << " x " << DimY << "]:" << endl;
    for (const auto& X : Data) {
        stream << "  ";
        for (auto Y : X) {
            stream << "("<< Y.real() << ", " << Y.imag() << ") ";
        }
        stream << endl;
    }

    s = stream.str();
    cout << s << endl;
}