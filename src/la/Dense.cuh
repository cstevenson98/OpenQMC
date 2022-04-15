//
// Created by Conor Stevenson on 03/04/2022.
//

#ifndef MAIN_DENSE_CUH
#define MAIN_DENSE_CUH

#include <vector>
#include <complex>

using namespace std;

struct Dense {
    int DimX;
    int DimY;
    vector<vector<complex<double> > > Data;

    Dense() : DimX(0), DimY(0) { };
    Dense(int dimX, int dimY);

    struct matrix_row {
        vector<complex<double> >& row;
        explicit matrix_row(vector<complex<double> >& r) : row(r) { }

        complex<double> operator[](int y) {
            return row.at(y);
        }
    };

    matrix_row operator[] (unsigned int x) {
        return matrix_row(Data.at(x));
    }

    Dense Add(const Dense &A) const;
    Dense RightMult(const Dense &A) const;
    Dense Scale(complex<double> alpha) const;
    Dense Transpose() const;
    Dense HermitianC() const;
    vector<complex<double> > FlattenedData() const;
    vector<int> FlattenedDataInt() const;

    Dense operator + (const Dense& A) const;
    Dense operator - (const Dense& A) const;
    Dense operator * (const complex<double>& alpha) const;
    Dense operator * (const Dense& A) const;
    Dense operator % (const Dense& A) const;

    void Print(unsigned int kind = 0) const;
    void PrintRe() const;
    void PrintIm() const;
    void PrintAbs() const;
};

Dense operator * (const complex<double>& alpha, const Dense& rhs);

#endif //MAIN_DENSE_CUH
