//
// Created by Conor Stevenson on 03/04/2022.
//

#ifndef MAIN_DENSE_H
#define MAIN_DENSE_H

#include <vector>
#include <complex>

using namespace std;

struct Dense {
    int DimX;
    int DimY;
    vector<vector<complex<double> > > Data;

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

    Dense Add(const Dense &A);
    Dense RightMult(const Dense &A);
    Dense Scale(complex<double> alpha);
    Dense Transpose();
    Dense HermitianC();

    void Print();
};


#endif //MAIN_DENSE_H
