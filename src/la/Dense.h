//
// Created by Conor Stevenson on 03/04/2022.
//

#ifndef MAIN_DENSE_H
#define MAIN_DENSE_H


#include <vector>
#include <complex>

struct Dense {
    int DimX;
    int DimY;
    std::vector<std::vector<std::complex<double> > > Data;

    Dense(int dimX, int dimY);

    struct matrix_row {
        std::vector<std::complex<double> >& row;
        explicit matrix_row(std::vector<std::complex<double> >& r) : row(r) { }

        std::complex<double> operator[](int y) {
            return row.at(y);
        }
    };

    matrix_row operator[] (unsigned int x) {
        return matrix_row(Data.at(x));
    }
};


#endif //MAIN_DENSE_H
