//
// Created by Conor Stevenson on 03/04/2022.
//

#include "dense.h"

dense::dense(int dimX, int dimY) : DimX(dimX), DimY(dimY) {
    Data.resize(dimX, std::vector<std::complex<double> >(dimY));
}

