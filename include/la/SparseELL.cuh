//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by conor on 11/04/2022.
//

#ifndef MAIN_SPARSEELL_CUH
#define MAIN_SPARSEELL_CUH

#include "la/Vect.cuh"
#include "la/Dense.h"
#include "la/Sparse.cuh"

struct SparseELL {
    int DimX;
    int DimY;
    int NNZ;
    int EntriesPerRow;

    Dense Values;
    Dense Indices;

    SparseELL() : DimX(0), DimY(0) { };
    SparseELL(int dimX, int dimY, int EntriesPerRow) 
        : 
          DimX(dimX), 
          DimY(dimY), 
          Values(dimX, EntriesPerRow), 
          Indices(dimX, EntriesPerRow),
          EntriesPerRow(EntriesPerRow)
        { };

    void LoadDense(const Dense& A);

    Dense ToDense();
    Vect VectMult(const Vect &vect) const;
};

SparseELL ToSparseELL(const Sparse& A);

#endif //MAIN_SPARSEELL_CUH
