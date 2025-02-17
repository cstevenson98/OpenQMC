//
// Created by conor on 11/04/2022.
//

#ifndef MAIN_SPARSEELL_CUH
#define MAIN_SPARSEELL_CUH

#include <complex>
#include <vector>
#include "Vect.cuh"
#include "Dense.cuh"
#include "Sparse.cuh"

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
