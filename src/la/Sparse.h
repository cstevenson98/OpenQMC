//
// Created by Conor Stevenson on 03/04/2022.
//

#ifndef MAIN_SPARSE_H
#define MAIN_SPARSE_H

#include <complex>
#include <vector>
#include "Dense.h"
#include "Vect.h"

using namespace std;

struct COOTuple {
    int Coords[2]{0, 0};
    complex<double> Val;

    COOTuple(int x, int y, complex<double> val) : Val(val) {
      Coords[0] = x; Coords[1] = y;
    };
};

struct CompressedRow {
    int Index;
    vector<COOTuple> RowData;

    explicit CompressedRow(int index) : Index(index) { };
    CompressedRow(int index, const vector<COOTuple>& rowData) : Index(index) {
       RowData = rowData;
    };
};

struct Sparse {
    int DimX;
    int DimY;
    std::vector<COOTuple> Data;

    Sparse(int dimX, int dimY) : DimX(dimX), DimY(dimY) { };
    Sparse Scale(complex<double> alpha);
    Sparse Add(const Sparse& A) const;
    Sparse RightMult(const Sparse& A) const;
    Sparse Transpose() const;
    Sparse HermitianC(const Sparse& A);

    Dense ToDense();

    void SortByRow();

    Vect VectMult(const Vect &vect) const;
};

Sparse ToSparseCOO(Dense d);
vector<CompressedRow> SparseRowsCOO(const Sparse& s);
vector<CompressedRow> SparseColsCOO(const Sparse& s);
CompressedRow SparseVectorSum(const CompressedRow& A, const CompressedRow& B);
complex<double> SparseDot(const CompressedRow& A, const CompressedRow& B);

#endif //MAIN_SPARSE_H
