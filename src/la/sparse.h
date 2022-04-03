//
// Created by Conor Stevenson on 03/04/2022.
//

#ifndef MAIN_SPARSE_H
#define MAIN_SPARSE_H

#include <complex>
#include <vector>

using namespace std;

struct cooTuple {
    int Coords[2]{0, 0};
    complex<double> Val;

    cooTuple(int x, int y, complex<double> val) : Val(val) {
      Coords[0] = x; Coords[1] = y;
    };
};

struct compressedRow {
    int Index;
    vector<cooTuple> RowData;

    explicit compressedRow(int index) : Index(index) { };
    compressedRow(int index, const vector<cooTuple>& rowData) : Index(index) {
       RowData = rowData;
    };
};

struct sparse {
    int DimX;
    int DimY;
    std::vector<cooTuple> Data;

    sparse(int dimX, int dimY) : DimX(dimX), DimY(dimY) { };
    sparse Scale(complex<double> alpha);
    sparse Add(sparse A);
    sparse RightMult(const sparse& A) const;
    sparse Transpose() const;
    sparse HermitianC(const sparse& A);

    dense ToDense();

    void SortByRow();
};

sparse ToSparseCOO(dense d);
vector<compressedRow> SparseRowsCOO(const sparse& s);
vector<compressedRow> SparseColsCOO(const sparse& s);
compressedRow SparseVectorSum(const compressedRow& A, const compressedRow& B);
complex<double> SparseDot(const compressedRow& A, const compressedRow& B);

#endif //MAIN_SPARSE_H
