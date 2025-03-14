//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by Conor Stevenson on 03/04/2022.
//

#ifndef MAIN_SPARSE_CUH
#define MAIN_SPARSE_CUH

#include <complex>
#include <vector>
#include "la/Dense.h"
#include "la/Vect.cuh"

struct COOTuple {
    int    Coords[2]{0, 0};
    th_cplx Val;

    COOTuple(int x, int y, t_cplx val) : Val(val) {
      Coords[0] = x; Coords[1] = y;
    };
};

struct CompressedRow {
    int Index;
    std::vector<COOTuple> RowData;

    explicit CompressedRow(int index) : Index(index) { };
    CompressedRow(int index, const std::vector<COOTuple>& rowData) : Index(index) {
       RowData = rowData;
    };
};

struct Sparse {
    int DimX;
    int DimY;
    std::vector<COOTuple> Data;

    Sparse(int dimX, int dimY) : DimX(dimX), DimY(dimY) { };

    Sparse Scale(const th_cplx &alpha) const;
    Sparse Add(const Sparse& B) const;
    Sparse RightMult(const Sparse& A) const;
    Sparse Transpose() const;
    Sparse HermitianC() const;
    Dense ToDense();
    void SortByRow();
    void Trim();
    Vect VectMult(const Vect &vect) const;

    Sparse operator + (const Sparse& A) const;
    Sparse operator - (const Sparse& A) const;
    Sparse operator * (const t_cplx& alpha) const;
    Sparse operator * (const Sparse& A) const;
    Sparse operator % (const Sparse& A) const;

    void Print() const;
    void PrintRe() const;
    void PrintIm() const;
    void PrintAbs() const;

    unsigned int NNZ() const;
};

Sparse ToSparseCOO(const Dense& d);
std::vector<CompressedRow> SparseRowsCOO(const Sparse& s);
std::vector<CompressedRow> SparseColsCOO(const Sparse& s);
CompressedRow SparseVectorSum(const CompressedRow& A, const CompressedRow& B);
std::complex<double> SparseDot(const CompressedRow& A, const CompressedRow& B);

Sparse operator * (const std::complex<double>& alpha, const Sparse& rhs);

#endif //MAIN_SPARSE_CUH
