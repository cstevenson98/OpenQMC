//
// Created by Conor Stevenson on 03/04/2022.
//

#ifndef MAIN_SPARSE_CUH
#define MAIN_SPARSE_CUH

#include <complex>
#include <vector>
#include "Dense.cuh"
#include "Vect.cuh"

using namespace std;
using t_cplx = thrust::complex<double>;
using t_hostVect = thrust::host_vector<thrust::complex<double>>;

struct COOTuple {
    int    Coords[2]{0, 0};
    t_cplx Val;

    COOTuple(int x, int y, t_cplx val) : Val(val) {
      Coords[0] = x; Coords[1] = y;
    };
};

bool operator == (const COOTuple& A, const COOTuple& B) {
    return A.Coords[0] == B.Coords[0] && A.Coords[1] == B.Coords[1] && A.Val == B.Val;
}

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

    Sparse Scale(const t_cplx &alpha) const;
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

bool operator == (const Sparse& A, const Sparse& B) {
    if (A.DimX != B.DimX || A.DimY != B.DimY) return false;

    return A.Data == B.Data;
}

Sparse ToSparseCOO(const Dense& d);
vector<CompressedRow> SparseRowsCOO(const Sparse& s);
vector<CompressedRow> SparseColsCOO(const Sparse& s);
CompressedRow SparseVectorSum(const CompressedRow& A, const CompressedRow& B);
complex<double> SparseDot(const CompressedRow& A, const CompressedRow& B);

Sparse operator * (const t_cplx& alpha, const Sparse& rhs);

#endif //MAIN_SPARSE_CUH
