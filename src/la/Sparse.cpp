//
// Created by Conor Stevenson on 03/04/2022.
//

#include <algorithm>
#include "Dense.h"
#include "Sparse.h"

Sparse Sparse::Scale(complex<double> alpha) {
    Sparse out(DimX, DimY);
    out.Data.resize(Data.size(), COOTuple(0, 0, 0));

    for (int i = 0; i < Data.size(); ++i) {
        out.Data[i] = COOTuple(
                Data[i].Coords[0], Data[i].Coords[0],
                alpha * Data[i].Val
                );
    }
    return out;
}

Sparse Sparse::Add(Sparse A) {
    Sparse out(A.DimX, A.DimY);

    auto rowsA = SparseRowsCOO(*this);
    auto rowsB = SparseRowsCOO(*this);

    unsigned int i = 0, j = 0;
    unsigned int I = rowsA.size();
    unsigned int J = rowsB.size();

    while (true) {
        if (i > I-1 || j > J-1) {
            break;
        }

        if (rowsA[i].Index == rowsB[j].Index) {
            auto sumRow = SparseVectorSum(rowsA[i], rowsB[j]);
            for (auto &elem : sumRow.RowData) {
                out.Data.emplace_back(elem.Coords[0], elem.Coords[1], elem.Val);
            }
            i++;
            j++;
            continue;
        }

        if (rowsA[i].Index < rowsB[j].Index) {
            for (auto &elem : rowsA[i].RowData) {
                out.Data.emplace_back(elem.Coords[0], elem.Coords[1], elem.Val);
            }
            i++;
            continue;
        }

        if (rowsA[i].Index > rowsB[j].Index) {
            for (auto &elem : rowsB[j].RowData) {
                out.Data.emplace_back(elem.Coords[0], elem.Coords[1], elem.Val);
            }
            j++;
            continue;
        }

    }

    return out;
}

Sparse Sparse::RightMult(const Sparse& A) const {
    Sparse out(DimX, A.DimY);

    auto thisRows = SparseRowsCOO(*this);
    auto ACols = SparseColsCOO(A);

    for (auto &row : thisRows) {
        int i = row.Index;
        for (auto &col : ACols) {
            int j = col.Index;
            out.Data.emplace_back(i, j, SparseDot(row, col));
        }
    }

    return out;
}

Sparse Sparse::Transpose() const {
    Sparse out(DimX, DimY);
    out.Data.resize(Data.size(), COOTuple(0, 0, 0));

    for (int i = 0; i < Data.size(); i++) {
        out.Data[i].Coords[0] = Data[i].Coords[1];
        out.Data[i].Coords[1] = Data[i].Coords[0];
        out.Data[i].Val = Data[i].Val;
    }

    out.SortByRow();

    return out;
}

Sparse Sparse::HermitianC(const Sparse& A) {
    Sparse out(DimX, DimY);
    out.Data.resize(Data.size(), COOTuple(0, 0, 0));

    for (int i = 0; i < Data.size(); i++) {
        out.Data[i].Coords[0] = Data[i].Coords[1];
        out.Data[i].Coords[1] = Data[i].Coords[0];
        out.Data[i].Val = conj(Data[i].Val);
    }

    out.SortByRow();

    return out;
}

bool CompareCOORows(COOTuple L, COOTuple R) {
    if (L.Coords[0] == R.Coords[0]) {
        return L.Coords[1] < R.Coords[1];
    }
    return L.Coords[0] < R.Coords[0];
};

void Sparse::SortByRow() {
    std::sort(Data.begin(), Data.end(), CompareCOORows);
}

Sparse ToSparseCOO(Dense d) {
    Sparse out(d.DimX, d.DimY);

    for (int i = 0; i < d.DimX; i++) {
        for (int j = 0; j < d.DimY; j++) {
            if (d.Data[i][j] != 0. ) {
                out.Data.emplace_back(i, j, d.Data[i][j]);
            }
        }
    }

    return out;
}

vector<CompressedRow> SparseRowsCOO(const Sparse& s) {
    vector<CompressedRow> sRows;
    vector<COOTuple> rowData;

    int index = 0;
    for (auto data : s.Data) {
        if (data.Coords[0] != index) {
            if (!rowData.empty()) {
                sRows.emplace_back(CompressedRow(index, rowData));

                rowData.erase(rowData.begin(), rowData.end());
            }

            index = data.Coords[0];
        }

        rowData.emplace_back(COOTuple(
                data.Coords[0], data.Coords[1],
                data.Val
                ));

    }

    sRows.emplace_back(CompressedRow(index, rowData));

    return sRows;
}

vector<CompressedRow> SparseColsCOO(const Sparse& s) {
    Sparse sT = s.Transpose();

    return SparseRowsCOO(sT);
}

CompressedRow SparseVectorSum(const CompressedRow &rowA, const CompressedRow &rowB) {
    unsigned int i = 0, j = 0;
    unsigned int I = rowA.RowData.size();
    unsigned int J = rowB.RowData.size();

    CompressedRow out(rowA.Index);
    while (true) {
        if (i > I-1 || j > J-1) {
            break;
        }

        if (rowA.RowData[i].Coords[1] == rowB.RowData[j].Coords[1]) {
            out.RowData.emplace_back(COOTuple(
                    rowA.Index, rowA.RowData[i].Coords[1],
                    rowA.RowData[i].Val + rowB.RowData[j].Val
                    ));
            i++;
            j++;
            continue;
        }

        if (rowA.RowData[i].Coords[1] < rowB.RowData[j].Coords[1]) {
            out.RowData.emplace_back(rowA.RowData[i]);
            i++;
            continue;
        }

        if (rowA.RowData[i].Coords[1] > rowB.RowData[j].Coords[1]) {
            out.RowData.emplace_back(rowB.RowData[j]);
            j++;
            continue;
        }
    }

    return out;
}

complex<double> SparseDot(const CompressedRow& A, const CompressedRow& B) {
    unsigned int i = 0, j = 0;
    unsigned int I = A.RowData.size();
    unsigned int J = B.RowData.size();

    complex<double> out = 0;
    while (true) {
        if (i > I-1 || j > J-1) {
            break;
        }

        if (A.RowData[i].Coords[1] == B.RowData[j].Coords[1]) {
            out += A.RowData[i].Val * B.RowData[j].Val;
            i++;
            j++;
            continue;
        }

        if (A.RowData[i].Coords[1] < B.RowData[j].Coords[1]) {
            i++;
            continue;
        }

        if (A.RowData[i].Coords[1] > B.RowData[j].Coords[1]) {
            j++;
            continue;
        }
    }

    return out;
}

Dense Sparse::ToDense() {
    Dense out(DimX, DimY);

    for (auto &elem : Data) {
        out.Data[elem.Coords[0]][elem.Coords[1]] = elem.Val;
    }

    return out;
}

Vect Sparse::VectMult(const Vect &vect) const {
    vector<CompressedRow> sRows = SparseRowsCOO(*this);

    Vect out(vect.Data.size());
    for (const auto& row : sRows) {
            for (const auto rowElem : row.RowData) {
                out.Data[row.Index] += rowElem.Val * vect.Data[rowElem.Coords[1]];
            }
    }

    return out;
}
