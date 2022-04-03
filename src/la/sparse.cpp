//
// Created by Conor Stevenson on 03/04/2022.
//

#include "dense.h"
#include "sparse.h"

sparse sparse::Scale(complex<double> alpha) {
    sparse out(DimX, DimY);
    out.Data.resize(Data.size(), cooTuple(0, 0, 0));

    for (int i = 0; i < Data.size(); ++i) {
        out.Data[i] = cooTuple(
                Data[i].Coords[0], Data[i].Coords[0],
                alpha * Data[i].Val
                );
    }
    return out;
}

sparse sparse::Add(sparse A) {
    sparse out(A.DimX, A.DimY);

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

sparse sparse::RightMult(const sparse& A) const {
    sparse out(DimX, A.DimY);

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


sparse sparse::Transpose() const {
    sparse out(DimX, DimY);
    out.Data.resize(Data.size(), cooTuple(0, 0, 0));

    for (int i = 0; i < Data.size(); i++) {
        out.Data[i].Coords[0] = Data[i].Coords[1];
        out.Data[i].Coords[1] = Data[i].Coords[0];
        out.Data[i].Val = Data[i].Val;
    }

    out.SortByRow();

    return out;
}

sparse sparse::HermitianC(const sparse& A) {
    sparse out(DimX, DimY);
    out.Data.resize(Data.size(), cooTuple(0, 0, 0));

    for (int i = 0; i < Data.size(); i++) {
        out.Data[i].Coords[0] = Data[i].Coords[1];
        out.Data[i].Coords[1] = Data[i].Coords[0];
        out.Data[i].Val = conj(Data[i].Val);
    }

    out.SortByRow();

    return out;
}

bool CompareCOORows(cooTuple L, cooTuple R) {
    if (L.Coords[0] == R.Coords[0]) {
        return L.Coords[1] < R.Coords[1];
    }
    return L.Coords[0] < R.Coords[0];
};

void sparse::SortByRow() {
    std::sort(Data.begin(), Data.end(), CompareCOORows);
}

sparse ToSparseCOO(dense d) {
    sparse out(d.DimX, d.DimY);

    for (int i = 0; i < d.DimX; i++) {
        for (int j = 0; j < d.DimY; j++) {
            if (d.Data[i][j] != 0. ) {
                out.Data.emplace_back(i, j, d.Data[i][j]);
            }
        }
    }

    return out;
}

vector<compressedRow> SparseRowsCOO(const sparse& s) {
    vector<compressedRow> sRows;
    vector<cooTuple> rowData;

    int index = 0;
    for (auto data : s.Data) {
        if (data.Coords[0] != index) {
            if (!rowData.empty()) {
                sRows.emplace_back(compressedRow(index, rowData));

                rowData.erase(rowData.begin(), rowData.end());
            }

            index = data.Coords[0];
        }

        rowData.emplace_back(cooTuple(
                data.Coords[0], data.Coords[1],
                data.Val
                ));

    }

    sRows.emplace_back(compressedRow(index, rowData));

    return sRows;
}

vector<compressedRow> SparseColsCOO(const sparse& s) {
    sparse sT = s.Transpose();

    return SparseRowsCOO(sT);
}

compressedRow SparseVectorSum(const compressedRow &rowA, const compressedRow &rowB) {
    unsigned int i = 0, j = 0;
    unsigned int I = rowA.RowData.size();
    unsigned int J = rowB.RowData.size();

    compressedRow out(rowA.Index);
    while (true) {
        if (i > I-1 || j > J-1) {
            break;
        }

        if (rowA.RowData[i].Coords[1] == rowB.RowData[j].Coords[1]) {
            out.RowData.emplace_back(cooTuple(
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

complex<double> SparseDot(const compressedRow& A, const compressedRow& B) {
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

dense sparse::ToDense() {
    dense out(DimX, DimY);

    for (auto &elem : Data) {
        out.Data[elem.Coords[0]][elem.Coords[1]] = elem.Val;
    }

    return out;
}
