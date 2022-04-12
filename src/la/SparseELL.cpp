//
// Created by conor on 11/04/2022.
//

#include "SparseELL.h"

SparseELL ToSparseELL(const Sparse& A) {
    auto rows = SparseRowsCOO(A);

    // determine length of longest row
    unsigned int highestCount = 0;
    for (auto row : rows) {
        unsigned int len = row.RowData.size();
        if (len > highestCount) {
            highestCount = len;
        }
    }

    SparseELL out(A.DimX, A.DimY, highestCount);
    for (auto row : rows) {
        for (int i = 0; i < row.RowData.size(); i++)
        {
            out.Values.Data[row.Index][i] = row.RowData[i].Val;
            out.Indices.Data[row.Index][i] = row.RowData[i].Coords[1];
        }

        // padding with '-1'
        for (int i = row.RowData.size(); i < highestCount; i++) {
            out.Values.Data[row.Index][i] = 0;
            out.Indices.Data[row.Index][i] = -1;
        }
    }

    return out;
}
