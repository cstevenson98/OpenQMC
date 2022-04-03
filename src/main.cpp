//
// Created by Conor Stevenson on 03/04/2022.
//

#include<iostream>

#include "la/dense.h"
#include "la/sparse.h"
#include "la/vect.h"

int main() {
    cout << "Hello" << endl;

    dense denseMat1(2, 2);

    denseMat1.Data = {
            {0, 2},
            {1, 0}
    };

    sparse sparse1 = ToSparseCOO(denseMat1);
    sparse sparse1Tr = sparse1.Transpose();
    vector<compressedRow> rows = SparseRowsCOO(sparse1);
    vector<compressedRow> cols = SparseColsCOO(sparse1);
    vector<complex<double> > data1{1, 0, 1};
    vector<complex<double> > data2{0, 1, 0};

    vect myVect1(data1);
    vect myVect2(data2);

    vect vect12 = myVect1.Add(myVect2);

    for (auto e : vect12.Data) {
        cout << e << endl;
    }

    return 0;
}
