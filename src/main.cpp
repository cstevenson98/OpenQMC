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
    sparse mult = sparse1.RightMult(sparse1);
    dense d = mult.ToDense();

    return 0;
}
