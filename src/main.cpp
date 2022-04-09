//
// Created by Conor Stevenson on 03/04/2022.
//

#include "la/Dense.h"

int main() {
    Dense mat1(2, 2);
    mat1.Data = {{1,   {1, -1}},
                 {0.5, -2}};

    return 0;
}
