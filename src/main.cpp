//
// Created by Conor Stevenson on 03/04/2022.
//

#include<iostream>

#include "la/Dense.h"
#include "la/Sparse.h"
#include "ode/Euler.h"

int main() {
    cout << "Hello" << endl;

    Dense denseMat1(2, 2);

    auto myFunc = [](Vect& dx, double t, const Vect& x) {
        dx.Data[0] = x.Data[1];
        dx.Data[1] = -x.Data[0];
    };

    Vect x0(2);
    x0.Data = {1., 0.};

    IVP ballProblem(0., x0, myFunc);
    Euler solver(ballProblem, 10e-3);

    SolveIVP(ballProblem, solver, 0.1, 1.);

    return 0;
}
