//
// Created by Conor Stevenson on 03/04/2022.
//

#include<iostream>
#include <fstream>

#include "la/Dense.h"
#include "la/Sparse.h"
#include "ode/Euler.h"
#include "models/ElasticChain1D.h"
#include "ode/RK4.h"

void ToCSV(const string& filename, const vector<State>& results) {
    std::ofstream myfile;
    myfile.open(filename);
    for (const auto& r : results) {
        myfile << r.T << " " ;
        for (const auto elem : r.Y.Data) {
            myfile << elem.real() << " ";

        }
        myfile << r.E << std::endl;
    }
    myfile.close();
}

int main() {
    const int N = 4;

    Vect x0(2*N);
    for (int i = 0; i < 2*N; ++i) {
        if (i%2 == 0) {
            x0.Data[i] = {1., 0.};
        } else {
            x0.Data[i] = {0., 0.};
        }
    }

    ElasticChain1D chain(N, 0.5, 1.);

    chain.Dx().VectMult(x0);
    auto myFunc = [&chain](Vect& dx, double t, const Vect& x) {
        dx = chain.Dx().VectMult(x);
    };


    IVP   ballProblem(0., x0, myFunc);
    RK4 solver(ballProblem, 1e-1);

    auto results = SolveIVP(ballProblem, solver, 0.1, 4. * M_PI);

    ToCSV("data/data.csv", results);
    return 0;
}
