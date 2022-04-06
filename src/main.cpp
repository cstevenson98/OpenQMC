//
// Created by Conor Stevenson on 03/04/2022.
//

#include<iostream>
#include <fstream>

#include "la/Dense.h"
#include "la/Sparse.h"
#include "ode/Euler.h"
#include "ode/RK4.h"
#include "models/ElasticChain1D.h"
#include "models/XYZModel.h"
#include "qm/Spins.h"

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
    Vect v(2);
    v.Data = {1., 2.};

    Vect w(2);
    w = v+2*v;


//    const int N = 1;
//    Vect x0(2*N);
//    for (int i = 0; i < 2*N; ++i) {
//        if (i%2 == 0) {
//            x0.Data[i] = {1., 0.};
//        } else {
//            x0.Data[i] = {0., 0.};
//        }
//    }
//
//    ElasticChain1D chain(N, 0.1, 6.67);
//    auto chainDx = chain.Dx();
//    auto myFunc = [&chain, &chainDx](Vect& dx, double t, const Vect& x) {
//        dx = chainDx.VectMult(x);
//    };
//
//    IVP ballProblem(0., x0, myFunc);
//    RK4 solver(ballProblem, 1e-2);
//
//    auto results = SolveIVP(ballProblem, solver, 0.1, 28. * M_PI);
//    ToCSV("data/data.csv", results);
//
//
//    XYZModel xyz(2, 0.1, 1.);
//
//    auto H = xyz.H(false);
//    H.ToDense().Print();

    return 0;
}
