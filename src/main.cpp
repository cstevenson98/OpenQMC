//
// Created by Conor Stevenson on 03/04/2022.
//

#include<iostream>
#include <fstream>

#include "la/Dense.h"
#include "la/Sparse.h"
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
    const int N = 16 * 16;

    Vect x0(2);
    x0.Data = {1., 0};
    auto myFunc = [](Vect& dx, double t, const Vect& x) {
        dx.Data[0] = x.Data[1];
        dx.Data[1] = -x.Data[0];
    };


    IVP ballProblem(0., x0, myFunc);
    RK4 solver(ballProblem, 1e-6);

    auto results = SolveIVP(ballProblem, solver, 0.1, 2. * M_PI);

    ToCSV("data/data.csv", results);
    return 0;
}
