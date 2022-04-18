//
// Created by conor on 09/04/2022.
//

#include <cstdlib>
#include <string>
#include <iostream>

#include "models/ElasticChain1D.cuh"
#include "ode/RK4.cuh"
#include "utils/CSV.cuh"
#include "la/Vect.cuh"

using namespace std;

const char *const ELASTIC_N = "ELASTIC_N";         // required!
const char *const ELASTIC_KAPPA = "ELASTIC_KAPPA"; // required!
const char *const ELASTIC_DELTA = "ELASTIC_DELTA"; // required!
const char *const OUT_PATH = "OUT_PATH";           // required!

int main () {

//    // Parameter: N, chain length
//    const char *N_char = getenv(ELASTIC_N);
//    int N = stoi(N_char ? N_char : "");
//
//    // Parameter: kappa, damping rate
//    const char *kappa_char = getenv(ELASTIC_KAPPA);
//    double kappa = stof(kappa_char ? kappa_char : "");
//
//    // Parameter: delta, coupling strength
//    const char *delta_char = getenv(ELASTIC_DELTA);
//    double delta = stof(delta_char ? delta_char : "");
//
//    // Where to save data
//    const char *out_path = getenv(OUT_PATH);
//
//    cout << "N = " << N << ", "
//         << "kappa = " << kappa << ", "
//         << "delta = " << delta << ", "
//         << "path = " << out_path << endl;
//
//    ElasticChain1D chain(N, kappa, delta);
//
//    cout << "Initialising chain...\n";
//    auto chainDx = chain.Dx();
//    auto myFunc = [&chain, &chainDx](Vect& dx, double t, const Vect& x) {
//        dx = chainDx.VectMult(x);
//    };
//
//    Vect x0(2*N);
//    for (int i = 0; i < 2*N; ++i) {
//        if (i%2 == 0) {
//            x0.Data[i] = {1., 0.};
//        } else {
//            x0.Data[i] = {0., 0.};
//        }
//    }
//
//    IVP ballProblem(0., x0, myFunc);
//    RK4 solver(ballProblem, 1e-2);
//
//    cout << "Solving ODE...\n";
//    auto results = SolveIVP(ballProblem, solver, 0.1, 28. * M_PI);
//    ToCSV(out_path, results);
//
//    cout << "DONE.\n";
//    return 0;
}