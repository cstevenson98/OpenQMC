//
// Created by Conor Stevenson on 03/04/2022.
//

#include <iostream>
#include "cuComplex.h"
#include "qm/Spins.cuh"
#include "la/SparseELL.cuh"
#include "ode/Integrator.cuh"
#include "ode/GPU.cuh"
#include "ode/Euler_GPU.cuh"
#include "ode/RK4.cuh"
#include "models/ElasticChain1D.cuh"
#include "utils/CSV.cuh"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>

#define N 3

vector<State> SolveIVPGPU(double t0,
                          thrust::host_vector<thrust::complex<double> >& y0,
                          Integrator& solver,
                          double stepsize,
                          double tEnd,
                          bool save) 
{
    unsigned int nx = y0.size();
    Vect         y(nx);
    double       t = t0;

    vector<State> results;
    while (t < tEnd) {
        State res(t, y, 0.);

        if (t-tEnd > 1e-10){
            stepsize = min(stepsize, (t-tEnd)*(1+1e-3));
        }

        stepsize = solver.Step(stepsize);
        assert(stepsize >= 0);

        if (save) {
            solver.State(res);
            results.emplace_back(res);
        }

        t += stepsize;
    }

    return results;
}

int main()
{
    thrust::host_vector<thrust::complex<double>> D_Yvals(2*N, 0);
    Vect y0(2*N);

    for (int i = 0; i < D_Yvals.size(); ++i) {
        if (i%2==0) {
            D_Yvals[i] = 1;
        } else {
            D_Yvals[i] = 0;
        }
    }
    y0.Data = {8, -1, 2, 0, -7, 2};

    ElasticChain1D chain(N, 1, 1, 1);

    auto chainDx = chain.Dx();
    auto chainDxELL = ToSparseELL(chainDx);

    chain.Dx().ToDense().PrintRe();
    chainDxELL.Values.PrintRe();
    chainDxELL.Indices.PrintRe();

    chainDxELL.VectMult(y0).PrintRe();

    auto func = [chainDx](Vect& dy, double t, const Vect& y) {
        dy = chainDx.VectMult(y);
    };

    RK4 testCPU(y0, 0, func, 0.1);
    Euler_GPU test(D_Yvals, 0, chainDxELL, 1e-1);

//     auto res = SolveIVPGPU(0, D_Yvals, test, 0.01, 1., true);
//
////    auto res = SolveIVP(y0, 0., testCPU, 0.1, 40.);
//
//    ToCSV("/home/conor/dev/OpenQMC/data/data.csv", res);
    return 0;
}
