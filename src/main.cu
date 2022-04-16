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
#include "qm/Open.cuh"
#include "la/Super.cuh"
#include "models/XYZModel.cuh"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>

#define N 7

using t_hostVect = thrust::host_vector<thrust::complex<double>>;

vector<State> SolveIVPGPU(double t0, t_hostVect& y0, Integrator& solver,
                          double stepsize, double tEnd, bool save)
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
//    t_hostVect D_Yvals(2*N, 0);
//
//    for (int i = 0; i < 2*N; ++i) {
//        D_Yvals[i] = (sin(10.*(i+1)/( (double)(N) ))/2.)
//                    * sin(10.*(i+1)/( (double)(N) ))/2. / (double)((i+1)/200.);
//    }
//
//    ElasticChain1D chain(N, 10, 0.25, 2);
//
//    auto chainDx = chain.Dx();
//    auto chainDxELL = ToSparseELL(chainDx);

    auto sX = Lindblad(SigmaMinus(2, 1), SigmaPlus(2, 1));
//    sX.ToDense().PrintRe();

    auto test = (SigmaPlus(3, 0) + SigmaPlus(3, 1));

    test.ToDense().PrintRe();

    std:cout << test.NNZ() << std::endl;

    XYZModel spinChain(3, 10., 0.);
//    auto H = spinChain.H(true);
//    H.ToDense().PrintRe();

//    (ToSuper(SigmaMinus(), SigmaPlus()).Add
//    (ToSuper(SigmaPlus()*SigmaMinus(), Identity(2)))).ToDense().PrintRe();

//    (SigmaMinus() - SigmaPlus()).ToDense().PrintRe();

//    Euler_GPU test(D_Yvals, 0, chainDxELL, 1e-1);
//    auto res = SolveIVPGPU(0, D_Yvals, test, 0.01, 10., true);
//    ToCSV("/home/conor/dev/OpenQMC/data/data.csv", res);

    return 0;
}
