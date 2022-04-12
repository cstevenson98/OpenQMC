//
// Created by Conor Stevenson on 03/04/2022.
//

#include <iostream>
#include "cuComplex.h"
#include "qm/Spins.h"
#include "la/SparseELL.h"
#include "ode/Integrator.h"
#include "ode/GPU.cuh"
#include "ode/Euler_GPU.cuh"
#include "models/ElasticChain1D.h"
#include "utils/CSV.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>

#define N 100

vector<State> SolveIVPGPU(double t0,
                          thrust::host_vector<thrust::complex<double> >& y0,
                          Integrator& solver,
                          double stepsize,
                          double tEnd,
                          bool save) {

    unsigned int nx = y0.size();
    Vect         y(nx);

    double t = t0;

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
    thrust::host_vector<thrust::complex<double> > D_Yvals(2*N, 1);

    ElasticChain1D chain(N, 1., 1.);
    auto chainDx = ToSparseELL(chain.Dx());
    Euler_GPU test(D_Yvals, 0, chainDx, 1e-1);

    auto res = SolveIVPGPU(0, D_Yvals, test, 0.1, 1., true);

    ToCSV("/home/conor/dev/OpenQMC/data/data.csv", res);
    return 0;
}
