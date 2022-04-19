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

#define N 2

using t_hostVect = thrust::host_vector<thrust::complex<double>>;

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


    auto Sm2 = Kronecker(Identity(2), SigmaMinus());
    Sm2.ToDense().PrintRe(0);

    auto A = 2*Kronecker(Sm2, Sm2);
    auto B = Kronecker(Sm2.HermitianC() * Sm2, Identity(4));
    auto C = Kronecker(Identity(4), Sm2.HermitianC() * Sm2);

    (B+C).ToDense().PrintRe(0);
    // Try to build lindblad operator manually

//    auto sX = Lindblad(Kronecker(Identity(2), SigmaMinus()));
//    sX.ToDense().PrintRe(0);
//

//
//    XYZModel spinChain(N, 1., 0., 0.);
//    auto H = spinChain.H(true);
//    H.ToDense().PrintRe(0);
//
//    auto L = spinChain.Dx(true);
//    L.ToDense().Print(0);

//    (ToSuper(SigmaMinus(), SigmaPlus()).Add
//    (ToSuper(SigmaPlus()*SigmaMinus(), Identity(2)))).ToDense().PrintRe();

//    (SigmaMinus() - SigmaPlus()).ToDense().PrintRe();

//    Euler_GPU test(D_Yvals, 0, chainDxELL, 1e-1);
//    auto res = SolveIVPGPU(0, D_Yvals, test, 0.01, 10., true);
//    ToCSV("/home/conor/dev/OpenQMC/data/data.csv", res);

    return 0;
}
