//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by Conor Stevenson on 04/04/2022.
//

// #include "la/Super.cuh"
// #include "models/ElasticChain1D.cuh"
// #include "ode/Euler_GPU.cuh"
// #include "ode/GPU.cuh"
// #include "ode/Integrator.cuh"
// #include "qm/Spins.cuh"
// #include "utils/CSV.cuh"

#include <complex>
#include <cstdio>
#include <iostream>

#include "la/Dense.h"
#include "la/Sparse.h"
#include "la/SparseELL.h"
#include "la/Vect.h"

int main() {  //
  using namespace std::complex_literals;
  auto d = t_hostMat{{1., 2. - 3.i}, {3., 4. + 1.i}};

  std::cout << std::endl;
  std::cout << std::endl;

  Dense dense(d);
  dense.Print();

  auto d2 = 222 * dense * dense;
  d2.Print();

  auto ell = SparseELL();

  Sparse sparse(d);

  t_hostMat dense2 = sparse.GetHostData();

  // auto d2 = dense + dense;

  //  std::cout << d2[0, 0] << std::endl;
  //  (int i = 0; i < 2 * N; ++i) {
  //   D_Yvals[i] = (sin(10. * (i + 1) / ((double)(N))) / 2.) *
  //                sin(10. * (i + 1) / ((double)(N))) / 2. /
  //                (double)((i + 1) /
  // 200.);                        //
  // }
  // auto chainDx = chain.Dx();
  // auto chainDxELL = ToSparseELL(chainDx);

  // auto Sm2 = Kronecker(Identity(2), SigmaMinus());
  // Sm2.ToDense().PrintRe(0);

  // auto A = 2. * Kronecker(Sm2, Sm2);
  // auto B = Kronecker(Sm2.HermitianC() * Sm2, Identity(4));
  // auto C = Kronecker(Identity(4), Sm2.HermitianC() * Sm2);

  // (B + C).ToDense().PrintRe(0);
  // // Try to build lindblad operator manually

  // //    auto sX = Lindblad(Kronecker(Identity(2), SigmaMinus()));
  // //    sX.ToDense().PrintRe(0);
  // //

  // //
  // //    XYZModel spinChain(N, 1., 0., 0.);
  // //    auto H = spinChain.H(true);
  // //    H.ToDense().PrintRe(0);
  // //
  // //    auto L = spinChain.Dx(true);
  // //    L.ToDense().Print(0);

  // //    (ToSuper(SigmaMinus(), SigmaPlus()).Add
  // //    (ToSuper(SigmaPlus()*SigmaMinus(),
  // Identity(2)))).ToDense().PrintRe();

  // //    (SigmaMinus() - SigmaPlus()).ToDense().PrintRe();

  // Euler_GPU test(D_Yvals, 0, chainDxELL, 1e-1);
  // Vect y0(2 * N);
  // auto res = SolveIVP(y0, 0., test, 0.01, 10000.);
  // ToCSV("/home/conor/dev/OpenQMC/data/data.csv", res);

  return 0;
}
