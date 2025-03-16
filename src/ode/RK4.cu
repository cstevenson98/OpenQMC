//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by conor on 04/04/2022.
//

#include "ode/RK4.cuh"
#include <iostream>

void RK4::State(struct State &dst) {
  dst.T = t;
  dst.Y = x;
  dst.E = Err;
}

// Source: https://mathworld.wolfram.com/Runge-KuttaMethod.html
double RK4::Step(double step) {
  Func(k1, t, x);
  k1 = step * k1;

  Func(k2, t + step / 2., x + 0.5 * k1);
  k2 = step * k2;

                  Vect y0(x.size());
  Vect y1(y0.size());

  y0 = x + k2;

  double nextStepSize = step;

  int iterations = 0;
  int maxDepth = 100;
  double err1;
  while (true) {
    if (iterations > maxDepth) {
      std::cout << "max depth reached, aborting" << std::endl;
      assert(false);
    }
    iterations++;

    Func(k2, t + nextStepSize / 2., x + 0.5 * k1);
    // or, with arithmetic operators:
    Func(k2, t + nextStepSize / 2., x + 0.5 * k1);
    k2 = nextStepSize * k2;

    Func(k3, t + nextStepSize / 2., x + 0.5 * k2);
    k3 = nextStepSize * k3;

    Func(k4, t + nextStepSize, x + k3);
    k4 = nextStepSize * k4;

    for (int i = 0; i < y1.size(); ++i) {
      // Need to convert all this to be directly using host part of VectImpl
      // class y1.GetDataRef()[i] = x.GetDataRef()[i] + 1./6.*k1.GetDataRef()[i]
      // + 1./3.*k2.GetDataRef()[i]
      //                        + 1./3.*k3.GetDataRef()[i]
      //                        + 1./6.*k4.GetDataRef()[i];
    }

    err1 = (y1 - y0).Norm();

    nextStepSize =
        0.9 * nextStepSize *
        std::min(std::max(std::sqrt(Tol / (2 * std::sqrt(err1 * err1))), 0.3),
                 2.);

    if (err1 >= Tol) {
      y0 = y1;
      continue;
    } else {
      break;
    }
  }

  x = y1;
  Err = err1;
  t += nextStepSize;

  return nextStepSize;
}
