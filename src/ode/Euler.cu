//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by conor on 04/04/2022.
//

#include "ode/Euler.cuh"
#include <algorithm>
#include <iostream>

void Euler::State(struct State &dst) {
  dst.T = t;
  dst.Y = x;
  dst.E = Err;
}

double Euler::Step(double step) {
  Func(dx, t, x);

  Vect y0 = x + step * dx;
  Vect y1(y0.size());

  double nextStepSize = step;

  int iterations = 0;
  int maxDepth = 100;
  double err1;
  while (true) {
    if (iterations > maxDepth) {
      std::cout << "max depth reached, aborting" << std::endl;
    }
    iterations++;

    y1 = x + 0.5 * nextStepSize * dx;
    err1 = (y1 - y0).Norm();

    nextStepSize =
        0.9 * nextStepSize *
        std::min(std::max(sqrt(Tol / (2 * sqrt(err1 * err1))), 0.3), 2.);

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
