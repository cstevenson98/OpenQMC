//
// Created by conor on 04/04/2022.
//

#include <iostream>
#include "RK4.cuh"

void RK4::State(struct State &dst) {
    dst.T = t;
    dst.Y = x;
    dst.E = Err;
}

// Source: https://mathworld.wolfram.com/Runge-KuttaMethod.html
double RK4::Step(double step) {
    Func(k1, t, x);
    k1 = k1.Scale(step);

    Func(k2, t + step/2., x.AddScaledVect(0.5, k1));
    k2 = k2.Scale(step);

    Vect y0(x.Data.size());
    Vect y1(y0.Data.size());

    y0 = x.Add(k2);

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

        Func(k2, t + nextStepSize/2., x.AddScaledVect(0.5, k1));
        k2 = k2.Scale(nextStepSize);

        Func(k3, t + nextStepSize/2., x.AddScaledVect(0.5, k2));
        k3 = k3.Scale(nextStepSize);

        Func(k4, t + nextStepSize, x.Add(k3));
        k4 = k4.Scale(nextStepSize);

        for (int i = 0; i < y1.Data.size(); ++i) {
            y1.Data[i] = x.Data[i] + 1./6.*k1.Data[i] + 1./3.*k2.Data[i]
                                   + 1./3.*k3.Data[i] + 1./6.*k4.Data[i];
        }

        err1 = y1.Subtract(y0).Norm();

        nextStepSize = 0.9 * nextStepSize * min(max(sqrt(Tol/(2*sqrt(err1*err1))), 0.3), 2.);

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
