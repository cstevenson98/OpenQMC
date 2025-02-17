//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by conor on 04/04/2022.
//

#ifndef MAIN_RK4_CUH
#define MAIN_RK4_CUH

#include "ode/Integrator.cuh"

class RK4 : public Integrator {
    Vect x, k1, k2, k3, k4;
    double t;
    std::function<void(Vect&, double, const Vect&)> Func;
    double Err, Tol;

public:
    RK4(Vect& y0, double t0, function<void(Vect& dy, double t, const Vect& y)> func, double tol) 
    : Tol(tol) 
    {
        const unsigned int N = y0.Data.size();
        x = y0;
        k1 = Vect(N); k2 = Vect(N);
        k3 = Vect(N); k4 = Vect(N);
        t = t0;
        Func = func;
        Err = 0;
    };

    void State(struct State &dst) override;

    double Step(double step) override;
};


#endif //MAIN_RK4_CUH
