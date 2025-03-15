//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by conor on 04/04/2022.
//

#ifndef MAIN_EULER_CUH
#define MAIN_EULER_CUH

#include <functional>
#include "ode/Integrator.cuh"

class Euler : public Integrator {
    Vect x, dx;
    double t;
    std::function<void(Vect&, double, const Vect&)> Func;
    double Err, Tol;
public:
    Euler(const IVP &ivp, double tol) : Tol(tol) {
        x = ivp.y0;
        dx = Vect(x.size());
        t = ivp.t0;
        Func = ivp.Func;
        Err = 0;
    };

    void State(struct State &dst) override;
    double Step(double step) override;

};

#endif //MAIN_EULER_CUH
