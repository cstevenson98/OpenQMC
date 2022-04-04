//
// Created by conor on 04/04/2022.
//

#ifndef MAIN_RK4_H
#define MAIN_RK4_H


#include "Integrator.h"

class RK4 : public Integrator {
    Vect x, k1, k2, k3, k4;
    double t;
    std::function<void(Vect&, double, const Vect&)> Func;
    double Err, Tol;

public:
    RK4(const IVP &ivp, double tol) : Tol(tol) {
        const unsigned int N = ivp.y0.Data.size();
        x = ivp.y0;
        k1 = Vect(N); k2 = Vect(N);
        k3 = Vect(N); k4 = Vect(N);
        t = ivp.t0;
        Func = ivp.Func;
        Err = 0;
    };

    void State(struct State &dst) override;

    double Step(double step) override;
};


#endif //MAIN_RK4_H
