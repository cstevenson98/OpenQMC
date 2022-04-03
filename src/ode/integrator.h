//
// Created by Conor Stevenson on 03/04/2022.
//

#ifndef MAIN_INTEGRATOR_H
#define MAIN_INTEGRATOR_H

#include <utility>

#include "../la/vect.h"

struct IVP {
    double t0;
    vect y0;
    function<void(vect& dy, double t, const vect& y)> Func;

    IVP(double t0, vect y0, function<void(vect& dy, double t, const vect& y)> f)
    : t0(t0), y0(std::move(y0)), Func(std::move(f)) { };
};

struct State {
    double T;
    vect Y;
    double E;

    State( double t, vect y, double e) : T(t), Y(std::move(y)), E(e) { };
};

class integrator {
public:
    virtual void Init(IVP ivp) = 0;
    virtual void State(State& dst) = 0;
    virtual double Step(double step) = 0;
};

vector<State> SolveIVP(const IVP& ivp, integrator& solver, double stepsize, double tEnd) {
    double t0 = ivp.t0;
    vect x0 = ivp.y0;

    unsigned int nx = x0.Data.size();

    vector<State> results;

    double t = t0;
    vect y(nx);

    while (t < tEnd) {
        State res(t, y, 0.);

        if (t-tEnd > 1e-10){
            stepsize = min(stepsize, (t-tEnd)*(1+1e-3));
        }

        stepsize = solver.Step(stepsize);
        assert(stepsize >= 0);

        solver.State(res);
        t += stepsize;
        results.emplace_back(res);
    }

    return results;
}

#endif //MAIN_INTEGRATOR_H
