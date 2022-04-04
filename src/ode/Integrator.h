//
// Created by Conor Stevenson on 03/04/2022.
//

#ifndef MAIN_INTEGRATOR_H
#define MAIN_INTEGRATOR_H

#include <utility>
#include <functional>
#include <cassert>

#include "../la/Vect.h"

using namespace std;

struct IVP {
    double t0;
    Vect y0;
    function<void(Vect& dy, double t, const Vect& y)> Func;

    IVP(double t0, Vect y0, function<void(Vect& dy, double t, const Vect& y)> f)
    : t0(t0), y0(move(y0)), Func(move(f)) { };
};

struct State {
    double T;
    Vect Y;
    double E;

    State(double t, Vect y, double e) : T(t), Y(move(y)), E(e) { };
};

class Integrator {
public:
    virtual void State(State& dst) = 0;
    virtual double Step(double step) = 0;
};

vector<State> SolveIVP(const IVP& ivp, Integrator& solver, double stepsize, double tEnd);

#endif //MAIN_INTEGRATOR_H
