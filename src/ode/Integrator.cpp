//
// Created by Conor Stevenson on 03/04/2022.
//

#include "Integrator.h"

vector<State> SolveIVP(const IVP& ivp, Integrator& solver, double stepsize, double tEnd) {
    double t0 = ivp.t0;
    Vect   x0 = ivp.y0;

    unsigned int nx = x0.Data.size();
    Vect         y(nx);

    double t = t0;

    vector<State> results;
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
