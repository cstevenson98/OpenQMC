//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by Conor Stevenson on 03/04/2022.
//

#include "ode/Integrator.cuh"

vector<State> SolveIVP(Vect& y0, double T0, Integrator& solver, double stepsize, double tEnd) {
    double t0 = T0;
    Vect   x0 = y0;

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

vector<State> SolveIVPGPU(double t0, t_hostVect& y0, Integrator& solver,
                          double stepsize, double tEnd, bool save)
{
    unsigned int nx = y0.size();
    Vect         y(nx);
    double       t = t0;

    vector<State> results;
    while (t < tEnd) {
        State res(t, y, 0.);

        if (t-tEnd > 1e-10){
            stepsize = min(stepsize, (t-tEnd)*(1+1e-3));
        }

        stepsize = solver.Step(stepsize);
        assert(stepsize >= 0);

        if (save) {
            solver.State(res);
            results.emplace_back(res);
        }

        t += stepsize;
    }

    return results;
}
