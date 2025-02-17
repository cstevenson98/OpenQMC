//
// Created by Conor Stevenson on 03/04/2022.
//

#ifndef MAIN_INTEGRATOR_CUH
#define MAIN_INTEGRATOR_CUH

#include <cassert>
#include <functional>
#include <utility>

#include "la/Vect.cuh"

using namespace std;

struct IVP {
  double t0;
  Vect y0;
  function<void(Vect &dy, double t, const Vect &y)> Func;

  IVP(double t0, Vect y0, function<void(Vect &dy, double t, const Vect &y)> f)
      : t0(t0), y0(std::move(y0)), Func(std::move(f)) {};
};

struct State {
  double T;
  Vect Y;
  double E;

  State(double t, Vect y, double e) : T(t), Y(std::move(y)), E(e) {};
};

class Integrator {
public:
  virtual void State(State &dst) = 0;
  virtual double Step(double step) = 0;
};

vector<State> SolveIVP(Vect &y0, double T0, Integrator &solver, double stepsize,
                       double tEnd);

#endif // MAIN_INTEGRATOR_CUH
