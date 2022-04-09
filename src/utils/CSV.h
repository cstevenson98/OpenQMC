//
// Created by conor on 09/04/2022.
//

#ifndef MAIN_CSV_H
#define MAIN_CSV_H

#include <string>
#include <vector>

#include "../ode/Integrator.h"
using namespace std;

void ToCSV(const string& filename, const vector<State>& results);

#endif //MAIN_CSV_H
