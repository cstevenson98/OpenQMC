//
// Copyright (C) 2025 Conor Stevenson
// Licensed under the GNU General Public License v3.0
// Created by conor on 09/04/2022.
//

#ifndef MAIN_CSV_CUH
#define MAIN_CSV_CUH

#include <string>
#include <vector>

#include "ode/Integrator.cuh"

void ToCSV(const std::string& filename, const std::vector<State>& results);

#endif //MAIN_CSV_CUH
