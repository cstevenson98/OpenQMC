//
// Created by conor on 09/04/2022.
//

#include <iostream>
#include <fstream>
#include "CSV.h"
#include "../ode/Integrator.h"

void ToCSV(const std::string& filename, const vector<State>& results) {
    std::ofstream myfile;
    myfile.open(filename);
    for (const auto& r : results) {
        myfile << r.T << " " ;
        for (const auto elem : r.Y.Data) {
            myfile << elem.real() << " ";
        }
        myfile << r.E << std::endl;
    }
    myfile.close();
}
