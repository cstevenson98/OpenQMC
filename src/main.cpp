//
// Created by Conor Stevenson on 03/04/2022.
//

#include <vector>
#include <cstdlib>
#include <iostream>

#include <boost/compute/event.hpp>
#include <boost/compute/system.hpp>
#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/async/future.hpp>
#include <boost/compute/container/vector.hpp>

namespace compute = boost::compute;

int main() {
    // get the default device
    compute::device device = compute::system::default_device();

    // print the device's name and platform
    std::cout << "hello from " << device.name();

    return 0;
}
