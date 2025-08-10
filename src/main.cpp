#include <iostream>
#include <optional>
#include <cuda_runtime.h>
#include "utils.hpp"
#include "simulation/simulation.hpp"
#include "simulation/propagation/propagate.cuh"

// TODO take config file as command line argument
// TODO get output data from device
// TODO write output data to some output file for easy comparison
// TODO test register count for unrolled / not unrolled for loop

int main()
{
    auto configuration = json_from_file("acceptance/acceptance.test.5-days.json");
    auto simulation = Simulation::from_json(configuration);

    // Launch test kernel to verify device memory setup
    std::cout << "Propagating" << std::endl;
    propagate(simulation);
    std::cout << "Propagated" << std::endl;

    return 0;
}
