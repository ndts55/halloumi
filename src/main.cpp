#include <iostream>
#include <optional>
#include <cuda_runtime.h>
#include "utils.hpp"
#include "simulation/simulation.hpp"
#include "simulation/propagate.cuh"

// TODO take config file as command line argument
// TODO ensure simulation is stopped if number of steps exceed given value
// TODO get output data from device
// TODO write output data to some output file for easy comparison
// TODO implement mathematical model
// TODO implement physical model
// TODO implement ode evaluation stages as for loop
// TODO test register count for unrolled / not unrolled for loop

int main()
{
    auto configuration = json_from_file("acceptance/acceptance.test.5-days.json");
    auto simulation = Simulation::from_json(configuration);

    std::cout << "Read environment\n"
              << "Ephemeris Bodies: "
              << simulation.ephemeris.n_bodies() << "\n"
              << "Constants Bodies: "
              << simulation.constants.n_bodies() << "\n"
              << std::endl;
    std::cout << "Read propagation context\n"
              << "States: "
              << simulation.propagation_context.samples_data.n_vecs << std::endl;
    std::cout << "Read RKF parameters\n"
              << "> abs tol: " << simulation.rkf_parameters.abs_tol << "\n"
              << "> rel tol: " << simulation.rkf_parameters.rel_tol << "\n"
              << "> initial time step: " << simulation.rkf_parameters.initial_time_step << "\n"
              << "> min time step: " << simulation.rkf_parameters.min_time_step << "\n"
              << "> max steps: " << simulation.rkf_parameters.max_steps << std::endl;

    // Launch test kernel to verify device memory setup
    std::cout << "Propagating" << std::endl;
    propagate(simulation);
    std::cout << "Propagated" << std::endl;

    return 0;
}
