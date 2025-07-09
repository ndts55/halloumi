#include <iostream>
#include <optional>
#include "ephemeris.hpp"
#include "samples.hpp"
#include "utils.hpp"
#include "simulation.hpp"

Ephemeris read_ephemeris(const nlohmann::json &configuration)
{
    auto exec_config = json_from_file(configuration["simConfig"]["config"]);
    auto brie_file = exec_config["model"]["environment"]["ephemeris"];
    return Ephemeris::from_brie(brie_file);
}

Samples read_samples(const nlohmann::json &configuration)
{
    auto samples_json = json_from_file(configuration["samples"]);
    return Samples::from_json(samples_json);
}

int main()
{
    auto configuration = json_from_file("acceptance/acceptance.test.5-days.json");
    auto simulation = Simulation::from_json(configuration);

    std::cout << "Read ephemeris\n"
              << simulation.ephemeris().n_bodies() << std::endl;
    std::cout << "Read samples\n"
              << simulation.samples().size() << std::endl;

    // TODO setup the simulation for CUDA devices
    // TODO ensure coalesced access to data in global memory
    // TODO implement kernel for ode evaluation (all stages)
    // TODO implement kernel for step advancement (adaptive step size)
    // TODO implement kernel for determining whether all samples have finished
    // TODO ensure simulation is stopped if number of steps exceed given value
    // TODO run the simulation on CUDA device
    // TODO get output data from device
    // TODO write output data to some output file for easy comparison
    // TODO implement mathematical model
    // TODO implement physical model
    // TODO implement coalesced access to current states
    // TODO implement ode evaluation stages as for loop
    // TODO test register count for unrolled / not unrolled for loop
    return 0;
}