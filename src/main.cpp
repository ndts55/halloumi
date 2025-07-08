#include <iostream>
#include <optional>
#include "ephemeris.hpp"
#include "samples.hpp"
#include "utils.hpp"

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

int main(int argc, char const *argv[])
{
    auto configuration = json_from_file("acceptance/acceptance.test.5-days.json");
    // read in ephemeris input data
    auto ephemeris = read_ephemeris(configuration);

    std::cout << "Read ephemeris\n"
              << ephemeris.n_bodies() << std::endl;

    // read in sample input data
    auto samples = read_samples(configuration);

    std::cout << "Read samples\n"
              << samples.size() << std::endl;

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