#include <iostream>
#include <optional>
#include <cuda_runtime.h>
#include "utils.hpp"
#include "simulation/simulation.hpp"
#include "simulation/propagation/propagate.cuh"
#include "cuda/vec.cuh"

// TODO take config file as command line argument
// TODO get output data from device
// TODO write output data to some output file for easy comparison
// TODO test register count for unrolled / not unrolled for loop

void print_array(const Float array[])
{
    std::cout << "[";
    for (int i = 0; i < STATE_DIM; ++i)
    {
        std::cout << array[i];
        if (i < STATE_DIM - 1)
        {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}

int main()
{
    auto configuration = json_from_file("acceptance/acceptance.test.5-days.json");
    auto simulation = Simulation::from_json(configuration);

    Float first_state_before[STATE_DIM] = {0.0};
    for (auto dim = 0; dim < STATE_DIM; ++dim)
    {
        first_state_before[dim] = simulation.propagation_state.states.at(dim, 0);
    }

    std::cout << "Propagating" << std::endl;
    propagate(simulation);
    std::cout << "Propagated" << std::endl;

    Float first_state_after[STATE_DIM] = {0.0};
    for (auto dim = 0; dim < STATE_DIM; ++dim)
    {
        first_state_after[dim] = simulation.propagation_state.states.at(dim, 0);
    }

    std::cout << "first state before: ";
    print_array(first_state_before);
    std::cout << "first state after: ";
    print_array(first_state_after);

    return 0;
}
