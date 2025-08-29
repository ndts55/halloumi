#include <iostream>
#include <optional>
#include <cuda_runtime.h>
#include <cmath>
#include "utils.hpp"
#include "simulation/simulation.hpp"
#include "simulation/propagation/propagate.cuh"
#include "cuda/vec.cuh"
#include "core/types.cuh"

// TODO take config file as command line argument
// ? write output data to some output file for easy comparison
// TODO test register count for unrolled / not unrolled for loop

struct Errors
{
    Float position_error;
    Float velocity_error;
    Float epoch_error;

    static Errors from_simulation(const Simulation &simulation)
    {
        std::vector<Float> state_error_components(simulation.n_samples() * STATE_DIM);
        std::transform(
            simulation.propagation_state.states.begin(),
            simulation.propagation_state.states.end(),
            simulation.expected_propagation_state.states_data.begin(),
            state_error_components.begin(),
            [](const Float &a, const Float &b)
            {
                return std::pow(a - b, 2.0);
            });
        std::vector<Float> position_error_components(simulation.n_samples(), 0.0);
        for (auto vidx = 0; vidx < simulation.n_samples(); ++vidx)
        {
            for (auto dim = POSITION_OFFSET; dim < POSITION_OFFSET + POSITION_DIM; ++dim)
            {
                position_error_components[vidx] += state_error_components[get_2d_index_(simulation.n_samples(), vidx, dim)];
            }
            position_error_components[vidx] = std::sqrt(position_error_components[vidx]);
        }
        Float position_error = *std::max_element(position_error_components.begin(), position_error_components.end());

        std::vector<Float> velocity_error_components(simulation.n_samples());
        for (auto vidx = 0; vidx < simulation.n_samples(); ++vidx)
        {
            for (auto dim = VELOCITY_OFFSET; dim < VELOCITY_OFFSET + VELOCITY_DIM; ++dim)
            {
                velocity_error_components[vidx] += state_error_components[get_2d_index_(simulation.n_samples(), vidx, dim)];
            }
            velocity_error_components[vidx] = std::sqrt(velocity_error_components[vidx]);
        }
        Float velocity_error = *std::max_element(velocity_error_components.begin(), velocity_error_components.end());

#ifndef NDEBUG
        print_vector_mean(position_error_components, "Position error components");
        print_failed_count(position_error_components, "Position error components", simulation.tolerances.position);
        print_vector_mean(velocity_error_components, "Velocity error components");
        print_failed_count(velocity_error_components, "Velocity error components", simulation.tolerances.velocity);
#endif

        std::vector<Float> epoch_error_components(simulation.n_samples());
        std::transform(
            simulation.propagation_state.epochs.begin(),
            simulation.propagation_state.epochs.end(),
            simulation.expected_propagation_state.epochs.begin(),
            epoch_error_components.begin(),
            [](const Float &a, const Float &b)
            {
                return std::abs(a - b);
            });
        Float epoch_error = *std::max_element(epoch_error_components.begin(), epoch_error_components.end());

        return Errors{position_error, velocity_error, epoch_error};
    }
};

bool validate(const Simulation &simulation)
{
    const auto errors = Errors::from_simulation(simulation);
    const auto pos_passed = errors.position_error <= simulation.tolerances.position;
    const auto vel_passed = errors.velocity_error <= simulation.tolerances.velocity;
    const auto epoch_passed = errors.epoch_error <= simulation.tolerances.time;
    if (!pos_passed)
    {
        std::cerr << "Position error: " << errors.position_error << " exceeds tolerance: " << simulation.tolerances.position << std::endl;
    }
    else
    {
        std::cout << "Position error: " << errors.position_error << " within tolerance: " << simulation.tolerances.position << std::endl;
    }
    if (!vel_passed)
    {
        std::cerr << "Velocity error: " << errors.velocity_error << " exceeds tolerance: " << simulation.tolerances.velocity << std::endl;
    }
    else
    {
        std::cout << "Velocity error: " << errors.velocity_error << " within tolerance: " << simulation.tolerances.velocity << std::endl;
    }
    if (!epoch_passed)
    {
        std::cerr << "Epoch error: " << errors.epoch_error << " exceeds tolerance: " << simulation.tolerances.time << std::endl;
    }
    else
    {
        std::cout << "Epoch error: " << errors.epoch_error << " within tolerance: " << simulation.tolerances.time << std::endl;
    }
    return pos_passed && vel_passed && epoch_passed;
}

int main()
{
    const std::string file = "acceptance/acceptance.test.5-days.json";
    std::cout << "Loading configuration from " << file << std::endl;
    auto configuration = json_from_file(file);
    auto simulation = Simulation::from_json(configuration);
    std::cout << "Loaded simulation with " << simulation.n_samples() << " samples." << std::endl;

    std::cout << "Propagating" << std::endl;
    propagate(simulation);
    if (simulation.propagated)
    {
        std::cout << "Propagated" << std::endl;
    }
    else
    {
        std::cerr << "Propagation failed!" << std::endl;
        return 1;
    }

    if (validate(simulation))
    {
        std::cout << "Validation successful!" << std::endl;
    }
    else
    {
        std::cerr << "Validation failed!" << std::endl;
        return 1;
    }

    return 0;
}
