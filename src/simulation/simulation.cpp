#include <iostream>
#include <nlohmann/json.hpp>
#include "core/types.cuh"
#include "simulation/simulation.hpp"
#include "utils.hpp"
#include "simulation/environment/ephemeris.cuh"
#include "cuda/cuda_array.hpp"

Simulation Simulation::from_json(const nlohmann::json &configuration)
{

    auto samples_json = json_from_file(configuration["samples"]);
    auto in_samples = samples_json["inSamples"];
    auto n_vecs = in_samples.size();

    // Mutable State
    CudaArray2D<Float, STATE_DIM> states(n_vecs);
    CudaArray2D<Float, STATE_DIM> start_states(n_vecs);
    CudaArray1D<Float> epochs(n_vecs);
    CudaArray1D<bool> terminated(n_vecs, false);

    // Immutable state that has to be calculated
    Float duration_in_days = configuration["simConfig"]["durationDays"];
    CudaArray1D<Float> start_epochs(n_vecs);
    CudaArray1D<Float> end_epochs(n_vecs);

    std::size_t idx = 0;
    for (auto sample : in_samples)
    {
        Float e = sample["epoch"];
        epochs.at(idx) = e;
        start_epochs.at(idx) = e;
        end_epochs.at(idx) = e + duration_in_days; // ? is this even correct?
        auto cart = sample["stateCart"];
        for (auto dim = 0; dim < STATE_DIM; dim++)
        {
            // put each element of vec<6> in the correct position
            states.at(dim, idx) = cart[dim];
            start_states.at(dim, idx) = cart[dim];
        }
        idx += 1;
    }

    auto ps = PropagationState{
        std::move(states),
        std::move(epochs),
        std::move(terminated),
        /* last_dts */ CudaArray1D<Float>(n_vecs, 0.0f),
        /* next_dts */ CudaArray1D<Float>(n_vecs, 0.0f),
        /* simulation_ended */ CudaArray1D<bool>(n_vecs, false),
        /* backwards */ CudaArray1D<bool>(n_vecs, false)};
    auto sd = SamplesData{
        n_vecs,
        /* center_of_integration */ samples_json["centre"],
        duration_in_days,
        std::move(end_epochs),
        std::move(start_epochs),
        std::move(start_states),
    };

    auto exec_config = json_from_file(configuration["simConfig"]["config"]);
    auto environment_json = exec_config["model"]["environment"];
    return Simulation(
        std::move(ps),
        std::move(sd),
        Ephemeris::from_brie(environment_json["ephemeris"]),
        RKFParameters::from_json(exec_config["integration"]));
}