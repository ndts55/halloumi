#include <iostream>
#include <nlohmann/json.hpp>
#include "core/types.cuh"
#include "simulation/simulation.cuh"
#include "utils.cuh"
#include "simulation/environment/ephemeris.cuh"

Simulation Simulation::from_json(const nlohmann::json &configuration)
{

    auto samples_json = json_from_file(configuration["samples"]);
    auto in_samples = samples_json.contains("inSamples") ? samples_json["inSamples"] : samples_json.contains("samples") ? samples_json["samples"] : nlohmann::json::array();
    auto n_vecs = in_samples.size();

    // Mutable State
    HostStatesMatrix states(n_vecs);
    HostFloatArray epochs(n_vecs);
    HostBoolArray terminated(n_vecs, false);

    // Immutable state that has to be calculated
    double duration_in_days = configuration["simConfig"]["durationDays"];
    HostFloatArray start_epochs(n_vecs);
    HostFloatArray end_epochs(n_vecs);

    // Immutable expected state
    std::vector<StateVector> expected_states;
    expected_states.reserve(n_vecs);
    std::vector<double> expected_epochs;
    expected_epochs.reserve(n_vecs);

    std::size_t idx = 0;
    for (auto sample : in_samples)
    {
        double e = sample["epoch"];
        epochs.at(idx) = e;
        start_epochs.at(idx) = e;
        end_epochs.at(idx) = e + duration_in_days;
        nlohmann::json cart = sample["stateCart"];
        for (auto dim = 0; dim < STATE_DIM; dim++)
        {
            // put each element of vec<6> in the correct position
            states.at(idx, dim) = cart[dim];
        }
        idx += 1;
    }

    auto ps = PropagationState{
        std::move(states),
        std::move(epochs),
        std::move(terminated),
        /* last_dts */ HostFloatArray(n_vecs, 0.0f),
        /* next_dts */ HostFloatArray(n_vecs, 0.0f),
        /* simulation_ended */ HostBoolArray(n_vecs, false),
        /* backwards */ HostBoolArray(n_vecs, false)};
    auto sd = SamplesData{
        n_vecs,
        /* center_of_integration */ samples_json["centre"],
        duration_in_days,
        std::move(end_epochs),
        std::move(start_epochs),
    };

    auto exec_config = json_from_file(configuration["simConfig"]["config"]);
    auto environment_json = exec_config["model"]["environment"];
    return Simulation(
        std::move(ps),
        std::move(sd),
        ExpectedPropagationState::from_json(samples_json),
        Tolerances::from_json(configuration),
        Ephemeris::from_brie(environment_json["ephemeris"]),
        RKFParameters::from_json(exec_config["integration"]));
}

ExpectedPropagationState ExpectedPropagationState::from_json(const nlohmann::json &json)
{
    if (!json.contains("outSamples")) {
        return ExpectedPropagationState{};
    }
    const auto out_samples = json["outSamples"];
    const auto n_vecs = out_samples.size();
    std::vector<double> states(n_vecs * STATE_DIM, 0.0);
    std::vector<double> epochs;
    epochs.reserve(n_vecs);
    auto index = 0;
    for (const auto &sample : out_samples)
    {
        epochs.push_back(sample["epoch"]);
        const auto state = sample["stateCart"];
        for (auto dim = 0; dim < STATE_DIM; ++dim)
        {
            states.at(get_2d_index(n_vecs, index, dim)) = state[dim];
        }
        index += 1;
    }
    return ExpectedPropagationState{
        std::move(states),
        std::move(epochs),
    };
}

Tolerances Tolerances::from_json(const nlohmann::json &configuration)
{
    Tolerances t;
    const auto abstol_json = configuration["validation"]["absTol"];
    t.position = abstol_json["pos"];
    t.velocity = abstol_json["vel"];
    t.time = abstol_json["time"];
    return t;
}
