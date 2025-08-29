#pragma once
#include <nlohmann/json.hpp>
#include "core/types.cuh"
#include "simulation/rkf_parameters.cuh"
#include "simulation/environment/constants.cuh"
#include "simulation/environment/ephemeris.cuh"
#include "simulation/environment/constants.cuh"

using ActiveBodies = HostIntegerArray;

struct PropagationState
{
    HostStatesMatrix states;
    HostFloatArray epochs;
    HostBoolArray terminated;
    HostFloatArray last_dts;
    HostFloatArray next_dts;
    HostBoolArray simulation_ended;
    HostBoolArray backwards;
};

struct SamplesData
{
    std::size_t n_vecs;
    Integer center_of_integration;
    Float duration_in_days;
    HostFloatArray end_epochs;
    HostFloatArray start_epochs;
};

struct ExpectedPropagationState
{
    std::vector<Float> states_data;
    std::vector<Float> epochs;

    static ExpectedPropagationState from_json(const nlohmann::json &json);
};

struct Tolerances
{
    Float position;
    Float velocity;
    Float time;

    static Tolerances from_json(const nlohmann::json &json);
};

struct Simulation
{
    PropagationState propagation_state;

    SamplesData samples_data;
    Ephemeris ephemeris;
    RKFParameters rkf_parameters;

    Constants constants{};
    ActiveBodies active_bodies{celestial_constants::BODY_IDS};

    ExpectedPropagationState expected_propagation_state;
    Tolerances tolerances;

    bool propagated = false;

    Simulation(
        PropagationState &&ps,
        SamplesData &&sd,
        ExpectedPropagationState &&eps,
        Tolerances &&t,
        Ephemeris &&e,
        RKFParameters &&rp) : propagation_state(std::move(ps)),
                              samples_data(std::move(sd)),
                              expected_propagation_state(std::move(eps)),
                              tolerances(std::move(t)),
                              ephemeris(std::move(e)),
                              rkf_parameters(std::move(rp))
    {
    }

    static Simulation from_json(const nlohmann::json &json);

    inline std::size_t n_samples() const { return propagation_state.states.n_vecs(); }
};
