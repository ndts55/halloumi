#pragma once
#include <nlohmann/json.hpp>
#include "core/types.cuh"
#include "simulation/environment/ephemeris.cuh"
#include "simulation/environment/constants.cuh"
#include "simulation/rkf_parameters.cuh"
#include "cuda/cuda_array.hpp"
#include "simulation/environment/constants.cuh"

using ActiveBodies = CudaArray1D<Integer>;

struct PropagationState
{
    CudaArray2D<Float, STATE_DIM> states;
    CudaArray1D<Float> epochs;
    CudaArray1D<bool> terminated;
    CudaArray1D<Float> last_dts;
    CudaArray1D<Float> next_dts;
    CudaArray1D<bool> simulation_ended;
    CudaArray1D<bool> backwards;
};

struct SamplesData
{
    std::size_t n_vecs;
    Integer center_of_integration;
    Float duration_in_days;
    CudaArray1D<Float> end_epochs;
    CudaArray1D<Float> start_epochs;
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

    const SamplesData samples_data;
    const Ephemeris ephemeris;
    const RKFParameters rkf_parameters;

    const Constants constants{};
    const ActiveBodies active_bodies{celestial_constants::BODY_IDS};

    const ExpectedPropagationState expected_propagation_state;
    const Tolerances tolerances;

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
