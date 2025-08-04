#pragma once
#include <nlohmann/json.hpp>
#include "core/types.hpp"
#include "simulation/propagation_context.hpp"
#include "simulation/environment/ephemeris.hpp"
#include "simulation/environment/constants.hpp"
#include "simulation/rkf_parameters.cuh"

struct Simulation
{
    PropagationContext propagation_context;
    Ephemeris ephemeris;
    const RKFParameters rkf_parameters;
    const Constants constants{};

    Simulation(
        PropagationContext &&pc,
        Ephemeris &&e,
        RKFParameters &&rp) : propagation_context(std::move(pc)),
                              ephemeris(std::move(e)),
                              rkf_parameters(std::move(rp)) {}

    static Simulation from_json(const nlohmann::json &json);

    inline std::size_t n_samples() const { return propagation_context.samples_data.n_vecs; }
};
