#pragma once
#include <nlohmann/json.hpp>
#include "core/types.hpp"
#include "simulation/propagation_context.hpp"
#include "simulation/environment/environment.hpp"
#include "simulation/rkf_parameters.hpp"

struct Simulation
{
    PropagationContext propagation_context;
    Environment environment;
    const RKFParameters rkf_parameters;

    Simulation(
        PropagationContext &&pc,
        Environment &&e,
        RKFParameters &&rp) : propagation_context(std::move(pc)),
                              environment(std::move(e)),
                              rkf_parameters(std::move(rp)) {}

    static Simulation from_json(const nlohmann::json &json);

    inline std::size_t n_samples() const { return propagation_context.samples_data.n_vecs; }
};
