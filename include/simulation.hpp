#pragma once
#include <nlohmann/json.hpp>
#include "propagation_context.hpp"
#include "types.hpp"
#include "ephemeris.hpp"
#include "rkf_parameters.hpp"

struct Simulation
{
    PropagationContext propagation_context;
    const Ephemeris ephemeris;
    const RKFParameters rkf_parameters;
    // ? with_collisison: bool
    // ? validation : Validation struct

    Simulation(
        PropagationContext &&pc,
        Ephemeris &&e,
        RKFParameters &&rp) : propagation_context(std::move(pc)),
                              ephemeris(std::move(e)),
                              rkf_parameters(std::move(rp)) {}

    static Simulation from_json(const nlohmann::json &json);
};
