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

    static Simulation from_json(const nlohmann::json &json);
};
