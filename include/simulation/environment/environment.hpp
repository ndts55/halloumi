#pragma once
#include <nlohmann/json.hpp>
#include "core/types.hpp"
#include "simulation/environment/ephemeris.hpp"

struct Environment
{
    Ephemeris ephemeris;
    // TODO constants
    // TODO active bodies

    Environment(Ephemeris &&e) : ephemeris(std::move(e)) {}

    static Environment from_json(const nlohmann::json &json);
};
