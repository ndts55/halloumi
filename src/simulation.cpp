#include <iostream>
#include <nlohmann/json.hpp>
#include "types.hpp"
#include "simulation.hpp"
#include "utils.hpp"
#include "propagation_context.hpp"
#include "ephemeris.hpp"

Simulation Simulation::from_json(const nlohmann::json &configuration)
{
    Float duration_in_days = configuration["simConfig"]["durationDays"];

    auto propagation_context = PropagationContext::from_json(configuration);

    auto exec_config = json_from_file(configuration["simConfig"]["config"]);
    auto brie_file = exec_config["model"]["environment"]["ephemeris"];
    auto ephemeris = Ephemeris::from_brie(brie_file);
    auto rkf_params = RKFParameters::from_json(exec_config["integration"]);

    return Simulation{std::move(propagation_context), std::move(ephemeris), std::move(rkf_params)};
}