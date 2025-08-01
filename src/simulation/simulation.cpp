#include <iostream>
#include <nlohmann/json.hpp>
#include "core/types.hpp"
#include "simulation/simulation.hpp"
#include "utils.hpp"
#include "simulation/propagation_context.hpp"
#include "simulation/ephemeris.hpp"

Simulation Simulation::from_json(const nlohmann::json &configuration)
{
    auto exec_config = json_from_file(configuration["simConfig"]["config"]);
    auto brie_file = exec_config["model"]["environment"]["ephemeris"];

    return Simulation(
        PropagationContext::from_json(configuration),
        Ephemeris::from_brie(brie_file),
        RKFParameters::from_json(exec_config["integration"]));
}