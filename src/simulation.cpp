#include <iostream>
#include <nlohmann/json.hpp>
#include "types.hpp"
#include "simulation.hpp"
#include "utils.hpp"
#include "propagation_context.hpp"
#include "ephemeris.hpp"

Simulation Simulation::from_json(const nlohmann::json &configuration)
{
    auto exec_config = json_from_file(configuration["simConfig"]["config"]);
    auto brie_file = exec_config["model"]["environment"]["ephemeris"];

    return Simulation(
        PropagationContext::from_json(configuration),
        Ephemeris::from_brie(brie_file),
        RKFParameters::from_json(exec_config["integration"]));
}