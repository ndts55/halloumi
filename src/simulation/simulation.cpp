#include <iostream>
#include <nlohmann/json.hpp>
#include "core/types.cuh"
#include "simulation/simulation.hpp"
#include "utils.hpp"
#include "simulation/propagation_context.hpp"
#include "simulation/environment/ephemeris.hpp"

Simulation Simulation::from_json(const nlohmann::json &configuration)
{
    auto exec_config = json_from_file(configuration["simConfig"]["config"]);
    auto environment_json = exec_config["model"]["environment"];

    return Simulation(
        PropagationContext::from_json(configuration),
        Ephemeris::from_brie(environment_json["ephemeris"]),
        RKFParameters::from_json(exec_config["integration"]));
}