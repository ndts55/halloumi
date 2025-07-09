#include <iostream>
#include <nlohmann/json.hpp>
#include "types.hpp"
#include "simulation.hpp"
#include "utils.hpp"
#include "samples.hpp"
#include "ephemeris.hpp"

Simulation Simulation::from_json(const nlohmann::json &configuration)
{
    Float duration_in_days = configuration["simConfig"]["durationDays"];

    auto samples_json = json_from_file(configuration["samples"]);
    auto samples = Samples::from_json(samples_json);

    auto exec_config = json_from_file(configuration["simConfig"]["config"]);
    auto brie_file = exec_config["model"]["environment"]["ephemeris"];
    auto ephemeris = Ephemeris::from_brie(brie_file);

    return Simulation(duration_in_days, std::move(samples), std::move(ephemeris));
}