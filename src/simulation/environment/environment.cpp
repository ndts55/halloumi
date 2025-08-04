#include "simulation/environment/environment.hpp"
#include <nlohmann/json.hpp>

Environment Environment::from_json(const nlohmann::json &json)
{
    Ephemeris ephemeris = Ephemeris::from_brie(json["ephemeris"]);

    return Environment(std::move(ephemeris));
}