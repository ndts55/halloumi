#pragma once
#include <nlohmann/json.hpp>
#include "samples.hpp"
#include "types.hpp"
#include "ephemeris.hpp"

class Simulation
{
private:
    Float duration_in_days_;
    Samples samples_;
    Ephemeris ephemeris_;
    // ? with_collisison: boolean
    // ? validation : Validation struct

public:
    Simulation(const Float d, const Samples &&s, const Ephemeris &&e) : duration_in_days_(d), samples_(std::move(s)), ephemeris_(std::move(e)) {}

    const Ephemeris &ephemeris() { return ephemeris_; }
    const Samples &samples() { return samples_; }
    const Float duration_in_days() { return duration_in_days_; }

    static Simulation from_json(const nlohmann::json &json);
};
