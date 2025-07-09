#pragma once
#include "types.hpp"
#include <nlohmann/json.hpp>

// Samples SoA
class Samples
{
private:
    size_t n_vecs;
    Integer center_of_integration;
    std::vector<Float> data;
    std::vector<Float> epochs;

public:
    Samples() = default;
    Samples(const std::vector<Float> &&d, const std::vector<Float> &&e, Integer center_of_integration);
    static Samples from_json(const nlohmann::json &json);
    size_t size() const;
};