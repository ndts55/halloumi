#pragma once
#include "types.hpp"
#include <nlohmann/json.hpp>

// Samples SoA
class Samples
{
private:
    size_t n_vecs; // ? redundant?
    std::vector<Float> data;
    std::vector<Float> epochs;
    // TODO add cois? for now we can get away with a default implementation that just returns 399 and is inlined

public:
    Samples() = delete;
    Samples(const std::vector<Float> &&d, const std::vector<Float> &&e);
    static Samples from_json(const nlohmann::json &json);
    size_t size() const;
};