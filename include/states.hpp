#pragma once
#include "types.hpp"
#include <nlohmann/json.hpp>

// TODO implement RAII-style classes for samples

// Number of vectors should be given by size of data / 6, we could also save that separately in the data-structure
class StatesArray
{
private:
    std::vector<Float> data;
    size_t n_vecs;

public:
    StatesArray() = delete;
    StatesArray(const std::vector<Float> &&d);
    static StatesArray from_json(const nlohmann::json &json);
    size_t size() const;
    template <size_t dim>
    const Float &operator[](const size_t &idx) const;
    template <size_t dim>
    Float &operator[](const size_t &idx);
};