#pragma once
#include "core/types.cuh"
#include "utils.cuh"
#include <string>
#include <nlohmann/json.hpp>

void dump_d_states(const HostDerivativesTensor &d_states, const std::string &filename = "d_states.json");

void dump_states(const HostStatesMatrix &states, const std::string &filename = "states.json");

template <typename T>
void dump_array(const HostArray<T> &array, const std::string &filename)
{
    auto json_array = nlohmann::json::array();
    for (std::size_t i = 0; i < array.size(); ++i)
    {
        json_array.push_back(array.at(i));
    }
    json_to_file(json_array, filename);
}