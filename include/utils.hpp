#pragma once
#include <nlohmann/json.hpp>
#include <fstream>
#include <string>
#include <iostream>
#include <cuda_runtime.h>

nlohmann::json json_from_file(const std::string &path);

nlohmann::json json_from_cbor(const std::string &file);

void json_to_file(const nlohmann::json &json, const std::string &path);

void print_json(const nlohmann::json &json);

__device__ __host__ inline std::size_t get_2d_index(const std::size_t &number_of_vectors, const std::size_t &component_index, const std::size_t &vector_index)
{
    // select component row, then index into it with vector index
    return component_index * number_of_vectors + vector_index;
}

template <std::size_t VEC_SIZE>
__device__ __host__ inline std::size_t get_3d_index(const std::size_t &number_of_vectors, const std::size_t &component_index, const std::size_t &stage_index, const std::size_t &vector_index)
{
    // select stage matrix from which we select component row, then index into it with vector index
    return (stage_index * VEC_SIZE + component_index) * number_of_vectors + vector_index; // = (stage_index * number_of_vectors * VEC_SIZE) + (component_index * number_of_vectors) + vector_index;
}

template <typename T>
void print_vector_mean(const std::vector<T> &vec, const std::string &name)
{
    if (vec.empty())
    {
        std::cout << name << " is empty." << std::endl;
        return;
    }
    T sum = std::accumulate(vec.begin(), vec.end(), T(0));
    T mean = sum / static_cast<T>(vec.size());
    std::cout << name << " mean: " << mean << std::endl;
}

template <typename T>
void print_failed_count(const std::vector<T> &vec, const std::string &name, const T &threshold = T(0))
{
    if (vec.empty())
    {
        std::cout << name << " is empty." << std::endl;
        return;
    }
    auto failed_count = std::count_if(vec.begin(), vec.end(), [&](const T &value)
                                      { return value > threshold; });
    std::cout << name << " failed count: " << failed_count << std::endl;
}
