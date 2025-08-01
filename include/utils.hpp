#pragma once
#include <nlohmann/json.hpp>
#include <fstream>
#include <string>
#include <iostream>
#include <cuda_runtime.h>
#include "core/types.hpp"

nlohmann::json json_from_file(const std::string &path);

nlohmann::json json_from_cbor(const std::string &file);

void print_json(const nlohmann::json &json);

__device__ __host__ inline std::size_t get_2d_index(const std::size_t &n_vecs, const std::size_t &dim, const std::size_t &index)
{
    return dim * n_vecs + index;
}

__device__ __host__ inline std::size_t get_3d_index(const std::size_t &n_vecs, const std::size_t &dim, const std::size_t &stage, const std::size_t &index)
{
    return (stage * STATE_DIM + dim) * n_vecs + index;
}
