#pragma once
#include <cuda_runtime.h>
#include "utils.hpp"

template <typename T>
struct DeviceArray1D
{
    T *data;
    std::size_t n_vecs;

    __device__ inline T &at(std::size_t index)
    {
        return data[index];
    }

    __device__ inline const T &at(std::size_t index) const
    {
        return data[index];
    }
};

template <typename T, std::size_t VEC_SIZE>
struct DeviceArray2D
{
    T *data;
    std::size_t n_vecs;

    __device__ inline T &at(std::size_t dim, std::size_t idx)
    {
        return data[get_2d_index(n_vecs, dim, idx)];
    }
    __device__ inline const T &at(std::size_t dim, std::size_t idx) const
    {
        return data[get_2d_index(n_vecs, dim, idx)];
    }
};

template <typename T, std::size_t VEC_SIZE, std::size_t N_STAGES>
struct DeviceArray3D
{
    T *data;
    std::size_t n_vecs;

    __device__ inline T &at(std::size_t dim, std::size_t stage, std::size_t idx)
    {
        return data[get_3d_index(n_vecs, dim, stage, idx)];
    }
    __device__ inline const T &at(std::size_t dim, std::size_t stage, std::size_t idx) const
    {
        return data[get_3d_index(n_vecs, dim, stage, idx)];
    }
};
