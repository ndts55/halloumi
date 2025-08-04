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

    // Device-side iterators
    __device__ inline T* begin() { return data; }
    __device__ inline T* end() { return data + n_vecs; }
    __device__ inline const T* cbegin() const { return data; }
    __device__ inline const T* cend() const { return data + n_vecs; }
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

    // Device-side iterators (over the flat data array)
    __device__ inline T* begin() { return data; }
    __device__ inline T* end() { return data + n_vecs * VEC_SIZE; }
    __device__ inline const T* cbegin() const { return data; }
    __device__ inline const T* cend() const { return data + n_vecs * VEC_SIZE; }
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

    // Device-side iterators (over the flat data array)
    __device__ inline T* begin() { return data; }
    __device__ inline T* end() { return data + n_vecs * VEC_SIZE * N_STAGES; }
    __device__ inline const T* cbegin() const { return data; }
    __device__ inline const T* cend() const { return data + n_vecs * VEC_SIZE * N_STAGES; }
};
