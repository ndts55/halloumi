#pragma once
#include <cuda_runtime.h>

template <typename T, std::size_t N>
struct Vec
{
    T data[N];

    __host__ __device__ T &operator[](std::size_t i)
    {
        return data[i];
    }

    __host__ __device__ const T &operator[](std::size_t i) const
    {
        return data[i];
    }

    __host__ __device__ inline std::size_t size() const
    {
        return N;
    }
};
