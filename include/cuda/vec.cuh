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

    __host__ __device__ Vec<T, N> operator+(const Vec<T, N> &other) const
    {
        Vec<T, N> result;
        for (std::size_t i = 0; i < N; ++i)
        {
            result[i] = data[i] + other[i];
        }
        return result;
    }

    __host__ __device__ Vec<T, N> operator-(const Vec<T, N> &other) const
    {
        Vec<T, N> result;
        for (std::size_t i = 0; i < N; ++i)
        {
            result[i] = data[i] - other[i];
        }
        return result;
    }

    __host__ __device__ Vec<T, N> &operator+=(const Vec<T, N> &other)
    {
        for (std::size_t i = 0; i < N; ++i)
            data[i] += other.data[i];
        return *this;
    }

    __host__ __device__ Vec<T, N> &operator-=(const Vec<T, N> &other)
    {
        for (std::size_t i = 0; i < N; ++i)
            data[i] -= other.data[i];
        return *this;
    }

    __host__ __device__ inline T *begin()
    {
        return data;
    }

    __host__ __device__ inline T *end()
    {
        return data + N;
    }

    __host__ __device__ inline const T *cbegin() const
    {
        return data;
    }

    __host__ __device__ inline const T *cend() const
    {
        return data + N;
    }
};
