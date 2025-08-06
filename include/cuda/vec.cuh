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

    template <std::size_t Start, std::size_t Size>
    __host__ __device__ inline Vec<T, Size> slice() const
    {
        Vec<T, Size> result;
        for (std::size_t i = 0; i < Size; ++i)
        {
            result[i] = data[Start + i];
        }
        return result;
    }

    template <std::size_t TailN, std::size_t TotalN = N + TailN>
    __host__ __device__ inline Vec<T, TotalN> append(const Vec<T, TailN> &other) const
    {
        Vec<T, TotalN> result;
        for (std::size_t i = 0; i < N; ++i)
        {
            result[i] = data[i];
        }

        for (std::size_t i = 0; i < TailN; ++i)
        {
            result[N + i] = other[i];
        }
        return result;
    }

    __host__ __device__ inline Vec<T, N> operator+(const Vec<T, N> &other) const
    {
        Vec<T, N> result;
        for (std::size_t i = 0; i < N; ++i)
        {
            result[i] = data[i] + other[i];
        }
        return result;
    }

    __host__ __device__ inline Vec<T, N> operator-(const Vec<T, N> &other) const
    {
        Vec<T, N> result;
        for (std::size_t i = 0; i < N; ++i)
        {
            result[i] = data[i] - other[i];
        }
        return result;
    }

    __host__ __device__ inline Vec<T, N> &operator+=(const Vec<T, N> &other)
    {
        for (std::size_t i = 0; i < N; ++i)
            data[i] += other.data[i];
        return *this;
    }

    __host__ __device__ inline Vec<T, N> &operator-=(const Vec<T, N> &other)
    {
        for (std::size_t i = 0; i < N; ++i)
            data[i] -= other.data[i];
        return *this;
    }

    __host__ __device__ inline Vec<T, N> operator*(const T scalar) const
    {
        Vec<T, N> result;
        for (std::size_t i = 0; i < N; ++i)
        {
            result[i] = data[i] * scalar;
        }
        return result;
    }

    __host__ __device__ inline Vec<T, N> &operator*=(const T scalar)
    {
        for (std::size_t i = 0; i < N; ++i)
            data[i] *= scalar;
        return *this;
    }

    __host__ __device__ inline Vec<T, N> operator+(const T scalar) const
    {
        Vec<T, N> result;
        for (std::size_t i = 0; i < N; ++i)
        {
            result[i] = data[i] + scalar;
        }
        return result;
    }

    __host__ __device__ inline Vec<T, N> operator-(const T scalar) const
    {
        Vec<T, N> result;
        for (std::size_t i = 0; i < N; ++i)
        {
            result[i] = data[i] - scalar;
        }
        return result;
    }

    __host__ __device__ inline Vec<T, N> &operator+=(const T scalar)
    {
        for (std::size_t i = 0; i < N; ++i)
            data[i] += scalar;
        return *this;
    }

    __host__ __device__ inline Vec<T, N> &operator-=(const T scalar)
    {
        for (std::size_t i = 0; i < N; ++i)
            data[i] -= scalar;
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
