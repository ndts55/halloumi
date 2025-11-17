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

#pragma region Array Operations

    template <std::size_t Start, std::size_t Size>
    __host__ __device__ inline Vec<T, Size> slice() const
    {
        Vec<T, Size> result;
#pragma unroll
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
#pragma unroll
        for (std::size_t i = 0; i < N; ++i)
        {
            result[i] = data[i];
        }

#pragma unroll
        for (std::size_t i = 0; i < TailN; ++i)
        {
            result[N + i] = other[i];
        }
        return result;
    }

#pragma endregion

#pragma region Vector Operations

    __host__ __device__ inline T squared_norm() const
    {
        T norm{0};
#pragma unroll
        for (const T &v : data)
        {
            norm += v * v;
        }
        return norm;
    }

    __host__ __device__ inline T cubed_norm() const
    {
        const auto sn = squared_norm();
        return sn * sqrtf(sn);
    }

    __host__ __device__ inline T reciprocal_cubed_norm() const
    {
        return 1.0 / cubed_norm();
    }

    __host__ __device__ inline Vec<T, N> componentwise_abs() const
    {
        Vec<T, N> result;
#pragma unroll
        for (std::size_t i = 0; i < N; ++i)
        {
            result[i] = abs(data[i]);
        }
        return result;
    }

    __host__ __device__ inline Vec<T, N> componentwise_max(const Vec<T, N> &other) const
    {
        Vec<T, N> result;
#pragma unroll
        for (std::size_t i = 0; i < N; ++i)
        {
            result[i] = max(data[i], other[i]);
        }
        return result;
    }

    __host__ __device__ inline T max_norm() const
    {
        T max_value = data[0];
#pragma unroll
        for (std::size_t i = 1; i < N; ++i)
        {
            max_value = max(max_value, data[i]);
        }
        return max_value;
    }

#pragma endregion

#pragma region Arithmetic Operators

    __host__ __device__ inline Vec<T, N> operator+(const Vec<T, N> &other) const
    {
        Vec<T, N> result;
#pragma unroll
        for (std::size_t i = 0; i < N; ++i)
        {
            result[i] = data[i] + other[i];
        }
        return result;
    }

    __host__ __device__ inline Vec<T, N> operator-(const Vec<T, N> &other) const
    {
        Vec<T, N> result;
#pragma unroll
        for (std::size_t i = 0; i < N; ++i)
        {
            result[i] = data[i] - other[i];
        }
        return result;
    }

    __host__ __device__ inline Vec<T, N> &operator+=(const Vec<T, N> &other)
    {
#pragma unroll
        for (std::size_t i = 0; i < N; ++i)
        {
            data[i] += other.data[i];
        }
        return *this;
    }

    __host__ __device__ inline Vec<T, N> &operator-=(const Vec<T, N> &other)
    {
#pragma unroll
        for (std::size_t i = 0; i < N; ++i)
            data[i] -= other.data[i];
        return *this;
    }

    __host__ __device__ inline Vec<T, N> operator*(const T scalar) const
    {
        Vec<T, N> result;
#pragma unroll
        for (std::size_t i = 0; i < N; ++i)
        {
            result[i] = data[i] * scalar;
        }
        return result;
    }

    __host__ __device__ inline Vec<T, N> &operator*=(const T scalar)
    {
#pragma unroll
        for (std::size_t i = 0; i < N; ++i)
        {
            data[i] *= scalar;
        }
        return *this;
    }

    __host__ __device__ inline Vec<T, N> operator+(const T scalar) const
    {
        Vec<T, N> result;
#pragma unroll
        for (std::size_t i = 0; i < N; ++i)
        {
            result[i] = data[i] + scalar;
        }
        return result;
    }

    __host__ __device__ inline Vec<T, N> operator-(const T scalar) const
    {
        Vec<T, N> result;
#pragma unroll
        for (std::size_t i = 0; i < N; ++i)
        {
            result[i] = data[i] - scalar;
        }
        return result;
    }

    __host__ __device__ inline Vec<T, N> &operator+=(const T scalar)
    {
#pragma unroll
        for (std::size_t i = 0; i < N; ++i)
        {
            data[i] += scalar;
        }
        return *this;
    }

    __host__ __device__ inline Vec<T, N> &operator-=(const T scalar)
    {
#pragma unroll
        for (std::size_t i = 0; i < N; ++i)
        {
            data[i] -= scalar;
        }
        return *this;
    }

    __host__ __device__ inline Vec<T, N> operator/(const T scalar) const
    {
        Vec<T, N> result;
#pragma unroll
        for (std::size_t i = 0; i < N; ++i)
        {
            result[i] = data[i] / scalar;
        }
        return result;
    }

    __host__ __device__ inline Vec<T, N> &operator/=(const T scalar)
    {
#pragma unroll
        for (std::size_t i = 0; i < N; ++i)
        {
            data[i] /= scalar;
        }
        return *this;
    }

    __host__ __device__ inline Vec<T, N> operator/(const Vec<T, N> &other) const
    {
        Vec<T, N> result;
#pragma unroll
        for (std::size_t i = 0; i < N; ++i)
        {
            result[i] = data[i] / other[i];
        }
        return result;
    }

    __host__ __device__ inline Vec<T, N> &operator/=(const Vec<T, N> &other)
    {
#pragma unroll
        for (std::size_t i = 0; i < N; ++i)
        {
            data[i] /= other[i];
        }
        return *this;
    }

#pragma endregion

#pragma region Iterator Functions

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

#pragma endregion
};
