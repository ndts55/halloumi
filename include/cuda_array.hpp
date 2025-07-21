#pragma once
#include <memory>
#include <vector>
#include <cuda_runtime.h>
#include "device_array.cuh"
#include "utils.hpp"

#pragma region CudaPtr

struct CudaDeleter
{
    void operator()(void *p) const { cudaFree(p); }
};

template <typename T>
using CudaPtr = std::unique_ptr<T, CudaDeleter>;

template <typename T>
CudaPtr<T[]> make_managed_cuda_array(std::size_t n_elements)
{
    void *pointer = nullptr;
    auto error = cudaMallocManaged(&pointer, sizeof(T) * n_elements, cudaMemAttachGlobal);
    if (error != cudaSuccess || !pointer)
    {
        throw std::bad_alloc();
    }
    return CudaPtr<T[]>(static_cast<T *>(pointer));
}

template <typename T>
void fill_cuda_array(std::size_t n_elements, CudaPtr<T[]> &data, T fill_value)
{
    std::fill(data.get(), data.get() + n_elements, fill_value);
}

template <typename T>
void prefetch_async(std::size_t n_elements, CudaPtr<T[]> &data)
{
    cudaError_t err = cudaMemPrefetchAsync(
        data.get(),
        n_elements * sizeof(T),
        0, // device ID
        0  // stream
    );
    if (err != cudaSuccess)
    {
        throw std::runtime_error("cudaMemPrefetchAsync failed");
    }
}

#pragma endregion

enum class CudaArrayPrefetch
{
    NoPrefetch,
    PrefetchToDevice
};

template <typename T>
class CudaArray1D
{
private:
    std::size_t n_elements_;
    CudaPtr<T[]> data;

public:
    CudaArray1D() = default;
    CudaArray1D(const CudaArray1D<T> &other)
    {
        n_elements_ = other.n_elements_;
        data = make_managed_cuda_array<T>(n_elements_);
        std::copy(other.data.get(), other.data.get() + n_elements_, data.get());
    }
    CudaArray1D(CudaArray1D<T> &&other) = default;
    CudaArray1D(std::size_t n_elements) : n_elements_(n_elements)
    {
        data = make_managed_cuda_array<T>(n_elements_);
    }
    CudaArray1D(std::size_t n_elements, T initial_value) : n_elements_(n_elements)
    {
        data = make_managed_cuda_array<T>(n_elements_);
        fill_cuda_array(n_elements_, data, initial_value);
    }
    CudaArray1D<T> &operator=(CudaArray1D<T> &&other) = default;
    inline T &at(std::size_t idx)
    {
#ifndef NDEBUG
        if (idx >= n_elements_)
        {
            throw std::out_of_range("Index out of range in CudaArray1D");
        }
#endif
        return data[idx];
    }
    inline const T &at(std::size_t idx) const
    {
#ifndef NDEBUG
        if (idx >= n_elements_)
        {
            throw std::out_of_range("Index out of range in CudaArray1D");
        }
#endif
        return data[idx];
    }
    std::size_t n_elements() const { return n_elements_; }
    std::size_t size() const { return n_elements_; }

    DeviceArray1D<T> get() const;
};

template <typename T, std::size_t VEC_SIZE>
class CudaArray2D
{
private:
    std::size_t n_vecs_;
    CudaPtr<T[]> data;

public:
    CudaArray2D() = default;
    CudaArray2D(CudaArray2D<T, VEC_SIZE> &&other) = default;
    CudaArray2D(std::size_t n_vecs) : n_vecs_(n_vecs)
    {
        data = make_managed_cuda_array<T>(n_vecs_ * VEC_SIZE);
    }
    CudaArray2D<T, VEC_SIZE> &operator=(CudaArray2D<T, VEC_SIZE> &&other) = default;
    inline T &at(std::size_t dim, std::size_t idx)
    {
        auto index = get_2d_index(n_vecs_, dim, idx);
#ifndef NDEBUG
        if (index > size())
        {
            throw std::out_of_range("Index out of range in CudaArray2D");
        }
#endif
        return data[index];
    }
    inline const T &at(std::size_t dim, std::size_t idx) const
    {
        auto index = get_2d_index(n_vecs_, dim, idx);
#ifndef NDEBUG
        if (index > size())
        {
            throw std::out_of_range("Index out of range in CudaArray2D");
        }
#endif
        return data[index];
    }
    std::size_t n_vecs() const { return n_vecs_; }
    std::size_t size() const { return n_vecs_ * VEC_SIZE; }

    DeviceArray2D<T, VEC_SIZE> get() const
    {
        return DeviceArray2D<T, VEC_SIZE>{data.get(), n_vecs_};
    }
};

template <typename T, std::size_t VEC_SIZE, std::size_t N_STAGES>
class CudaArray3D
{
private:
    std::size_t n_vecs_;
    CudaPtr<T[]> data;

public:
    CudaArray3D() = default;
    CudaArray3D(CudaArray3D<T, VEC_SIZE, N_STAGES> &&other) = default;
    CudaArray3D(std::size_t n_vecs) : n_vecs_(n_vecs)
    {
        data = make_managed_cuda_array<T>(n_vecs_ * VEC_SIZE * N_STAGES);
    }
    CudaArray3D<T, VEC_SIZE, N_STAGES> &operator=(CudaArray3D<T, VEC_SIZE, N_STAGES> &&other) = default;
    T &at(std::size_t dim, std::size_t stage, std::size_t idx)
    {
        auto index = get_3d_index(n_vecs_, dim, stage, idx);
#ifndef NDEBUG
        if (index >= size())
        {
            throw std::out_of_range("Index out of range in CudaArray3D");
        }
#endif
        return data[index];
    }
    const T &at(std::size_t dim, std::size_t stage, std::size_t idx) const
    {
        auto index = get_3d_index(n_vecs_, dim, stage, idx);
#ifndef NDEBUG
        if (index >= size())
        {
            throw std::out_of_range("Index out of range in CudaArray3D");
        }
#endif
        return data[index];
    }
    std::size_t n_vecs() const { return n_vecs_; }
    std::size_t size() const { return n_vecs_ * VEC_SIZE * N_STAGES; }

    DeviceArray3D<T, VEC_SIZE, N_STAGES> get() const
    {
        return DeviceArray3D<T, VEC_SIZE, N_STAGES>{data.get(), n_vecs_};
    }
};
