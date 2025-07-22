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
void fill_cuda_array(std::size_t n_elements, CudaPtr<T[]> &data_, T fill_value)
{
    std::fill(data_.get(), data_.get() + n_elements, fill_value);
}

template <typename T>
void prefetch_async(std::size_t n_elements, CudaPtr<T[]> &data_)
{
    cudaError_t err = cudaMemPrefetchAsync(
        data_.get(),
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
    CudaPtr<T[]> data_;

public:
    CudaArray1D() = default;                 // Default Constructor
    CudaArray1D(const CudaArray1D<T> &other) // Copy Constructor
    {
        n_elements_ = other.n_elements_;
        data_ = make_managed_cuda_array<T>(n_elements_);
        std::copy(other.data_.get(), other.data_.get() + n_elements_, data_.get());
    }
    CudaArray1D(CudaArray1D<T> &&other) = default;               // Move Constructor
    CudaArray1D<T> &operator=(CudaArray1D<T> &&other) = default; // Move Assignment
    CudaArray1D(std::size_t n_elements) : n_elements_(n_elements)
    {
        data_ = make_managed_cuda_array<T>(n_elements_);
    }
    CudaArray1D(std::size_t n_elements, T initial_value) : n_elements_(n_elements)
    {
        data_ = make_managed_cuda_array<T>(n_elements_);
        fill_cuda_array(n_elements_, data_, initial_value);
    }

    inline T &at(std::size_t idx)
    {
#ifndef NDEBUG
        if (idx >= n_elements_)
        {
            throw std::out_of_range("Index out of range in CudaArray1D");
        }
#endif
        return data_[idx];
    }
    inline const T &at(std::size_t idx) const
    {
#ifndef NDEBUG
        if (idx >= n_elements_)
        {
            throw std::out_of_range("Index out of range in CudaArray1D");
        }
#endif
        return data_[idx];
    }

    inline std::size_t n_elements() const { return n_elements_; }
    inline std::size_t size() const { return n_elements_; }

    inline DeviceArray1D<T> get(CudaArrayPrefetch prefetch = CudaArrayPrefetch::NoPrefetch) const
    {
        if (prefetch == CudaArrayPrefetch::PrefetchToDevice)
        {
            prefetch_async(n_elements_, data_);
        }
        return DeviceArray1D<T>(data_.get(), n_elements_);
    }

    CudaPtr<T[]> &data() { return data_; }

    // Iterator support
    using iterator = T *;
    using const_iterator = const T *;

    iterator begin() { return data_.get(); }
    iterator end() { return data_.get() + n_elements_; }

    const_iterator begin() const { return data_.get(); }
    const_iterator end() const { return data_.get() + n_elements_; }

    const_iterator cbegin() const { return data_.get(); }
    const_iterator cend() const { return data_.get() + n_elements_; }
};

template <typename T, std::size_t VEC_SIZE>
class CudaArray2D
{
private:
    std::size_t n_vecs_;
    CudaPtr<T[]> data_;

public:
    CudaArray2D() = default;                                 // Default Constructor
    CudaArray2D(CudaArray2D<T, VEC_SIZE> &&other) = default; // Move Constructor
    CudaArray2D(std::size_t n_vecs) : n_vecs_(n_vecs)
    {
        data_ = make_managed_cuda_array<T>(n_vecs_ * VEC_SIZE);
    }
    CudaArray2D<T, VEC_SIZE> &operator=(CudaArray2D<T, VEC_SIZE> &&other) = default; // Move Assignment

    inline T &at(std::size_t dim, std::size_t idx)
    {
        auto index = get_2d_index(n_vecs_, dim, idx);
#ifndef NDEBUG
        if (index > size())
        {
            throw std::out_of_range("Index out of range in CudaArray2D");
        }
#endif
        return data_[index];
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
        return data_[index];
    }

    inline std::size_t n_vecs() const { return n_vecs_; }
    inline std::size_t size() const { return n_vecs_ * VEC_SIZE; }
    inline std::size_t vec_size() const { return VEC_SIZE; }

    inline DeviceArray2D<T, VEC_SIZE> get(CudaArrayPrefetch prefetch = CudaArrayPrefetch::NoPrefetch) const
    {
        if (prefetch == CudaArrayPrefetch::PrefetchToDevice)
        {
            prefetch_async(size(), data_);
        }
        return DeviceArray2D<T, VEC_SIZE>{data_.get(), n_vecs_};
    }

    CudaPtr<T[]> &data() { return data_; }

    // Iterator support
    using iterator = T *;
    using const_iterator = const T *;

    iterator begin() { return data_.get(); }
    iterator end() { return data_.get() + size(); }

    const_iterator begin() const { return data_.get(); }
    const_iterator end() const { return data_.get() + size(); }

    const_iterator cbegin() const { return data_.get(); }
    const_iterator cend() const { return data_.get() + size(); }
};

template <typename T, std::size_t VEC_SIZE, std::size_t N_STAGES>
class CudaArray3D
{
private:
    std::size_t n_vecs_;
    CudaPtr<T[]> data_;

public:
    CudaArray3D() = default;                                                                             // Default Constructor
    CudaArray3D(CudaArray3D<T, VEC_SIZE, N_STAGES> &&other) = default;                                   // Move Constructor
    CudaArray3D<T, VEC_SIZE, N_STAGES> &operator=(CudaArray3D<T, VEC_SIZE, N_STAGES> &&other) = default; // Move Assignment
    CudaArray3D(std::size_t n_vecs) : n_vecs_(n_vecs)
    {
        data_ = make_managed_cuda_array<T>(n_vecs_ * VEC_SIZE * N_STAGES);
    }

    inline T &at(std::size_t dim, std::size_t stage, std::size_t idx)
    {
        auto index = get_3d_index(n_vecs_, dim, stage, idx);
#ifndef NDEBUG
        if (index >= size())
        {
            throw std::out_of_range("Index out of range in CudaArray3D");
        }
#endif
        return data_[index];
    }
    inline const T &at(std::size_t dim, std::size_t stage, std::size_t idx) const
    {
        auto index = get_3d_index(n_vecs_, dim, stage, idx);
#ifndef NDEBUG
        if (index >= size())
        {
            throw std::out_of_range("Index out of range in CudaArray3D");
        }
#endif
        return data_[index];
    }

    inline std::size_t n_vecs() const { return n_vecs_; }
    inline std::size_t size() const { return n_vecs_ * VEC_SIZE * N_STAGES; }
    inline std::size_t vec_size() const { return VEC_SIZE; }
    inline std::size_t n_stages() const { return N_STAGES; }

    inline DeviceArray3D<T, VEC_SIZE, N_STAGES> get(CudaArrayPrefetch prefetch = CudaArrayPrefetch::NoPrefetch) const
    {
        if (prefetch == CudaArrayPrefetch::PrefetchToDevice)
        {
            prefetch_async(size(), data_);
        }
        return DeviceArray3D<T, VEC_SIZE, N_STAGES>{data_.get(), n_vecs_};
    }

    CudaPtr<T[]> &data() { return data_; }

    // Iterator support
    using iterator = T *;
    using const_iterator = const T *;

    iterator begin() { return data_.get(); }
    iterator end() { return data_.get() + size(); }

    const_iterator begin() const { return data_.get(); }
    const_iterator end() const { return data_.get() + size(); }

    const_iterator cbegin() const { return data_.get(); }
    const_iterator cend() const { return data_.get() + size(); }
};
