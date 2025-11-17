#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include "core/utils.cuh"
#include "core/cuda_ptr.cuh"

template <typename T>
struct DeviceArray
{
    using value_type = T;
    std::size_t n_elements;
    T *data;

    __device__ inline T &at(std::size_t index)
    {
        return data[index];
    }

    __device__ inline const T &at(std::size_t index) const
    {
        return data[index];
    }

    // Device-side iterators
    __device__ inline T *begin() { return data; }
    __device__ inline T *end() { return data + n_elements; }
    __device__ inline const T *cbegin() const { return data; }
    __device__ inline const T *cend() const { return data + n_elements; }
};

template <typename T>
class HostArray
{
private:
    std::vector<T> host_data_;
    CudaPtr<T[]> device_data_;

public:
    using value_type = T;
    HostArray(std::size_t n_elements) : host_data_(n_elements), device_data_(make_cuda_array<T>(n_elements)) {}
    HostArray(
        std::size_t n_elements,
        const T &fill_value) : host_data_(n_elements, fill_value),
                               device_data_(make_cuda_array<T>(n_elements)) {}
    HostArray(std::vector<T> &&data) : host_data_(std::move(data)),
                                       device_data_(make_cuda_array<T>(host_data_.size())) {}
    template <std::size_t N>
    HostArray(std::array<T, N> array) : host_data_(array.begin(), array.end()),
                                        device_data_(make_cuda_array<T>(array.size())) {}
    inline T &at(std::size_t index)
    {
        return host_data_.at(index);
    }

    inline const T &at(std::size_t index) const
    {
        return host_data_.at(index);
    }

    cudaError_t copy_to_device()
    {
        return copy_to_device_(host_data_, device_data_);
    }

    cudaError_t copy_to_host()
    {
        return copy_to_host_<T>(device_data_, host_data_);
    }

    DeviceArray<T> get() const
    {
        return {.n_elements = host_data_.size(), .data = device_data_.get()};
    }

    inline std::size_t n_elements() const { return host_data_.size(); }
    inline std::size_t size() const { return host_data_.size(); }
    inline bool empty() const { return host_data_.empty(); }

    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;

    iterator begin() { return host_data_.begin(); }
    iterator end() { return host_data_.end(); }

    const_iterator begin() const { return host_data_.begin(); }
    const_iterator end() const { return host_data_.end(); }

    const_iterator cbegin() const { return host_data_.cbegin(); }
    const_iterator cend() const { return host_data_.cend(); }
};