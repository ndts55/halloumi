#pragma once
#include <cuda_runtime.h>
#include "cuda/utils.cuh"
#include "cuda/vec.cuh"
#include "utils.hpp"
#include <iostream>
#include "cuda/cuda_ptr.cuh"

template <typename T, std::size_t VEC_SIZE>
struct DeviceMatrix
{
    std::size_t n_vecs;
    T *data;

    __device__ inline Vec<T, VEC_SIZE> vector_at(std::size_t vec_index) const
    {
        Vec<T, VEC_SIZE> vec;
#pragma unroll
        for (std::size_t dim = 0; dim < VEC_SIZE; ++dim)
        {
            vec[dim] = at(vec_index, dim);
        }
        return vec;
    }

    __device__ inline void set_vector_at(std::size_t vec_index, const Vec<T, VEC_SIZE> &vec)
    {
#pragma unroll
        for (std::size_t dim = 0; dim < VEC_SIZE; ++dim)
        {
            at(vec_index, dim) = vec[dim];
        }
    }

    __device__ inline T &at(
        const std::size_t &vec_index,
        const std::size_t &com_index)
    {
        return data[get_2d_index_(n_vecs, vec_index, com_index)];
    }
    __device__ inline const T &at(const std::size_t &vec_index, const std::size_t &com_index) const
    {
        return data[get_2d_index_(n_vecs, vec_index, com_index)];
    }

    // Device-side iterators (over the flat data array)
    __device__ inline T *begin() { return data; }
    __device__ inline T *end() { return data + n_vecs * VEC_SIZE; }
    __device__ inline const T *cbegin() const { return data; }
    __device__ inline const T *cend() const { return data + n_vecs * VEC_SIZE; }
};

template <typename T, std::size_t VEC_SIZE>
class HostMatrix
{
private:
    std::size_t n_vecs_;
    std::vector<T> host_data_;
    CudaPtr<T[]> device_data_;

public:
    HostMatrix(std::size_t n_vecs) : n_vecs_(n_vecs), host_data_(n_vecs * VEC_SIZE), device_data_(make_cuda_array<T>(n_vecs * VEC_SIZE)) {}
    HostMatrix(
        std::size_t n_vecs,
        const T &fill_value) : n_vecs_(n_vecs), host_data_(n_vecs * VEC_SIZE, fill_value),
                               device_data_(make_cuda_array<T>(n_vecs * VEC_SIZE)) {}
    HostMatrix(std::vector<T> &&data) : n_vecs_(data.size() / VEC_SIZE),
                                        host_data_(std::move(data)),
                                        device_data_(make_cuda_array<T>(host_data_.size()))
    {
        if (host_data_.size() % VEC_SIZE != 0)
        {
            throw std::invalid_argument("HostMatrix data size must be a multiple of VEC_SIZE");
        }
    }
    inline T &at(
        const std::size_t &vec_index,
        const std::size_t &com_index)
    {
        return host_data_.at(get_2d_index_(n_vecs_, vec_index, com_index));
    }

    inline const T &at(
        const std::size_t &vec_index,
        const std::size_t &com_index) const
    {
        return host_data_.at(get_2d_index_(n_vecs_, vec_index, com_index));
    }

    cudaError_t copy_to_device()
    {
        return copy_to_device_(host_data_, device_data_);
    }

    cudaError_t copy_to_host()
    {
        std::cout << "Copying Matrix to host" << std::endl;
        return copy_to_host_(device_data_, host_data_);
    }

    DeviceMatrix<T, VEC_SIZE> get() const
    {
        return {.n_vecs = n_vecs_, .data = device_data_.get()};
    }

    inline std::size_t size() const { return host_data_.size(); }
    inline bool empty() const { return host_data_.empty(); }
    inline std::size_t n_vecs() const { return n_vecs_; }

    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;

    iterator begin() { return host_data_.begin(); }
    iterator end() { return host_data_.end(); }

    const_iterator begin() const { return host_data_.begin(); }
    const_iterator end() const { return host_data_.end(); }

    const_iterator cbegin() const { return host_data_.cbegin(); }
    const_iterator cend() const { return host_data_.cend(); }
};
