#pragma once
#include <cuda_runtime.h>
#include "core/utils.cuh"
#include "core/vec.cuh"
#include "utils.cuh"
#include "core/cuda_ptr.cuh"

template <typename T, std::size_t MAT_SIZE, std::size_t VEC_SIZE>
struct DeviceTensor
{
    std::size_t n_vecs;
    T *data;

    __device__ inline Vec<T, VEC_SIZE> vector_at(std::size_t mat_index, std::size_t vec_index) const
    {
        Vec<T, VEC_SIZE> vec;
#pragma unroll
        for (std::size_t dim = 0; dim < VEC_SIZE; ++dim)
        {
            vec[dim] = at(mat_index, vec_index, dim);
        }
        return vec;
    }

    __device__ inline void set_vector_at(std::size_t mat_index, std::size_t vec_index, const Vec<T, VEC_SIZE> &vec)
    {
#pragma unroll
        for (std::size_t dim = 0; dim < VEC_SIZE; ++dim)
        {
            at(mat_index, vec_index, dim) = vec[dim];
        }
    }

    __device__ inline T &at(
        const std::size_t &mat_index,
        const std::size_t &vec_index,
        const std::size_t &com_index)
    {
        return data[get_3d_index<VEC_SIZE>(n_vecs, mat_index, vec_index, com_index)];
    }
    __device__ inline const T &at(
        const std::size_t &mat_index,
        const std::size_t &vec_index,
        const std::size_t &com_index) const
    {
        return data[get_3d_index<VEC_SIZE>(n_vecs, mat_index, vec_index, com_index)];
    }

    // Device-side iterators (over the flat data array)
    __device__ inline T *begin() { return data; }
    __device__ inline T *end() { return data + n_vecs * VEC_SIZE * MAT_SIZE; }
    __device__ inline const T *cbegin() const { return data; }
    __device__ inline const T *cend() const { return data + n_vecs * VEC_SIZE * MAT_SIZE; }
};

template <typename T, std::size_t MAT_SIZE, std::size_t VEC_SIZE>
class HostTensor
{
private:
    std::size_t n_mats_;
    std::vector<T> host_data_;
    CudaPtr<T[]> device_data_;

public:
    HostTensor(std::size_t n_mats) : n_mats_(n_mats),
                                     host_data_(n_mats_ * VEC_SIZE * MAT_SIZE),
                                     device_data_(make_cuda_array<T>(host_data_.size())) {}
    HostTensor(
        std::size_t n_mats,
        const T &fill_value) : n_mats_(n_mats),
                               host_data_(n_mats_ * VEC_SIZE * MAT_SIZE, fill_value),
                               device_data_(make_cuda_array<T>(host_data_.size())) {}
    HostTensor(std::vector<T> &&data) : n_mats_(data.size() / (VEC_SIZE * MAT_SIZE)),
                                        host_data_(std::move(data)),
                                        device_data_(make_cuda_array<T>(host_data_.size()))
    {
        if (host_data_.size() % (MAT_SIZE * VEC_SIZE) != 0)
        {
            throw std::invalid_argument("HostMatrix data size must be a multiple of MAT_SIZE * VEC_SIZE");
        }
    }
    inline T &at(
        const std::size_t &mat_index,
        const std::size_t &vec_index,
        const std::size_t &com_index)
    {
        return host_data_.at(get_3d_index<VEC_SIZE>(n_mats_, mat_index, vec_index, com_index));
    }

    inline const T &at(
        const std::size_t &mat_index,
        const std::size_t &vec_index,
        const std::size_t &com_index) const
    {
        return host_data_.at(get_3d_index<VEC_SIZE>(n_mats_, mat_index, vec_index, com_index));
    }

    cudaError_t copy_to_device()
    {
        return copy_to_device_(host_data_, device_data_);
    }

    cudaError_t copy_to_host()
    {
        return copy_to_host_(device_data_, host_data_);
    }

    DeviceTensor<T, MAT_SIZE, VEC_SIZE> get() const
    {
        return {.n_vecs = n_mats_, .data = device_data_.get()};
    }

    inline std::size_t size() const { return host_data_.size(); }
    inline bool empty() const { return host_data_.empty(); }
    inline std::size_t n_mats() const { return n_mats_; }

    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;

    iterator begin() { return host_data_.begin(); }
    iterator end() { return host_data_.end(); }

    const_iterator begin() const { return host_data_.begin(); }
    const_iterator end() const { return host_data_.end(); }

    const_iterator cbegin() const { return host_data_.cbegin(); }
    const_iterator cend() const { return host_data_.cend(); }
};
