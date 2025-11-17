#pragma once
#include <cuda_runtime.h>
#include <vector>
#include "core/cuda_ptr.cuh"

// Optimized for coalesced access when vec_index is thread index
__device__ __host__ inline std::size_t get_2d_index(
    const std::size_t &n_vecs,
    const std::size_t &vec_index,
    const std::size_t &com_index)
{
    return com_index * n_vecs + vec_index;
}

// Optimized for coalesced access when vec_index is thread index.
template <std::size_t VEC_SIZE>
__device__ __host__ inline std::size_t get_3d_index(
    const std::size_t &n_mats,
    const std::size_t &mat_index,
    const std::size_t &vec_index,
    const std::size_t &com_index)
{
    return (mat_index * VEC_SIZE + com_index) * n_mats + vec_index;
}

template <typename T>
cudaError_t copy_to_device_(std::vector<T> &host_data, CudaPtr<T[]> &device_data)
{
    if (host_data.empty())
    {
        return cudaSuccess;
    }

    cudaError_t error = cudaMemcpy(
        device_data.get(),
        host_data.data(),
        host_data.size() * sizeof(T),
        cudaMemcpyHostToDevice);
    return error;
}

template<>
inline cudaError_t copy_to_device_<bool>(std::vector<bool> &host_data, CudaPtr<bool[]> &device_data)
{
    if (host_data.empty())
    {
        return cudaSuccess;
    }

    std::vector<unsigned char> buffer(host_data.size());

    for(std::size_t i = 0; i < host_data.size(); ++i)
    {
        buffer[i] = host_data[i] ? 1 : 0;
    }

    cudaError_t error = cudaMemcpy(
        device_data.get(),
        buffer.data(),
        buffer.size() * sizeof(unsigned char),
        cudaMemcpyHostToDevice);
    return error;
}

template <typename T>
cudaError_t copy_to_host_(CudaPtr<T[]> &device_data, std::vector<T> &host_data)
{
    if (host_data.empty())
    {
        return cudaSuccess;
    }

    cudaError_t error = cudaMemcpy(
        host_data.data(),
        device_data.get(),
        host_data.size() * sizeof(T),
        cudaMemcpyDeviceToHost);
    return error;
}

template<>
inline cudaError_t copy_to_host_<bool>(CudaPtr<bool[]> &device_data, std::vector<bool> &host_data)
{
    if (host_data.empty())
    {
        return cudaSuccess;
    }

    std::vector<unsigned char> buffer(host_data.size());

    cudaError_t error = cudaMemcpy(
        buffer.data(),
        device_data.get(),
        buffer.size() * sizeof(unsigned char),
        cudaMemcpyDeviceToHost);

    if (error == cudaSuccess)
    {
        for(std::size_t i = 0; i < host_data.size(); ++i)
        {
            host_data[i] = buffer[i] != 0;
        }
    }

    return error;
}