#pragma once
#include <cuda_runtime.h>
#include <vector>
#include "cuda/cuda_ptr.cuh"

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