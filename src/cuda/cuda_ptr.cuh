#pragma once
#include <cuda_runtime.h>
#include <memory>
#include <iostream>

struct CudaDeleter
{
    void operator()(void *p) const { cudaFree(p); }
};

template <typename T>
using CudaPtr = std::unique_ptr<T, CudaDeleter>;

template <typename T>
CudaPtr<T[]> make_cuda_array(std::size_t n_elements)
{
    void *pointer = nullptr;
    cudaError_t error = cudaMalloc(&pointer, sizeof(T) * n_elements);
    if (error != cudaSuccess || !pointer)
    {
        throw std::bad_alloc();
    }
    return CudaPtr<T[]>(static_cast<T *>(pointer));
}
