#pragma once
#include <cuda_runtime.h>
#include "core/types.cuh"

__host__ DeviceBoolArray::value_type all_terminated(
    const DeviceBoolArray &termination_flags,
    DeviceBoolArray &reduction_buffer,
    std::size_t first_grid_size,
    std::size_t block_size);

__global__ void reduce_bool_with_and(const DeviceBoolArray termination_flags, DeviceBoolArray result_buffer);
