#include "propagation/all_terminated.cuh"
#include "propagation/cuda_utils.cuh"

__host__ DeviceBoolArray::value_type all_terminated(
    const DeviceBoolArray &termination_flags,
    DeviceBoolArray &reduction_buffer,
    std::size_t first_grid_size,
    std::size_t block_size)
{
    // TODO move reduction buffer creation into this function
    size_t shared_mem_size = block_size * sizeof(DeviceBoolArray::value_type);

    reduce_bool_with_and<<<first_grid_size, block_size, shared_mem_size>>>(termination_flags, reduction_buffer);
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError(), "first reduction pass on GPU");

    if (first_grid_size > 1)
    {
        auto second_grid_size = grid_size(block_size, first_grid_size);
        reduce_bool_with_and<<<second_grid_size, block_size, shared_mem_size>>>(reduction_buffer, reduction_buffer);
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError(), "second reduction pass on GPU");
    }

    DeviceBoolArray::value_type result;
    check_cuda_error(
        cudaMemcpy(&result, reduction_buffer.data, sizeof(DeviceBoolArray::value_type), cudaMemcpyDeviceToHost),
        "Error copying reduction result from device to host");

    return result;
}

// We assume that the length of termination_flags is less than or equal to the number of threads in the grid.
__global__ void reduce_bool_with_and(const DeviceBoolArray termination_flags, DeviceBoolArray result_buffer)
{
    extern __shared__ bool block_buffer[];

    const CudaIndex local_index = index_in_block();
    const CudaIndex global_index = index_in_grid();
    block_buffer[local_index] = global_index < termination_flags.n_elements ? termination_flags.at(global_index) : true;

    __syncthreads();

    for (auto lim = blockDim.x / 2; lim >= 1; lim /= 2)
    {
        if (local_index < lim)
        {
            block_buffer[local_index] = block_buffer[local_index] && block_buffer[local_index + lim];
        }
        __syncthreads();
    }

    if (local_index == 0)
    {
        result_buffer.at(blockIdx.x) = block_buffer[0];
    }
}
