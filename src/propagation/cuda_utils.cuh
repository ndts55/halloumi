#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include "core/types.cuh"
#include "core/vec.cuh"
#include "simulation/ephemeris.cuh"
#include "simulation/simulation.cuh"
#include "simulation/tableau.cuh"
#include "simulation/rkf_parameters.cuh"

__device__ inline CudaIndex index_in_grid()
{
    return blockIdx.x * blockDim.x + threadIdx.x;
}
__device__ inline CudaIndex index_in_block()
{
    return threadIdx.x;
}

std::size_t block_size_from_env();

std::size_t grid_size(std::size_t block_size, std::size_t n_samples);

void check_cuda_error(cudaError_t error, const std::string &message = "CUDA error");

#pragma region Sync to Device

void copy_rkf_parameters_to_device(const RKFParameters &rkf_parameters);

void copy_ephemeris_to_device(Ephemeris &ephemeris);

void copy_propagation_state_to_device(PropagationState &propagation_state);

void copy_samples_data_to_device(SamplesData &samples_data);

void copy_constants_to_device(Constants &constants);

void copy_active_bodies_to_device(ActiveBodies &active_bodies);

void sync_to_device(Simulation &simulation);

#pragma endregion

#pragma region Sync to Host

void sync_to_host(Simulation &simulation);
