#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include "core/types.cuh"
#include "cuda/vec.cuh"
#include "simulation/environment/ephemeris.cuh"
#include "simulation/simulation.hpp"
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

std::size_t block_size_from_env()
{
    static std::size_t cached_block_size = 0;
    if (cached_block_size == 0)
    {
        const char *env_block = std::getenv("HALLOUMI_BLOCK_SIZE");
        cached_block_size = env_block ? std::stoi(env_block) : 128;
    }

    return cached_block_size;
}

std::size_t grid_size(std::size_t block_size, std::size_t n_samples)
{
    return (n_samples + block_size - 1) / block_size;
}

void check_cuda_error(cudaError_t error, const std::string &message = "CUDA error")
{
    if (error != cudaSuccess)
    {
        throw std::runtime_error(message + ": " + cudaGetErrorString(error));
    }
}

#pragma region Sync to Device

void prefetch_rkf_parameters(const RKFParameters &rkf_parameters)
{
    check_cuda_error(initialize_rkf_parameters_on_device(rkf_parameters),
                     "Failed to copy RKF parameters to device");
}

void prefetch_ephemeris(const Ephemeris &ephemeris)
{
    check_cuda_error(ephemeris.data.prefetch_to_device());
    check_cuda_error(ephemeris.integers.prefetch_to_device());
    check_cuda_error(ephemeris.floats.prefetch_to_device());
}

void prefetch_propagation_state(const PropagationState &propagation_state)
{
    check_cuda_error(propagation_state.states.prefetch_to_device());
    check_cuda_error(propagation_state.epochs.prefetch_to_device());
    check_cuda_error(propagation_state.terminated.prefetch_to_device());
    check_cuda_error(propagation_state.last_dts.prefetch_to_device());
    check_cuda_error(propagation_state.next_dts.prefetch_to_device());
    check_cuda_error(propagation_state.simulation_ended.prefetch_to_device());
    check_cuda_error(propagation_state.backwards.prefetch_to_device());
}
void prefetch_samples_data(const SamplesData &samples_data)
{
    check_cuda_error(samples_data.end_epochs.prefetch_to_device());
    check_cuda_error(samples_data.start_epochs.prefetch_to_device());
}

void prefetch_constants(const Constants &constants)
{
    check_cuda_error(constants.body_ids.prefetch_to_device());
    check_cuda_error(constants.gms.prefetch_to_device());
}

void prefetch_active_bodies(const ActiveBodies &active_bodies)
{
    check_cuda_error(active_bodies.prefetch_to_device());
}

void prepare_device_memory(const Simulation &simulation)
{
    std::cout << "Preparing device memory for simulation..." << std::endl;

    prefetch_rkf_parameters(simulation.rkf_parameters);
    prefetch_ephemeris(simulation.ephemeris);
    prefetch_propagation_state(simulation.propagation_state);
    prefetch_samples_data(simulation.samples_data);
    check_cuda_error(RKF78::initialize_device_tableau(), "Error initializing RKF78 tableau");
    prefetch_constants(simulation.constants);
    prefetch_active_bodies(simulation.active_bodies);

    check_cuda_error(cudaDeviceSynchronize(), "Error synchronizing after prefetching");
    std::cout << "Successfully set up device" << std::endl;
}

#pragma endregion

#pragma region Sync to Host

void sync_to_host(const Simulation &simulation)
{
    std::cout << "Syncing to host" << std::endl;

    check_cuda_error(simulation.propagation_state.states.prefetch_to_host());
    check_cuda_error(simulation.propagation_state.epochs.prefetch_to_host());
    check_cuda_error(simulation.propagation_state.terminated.prefetch_to_host());
    check_cuda_error(simulation.propagation_state.last_dts.prefetch_to_host());
    check_cuda_error(simulation.propagation_state.next_dts.prefetch_to_host());
    check_cuda_error(simulation.propagation_state.simulation_ended.prefetch_to_host());
    check_cuda_error(simulation.propagation_state.backwards.prefetch_to_host());

    check_cuda_error(cudaDeviceSynchronize(), "Error synchronizing to host");
    std::cout << "Successfully synced to host" << std::endl;
}
