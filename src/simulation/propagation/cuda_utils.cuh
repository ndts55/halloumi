#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include "core/types.cuh"
#include "cuda/vec.cuh"
#include "simulation/environment/ephemeris.cuh"
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

void copy_rkf_parameters_to_device(const RKFParameters &rkf_parameters)
{
    check_cuda_error(initialize_rkf_parameters_on_device(rkf_parameters),
                     "Failed to copy RKF parameters to device");
}

void copy_ephemeris_to_device(Ephemeris &ephemeris)
{
    check_cuda_error(ephemeris.data.copy_to_device());
    check_cuda_error(ephemeris.integers.copy_to_device());
    check_cuda_error(ephemeris.floats.copy_to_device());
}

void copy_propagation_state_to_device(PropagationState &propagation_state)
{
    check_cuda_error(propagation_state.states.copy_to_device());
    check_cuda_error(propagation_state.epochs.copy_to_device());
    check_cuda_error(propagation_state.terminated.copy_to_device());
    check_cuda_error(propagation_state.last_dts.copy_to_device());
    check_cuda_error(propagation_state.next_dts.copy_to_device());
    check_cuda_error(propagation_state.simulation_ended.copy_to_device());
    check_cuda_error(propagation_state.backwards.copy_to_device());
}
void copy_samples_data_to_device(SamplesData &samples_data)
{
    check_cuda_error(samples_data.end_epochs.copy_to_device());
    check_cuda_error(samples_data.start_epochs.copy_to_device());
}

void copy_constants_to_device(Constants &constants)
{
    check_cuda_error(constants.body_ids.copy_to_device());
    check_cuda_error(constants.gms.copy_to_device());
}

void copy_active_bodies_to_device(ActiveBodies &active_bodies)
{
    check_cuda_error(active_bodies.copy_to_device());
}

void sync_to_device(Simulation &simulation)
{
    std::cout << "Preparing device memory for simulation..." << std::endl;

    copy_rkf_parameters_to_device(simulation.rkf_parameters);
    copy_ephemeris_to_device(simulation.ephemeris);
    copy_propagation_state_to_device(simulation.propagation_state);
    copy_samples_data_to_device(simulation.samples_data);
    check_cuda_error(RKF78::initialize_device_tableau(), "Error initializing RKF78 tableau");
    copy_constants_to_device(simulation.constants);
    copy_active_bodies_to_device(simulation.active_bodies);

    check_cuda_error(cudaDeviceSynchronize(), "Error synchronizing after copying");
    std::cout << "Successfully set up device" << std::endl;
}

#pragma endregion

#pragma region Sync to Host

void sync_to_host(Simulation &simulation)
{
    std::cout << "Syncing to host" << std::endl;

    check_cuda_error(simulation.propagation_state.states.copy_to_host());
    check_cuda_error(simulation.propagation_state.epochs.copy_to_host());
    check_cuda_error(simulation.propagation_state.terminated.copy_to_host());
    check_cuda_error(simulation.propagation_state.last_dts.copy_to_host());
    check_cuda_error(simulation.propagation_state.next_dts.copy_to_host());
    check_cuda_error(simulation.propagation_state.simulation_ended.copy_to_host());
    check_cuda_error(simulation.propagation_state.backwards.copy_to_host());

    check_cuda_error(cudaDeviceSynchronize(), "Error synchronizing to host");
    std::cout << "Successfully synced to host" << std::endl;
}
