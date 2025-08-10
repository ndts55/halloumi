#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include "core/types.cuh"
#include "cuda/vec.cuh"
#include "simulation/environment/ephemeris.cuh"
#include "simulation/simulation.hpp"
#include "simulation/tableau.cuh"
#include "simulation/rkf_parameters.cuh"

using CudaIndex = unsigned int;

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

void prefetch_rkf_parameters(const RKFParameters &rkf_parameters)
{
    check_cuda_error(initialize_rkf_parameters_on_device(rkf_parameters),
                     "Failed to copy RKF parameters to device");
}

void prefetch_ephemeris(const Ephemeris &ephemeris)
{
    check_cuda_error(ephemeris.data.prefetch());
    check_cuda_error(ephemeris.integers.prefetch());
    check_cuda_error(ephemeris.floats.prefetch());
}

void prefetch_propagation_context(const PropagationContext &propagation_context)
{
    check_cuda_error(propagation_context.propagation_state.states.prefetch());
    check_cuda_error(propagation_context.propagation_state.epochs.prefetch());
    check_cuda_error(propagation_context.propagation_state.terminated.prefetch());
    check_cuda_error(propagation_context.propagation_state.last_dts.prefetch());
    check_cuda_error(propagation_context.propagation_state.next_dts.prefetch());
    check_cuda_error(propagation_context.propagation_state.simulation_ended.prefetch());
    check_cuda_error(propagation_context.propagation_state.backwards.prefetch());
    check_cuda_error(propagation_context.samples_data.end_epochs.prefetch());
    check_cuda_error(propagation_context.samples_data.start_epochs.prefetch());
}

void prefetch_constants(const Constants &constants)
{
    check_cuda_error(constants.body_ids.prefetch());
    check_cuda_error(constants.gms.prefetch());
}

void prefetch_active_bodies(const ActiveBodies &active_bodies)
{
    check_cuda_error(active_bodies.prefetch());
}

void prepare_device_memory(const Simulation &simulation)
{
    std::cout << "Preparing device memory for simulation..." << std::endl;

    prefetch_rkf_parameters(simulation.rkf_parameters);
    prefetch_ephemeris(simulation.ephemeris);
    prefetch_propagation_context(simulation.propagation_context);
    check_cuda_error(RKF78::initialize_device_tableau(), "Error initializing RKF78 tableau");
    prefetch_constants(simulation.constants);
    prefetch_active_bodies(simulation.active_bodies);

    check_cuda_error(cudaDeviceSynchronize(), "Error synchronizing after prefetching");
    std::cout << "Successfully set up device" << std::endl;
}