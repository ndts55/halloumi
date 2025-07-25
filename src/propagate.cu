#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include "propagate.cuh"
#include "types.hpp"
#include "simulation.hpp"
#include "constants.cuh"
#include "device_ephemeris.cuh"

#pragma region CUDA Helpers

using CudaIndex = unsigned int;

__device__ CudaIndex index_in_grid()
{
    return blockIdx.x * blockDim.x + threadIdx.x;
}
__device__ CudaIndex index_in_block()
{
    return threadIdx.x;
}

void check_cuda_error(cudaError_t error, const std::string &message = "CUDA error")
{
    if (error != cudaSuccess)
    {
        throw std::runtime_error(message + ": " + cudaGetErrorString(error));
    }
}

void prepare_device_memory(const Simulation &simulation)
{
    std::cout << "Preparing device memory for simulation..." << std::endl;
    check_cuda_error(
        cudaMemcpyToSymbolAsync(
            static_cast<const void *>(&device_rkf_parameters),
            &simulation.rkf_parameters,
            sizeof(RKFParameters),
            0, // offset
            cudaMemcpyHostToDevice),
        "Failed to copy RKF parameters to device");

    DeviceEphemeris de(simulation.ephemeris);
    check_cuda_error(
        cudaMemcpyToSymbolAsync(
            static_cast<const void *>(&device_ephemeris),
            &de,
            sizeof(DeviceEphemeris),
            0,
            cudaMemcpyHostToDevice),
        "Failed to copy DeviceEphemeris to device");

    check_cuda_error(simulation.ephemeris.data.prefetch());
    check_cuda_error(simulation.ephemeris.integers.prefetch());
    check_cuda_error(simulation.ephemeris.floats.prefetch());
    check_cuda_error(simulation.propagation_context.propagation_state.states.prefetch());
    check_cuda_error(simulation.propagation_context.propagation_state.epochs.prefetch());
    check_cuda_error(simulation.propagation_context.propagation_state.terminated.prefetch());
    check_cuda_error(simulation.propagation_context.propagation_state.dt_last.prefetch());
    check_cuda_error(simulation.propagation_context.propagation_state.dt_next.prefetch());
    check_cuda_error(simulation.propagation_context.propagation_state.simulation_ended.prefetch());
    check_cuda_error(simulation.propagation_context.propagation_state.backwards.prefetch());
    check_cuda_error(simulation.propagation_context.samples_data.end_epochs.prefetch());
    check_cuda_error(simulation.propagation_context.samples_data.start_epochs.prefetch());

    check_cuda_error(cudaDeviceSynchronize(), "Error synchronizing after test kernel");
    check_cuda_error(cudaGetLastError(), "Error launching test kernel");

    std::cout << "Successfully set up device" << std::endl;
}

#pragma endregion

#pragma region Kernels

__global__ void setup_simulation_kernel(
    DeviceArray1D<bool> terminated,
    DeviceArray1D<bool> simulation_ended,
    DeviceArray1D<bool> backwards,
    DeviceArray1D<Float> dt_next,
    const DeviceArray1D<Float> end_epochs,
    const DeviceArray1D<Float> start_epochs)
{
    const auto i = index_in_grid();
    if (i >= terminated.n_vecs)
    {
        return;
    }

    if (simulation_ended.at(i))
    {
        terminated.at(i) = false;
        simulation_ended.at(i) = false;
    }

    auto span = end_epochs.at(i) - start_epochs.at(i);
    dt_next.at(i) = copysign(device_rkf_parameters.initial_time_step, span);
    backwards.at(i) = span < 0;
}

// __global__ void advance_step(
//     DeviceArray2D<Float, STATE_DIM> states,
//     DeviceArray1D<Float> next_dts,
//     DeviceArray1D<Float> last_dts,
//     DeviceArray1D<Float> epochs,
//     const DeviceArray1D<Float> end_epochs,
//     DeviceArray1D<bool> termination_flags);

// __global__ void evaluate_ode(
//     DeviceArray1D<Float> epochs,
//     DeviceArray2D<Float, STATE_DIM> states,
//     DeviceArray3D<Float, STATE_DIM, RKF78::NStages> d_states,
//     const DeviceArray1D<Float> next_dts,
//     const DeviceArray1D<bool> termination_flags);

// __global__ void all_terminated(
//     const DeviceArray1D<bool> termination_flags,
//     DeviceArray1D<bool> result_buffer,
//     bool initial_value);

#pragma endregion

#pragma region Propagate

std::size_t block_size()
{
    const char *env_block = std::getenv("HALLOUMI_BLOCK_SIZE");
    return env_block ? std::stoi(env_block) : 128;
}

std::size_t grid_size(std::size_t block_size, std::size_t n_samples)
{
    return (n_samples + block_size - 1) / block_size;
}

__host__ void propagate(Simulation &simulation)
{
    prepare_device_memory(simulation);
    // figure out grid size and block size
    auto n = simulation.n_samples();
    auto bs = block_size();
    auto gs = grid_size(bs, n);
    // prepare for continuation
    setup_simulation_kernel<<<gs, bs>>>(
        simulation.propagation_context.propagation_state.terminated.get(),
        simulation.propagation_context.propagation_state.simulation_ended.get(),
        simulation.propagation_context.propagation_state.backwards.get(),
        simulation.propagation_context.propagation_state.dt_next.get(),
        simulation.propagation_context.samples_data.end_epochs.get(),
        simulation.propagation_context.samples_data.start_epochs.get());
    // TODO set up bool reduction buffer for termination flag kernel
    // TODO integrate until max steps or exit condition is met
    // TODO compute dstates
    // TODO advance steps using dstates
    // TODO check for termination condition
    // TODO retrieve final results and return them
}

#pragma endregion