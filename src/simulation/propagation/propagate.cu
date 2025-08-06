#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include "core/types.cuh"
#include "cuda/vec.cuh"
#include "simulation/environment/ephemeris.cuh"
#include "simulation/propagation/propagate.cuh"
#include "simulation/simulation.hpp"
#include "simulation/tableau.cuh"
#include "simulation/rkf_parameters.cuh"
#include "simulation/propagation/cuda_utils.cuh"
#include "simulation/propagation/math_utils.cuh"

#pragma region Kernels

__global__ void prepare_simulation_run(
    // Input
    const DeviceArray1D<Float> end_epochs,
    const DeviceArray1D<Float> start_epochs,

    // Output
    DeviceArray1D<bool> termination_flags,
    DeviceArray1D<bool> simulation_ended,
    DeviceArray1D<bool> backwards,
    DeviceArray1D<Float> dt_next)
{
    const auto i = index_in_grid();
    if (i >= termination_flags.n_vecs)
    {
        return;
    }

    if (simulation_ended.at(i))
    {
        termination_flags.at(i) = false;
        simulation_ended.at(i) = false;
    }

    auto span = end_epochs.at(i) - start_epochs.at(i);
    dt_next.at(i) = copysign(device_rkf_parameters.initial_time_step, span);
    backwards.at(i) = span < 0;
}

__global__ void evaluate_ode(
    // Input data
    const DeviceArray2D<Float, STATE_DIM> states,
    const DeviceArray1D<Float> epochs,
    const DeviceArray1D<Float> next_dts,
    // Output data
    DeviceArray3D<Float, STATE_DIM, RKF78::NStages> d_states,
    // Control flags
    const DeviceArray1D<bool> termination_flags,
    // Physics configs
    const Integer center_of_integration,
    const DeviceArray1D<Integer> active_bodies,
    const DeviceConstants constants,
    const DeviceEphemeris ephemeris)
{
    const auto index = index_in_grid();
    if (index >= termination_flags.n_vecs || termination_flags.at(index))
    {
        return;
    }

    const auto dt = next_dts.at(index);
    const auto epoch = epochs.at(index);

    { // ! Optimization for stage = 0;
        // Simply read out the state from states
        Vec<Float, STATE_DIM> current_state{0.0};
        for (auto dim = 0; dim < STATE_DIM; ++dim)
        {
            current_state[dim] = states.at(dim, index);
        }
        auto state_delta = calculate_state_delta(
            current_state,
            epoch,
            center_of_integration,
            active_bodies,
            constants,
            ephemeris);
        for (auto dim = 0; dim < STATE_DIM; ++dim)
        {
            d_states.at(dim, 0, index) = state_delta[dim];
        }
    }

    // ! Starts at 1 due to optimization above
    for (auto stage = 1; stage < RKF78::NStages; ++stage)
    {
        const auto current_state = calculate_current_state(states, d_states, index, stage, dt);
        auto state_delta = calculate_state_delta(
            current_state,
            /* t */ epoch + RKF78::node(stage) * dt,
            center_of_integration,
            active_bodies,
            constants,
            ephemeris);
        for (auto dim = 0; dim < STATE_DIM; ++dim)
        {
            d_states.at(dim, stage, index) = state_delta[dim];
        }
    }
}

// __global__ void advance_step(
//     DeviceArray2D<Float, STATE_DIM> states,
//     DeviceArray1D<Float> next_dts,
//     DeviceArray1D<Float> last_dts,
//     DeviceArray1D<Float> epochs,
//     const DeviceArray1D<Float> end_epochs,
//     DeviceArray1D<bool> termination_flags);

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

    // set up bool reduction buffer for termination flag kernel
    CudaArray1D<bool> host_reduction_buffer(n, false);
    check_cuda_error(host_reduction_buffer.prefetch());

    // 'get' device arrays
    const auto backwards_flags = simulation.propagation_context.propagation_state.backwards.get();
    const auto end_epochs = simulation.propagation_context.samples_data.end_epochs.get();
    const auto start_epochs = simulation.propagation_context.samples_data.start_epochs.get();

    auto dt_next = simulation.propagation_context.propagation_state.dt_next.get();
    auto reduction_buffer = host_reduction_buffer.get();
    auto termination_flags = simulation.propagation_context.propagation_state.terminated.get();
    auto simulation_ended_flags = simulation.propagation_context.propagation_state.simulation_ended.get();

    // prepare for continuation
    prepare_simulation_run<<<gs, bs>>>(
        end_epochs,
        start_epochs,
        termination_flags,
        simulation_ended_flags,
        backwards_flags,
        dt_next);

    for (auto step = 0; step < simulation.rkf_parameters.max_steps; ++step)
    {
        // TODO compute dstates of each sample
        // TODO advance steps using dstates, set termination flag for each sample
        // TODO check for termination condition on all samples
    }

    // TODO retrieve final results and return them
}

#pragma endregion