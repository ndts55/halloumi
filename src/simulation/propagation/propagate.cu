#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include "core/types.cuh"
#include "cuda/vec.cuh"
#include "simulation/environment/ephemeris.cuh"
#include "simulation/propagation/propagate.cuh"
#include "simulation/simulation.hpp"
#include "simulation/tableau.cuh"
#include "simulation/rkf_parameters.cuh"
#include "simulation/propagation/cuda_utils.cuh"
#include "simulation/propagation/math_utils.cuh"
#include "simulation/propagation/time_step_criterion.cuh"

__global__ void prepare_simulation_run(
    // Input
    const DeviceArray1D<Float> end_epochs,
    const DeviceArray1D<Float> start_epochs,

    // Output
    DeviceArray1D<bool> termination_flags,
    DeviceArray1D<bool> simulation_ended,
    DeviceArray1D<bool> backwards,
    DeviceArray1D<Float> next_dts)
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
    next_dts.at(i) = copysign(device_rkf_parameters.initial_time_step, span);
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

__global__ void advance_step(
    // Input data
    const DeviceArray3D<Float, STATE_DIM, RKF78::NStages> d_states,
    const DeviceArray1D<Float> end_epochs,
    const DeviceArray1D<Float> start_epochs,
    // Output data
    DeviceArray2D<Float, STATE_DIM> states,
    DeviceArray1D<Float> next_dts,
    DeviceArray1D<Float> last_dts,
    DeviceArray1D<bool> termination_flags,
    DeviceArray1D<Float> epochs)
{
    const auto index = index_in_grid();

    if (index >= states.n_vecs || termination_flags.at(index))
    {
        return;
    }

    const auto dt = next_dts.at(index);
    auto criterion = TimeStepCriterion::from_dts(dt, dt * device_rkf_parameters.max_dt_scale);

    auto current_d_state = calculate_final_d_state(d_states, index) * dt + states.vector_at(index);
    Vec<Float, STATE_DIM> next_state = states.vector_at(index) + current_d_state * dt;

    criterion.evaluate_error(
        dt,
        current_d_state,
        states.vector_at(index),
        next_state,
        d_states,
        index);

    // check for end of simulation
    if (!criterion.terminate && !criterion.reject)
    {
        criterion.evaluate_simulation_end(
            criterion.current_dt,
            criterion.next_dt,
            epochs.at(index),
            start_epochs.at(index),
            end_epochs.at(index));
    }

    if (!criterion.terminate && !criterion.refine)
    {
        // if next time step would be too small just terminate the sample
        criterion.terminate = (criterion.reject ? abs(criterion.current_dt) : abs(criterion.next_dt)) < device_rkf_parameters.min_time_step;
    }

    if (criterion.reject)
    {
        // reject the current time step
        // results are discarded and re-evaluated with shorter dt
        next_dts.at(index) = criterion.current_dt;
        termination_flags.at(index) = criterion.terminate;
        return;
    }

    // no rejection, no termination
    // advance
    epochs.at(index) += dt;
    states.set_vector_at(index, next_state);
    last_dts.at(index) = dt;
    termination_flags.at(index) = criterion.terminate;
    if (!criterion.terminate)
    {
        next_dts.at(index) = criterion.next_dt;
    }
}

/* We assume that the length of termination_flags is less than or equal to the number of threads in the grid.
 */
__global__ void reduce_bool_with_and(const DeviceArray1D<bool> termination_flags, DeviceArray1D<bool> result_buffer)
{
    extern __shared__ bool block_buffer[];

    const auto local_index = index_in_block();
    const auto global_index = index_in_grid();
    if (global_index <= termination_flags.n_vecs)
    {
        block_buffer[local_index] = termination_flags.at(global_index);
    }
    else
    {
        // Neutral element for (bool, &&) is true
        block_buffer[local_index] = true;
    }
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

__host__ bool all_terminated(
    const DeviceArray1D<bool> &termination_flags,
    DeviceArray1D<bool> &reduction_buffer,
    std::size_t first_grid_size,
    std::size_t block_size)
{
    size_t shared_mem_size = block_size * sizeof(bool);

    reduce_bool_with_and<<<first_grid_size, block_size, shared_mem_size>>>(termination_flags, reduction_buffer);
    check_cuda_error(cudaDeviceSynchronize(), "first reduction pass on GPU");

    if (first_grid_size > 1)
    {
        auto second_grid_size = grid_size(block_size, first_grid_size);
        reduce_bool_with_and<<<second_grid_size, block_size, shared_mem_size>>>(reduction_buffer, reduction_buffer);
        check_cuda_error(cudaDeviceSynchronize(), "second reduction pass on GPU");
    }

    bool result;
    check_cuda_error(
        cudaMemcpy(&result, reduction_buffer.data, sizeof(bool), cudaMemcpyDeviceToHost),
        "Error copying reduction result from device to host");

    return result;
}

__host__ void propagate(Simulation &simulation)
{
    prepare_device_memory(simulation);
    // figure out grid size and block size
    auto n = simulation.n_samples();
    auto bs = block_size_from_env();
    auto gs = grid_size(bs, n);

    std::cout << "Grid size: " << gs << ", Block size: " << bs << std::endl;

    // set up bool reduction buffer for termination flag kernel
    CudaArray1D<bool> host_reduction_buffer(gs, false); // One entry per block
    check_cuda_error(host_reduction_buffer.prefetch());
    CudaArray3D<Float, STATE_DIM, RKF78::NStages> host_d_states(n);
    check_cuda_error(host_d_states.prefetch());

    // 'get' device arrays
    const auto backwards_flags = simulation.propagation_state.backwards.get();
    const auto end_epochs = simulation.samples_data.end_epochs.get();
    const auto start_epochs = simulation.samples_data.start_epochs.get();

    auto next_dts = simulation.propagation_state.next_dts.get();
    auto last_dts = simulation.propagation_state.last_dts.get();
    auto reduction_buffer = host_reduction_buffer.get();
    auto termination_flags = simulation.propagation_state.terminated.get();
    auto simulation_ended_flags = simulation.propagation_state.simulation_ended.get();

    auto states = simulation.propagation_state.states.get();
    auto epochs = simulation.propagation_state.epochs.get();
    auto d_states = host_d_states.get();
    auto center_of_integration = simulation.samples_data.center_of_integration;
    auto active_bodies = simulation.active_bodies.get();
    auto constants = simulation.constants.get();
    auto ephemeris = simulation.ephemeris.get();

    prepare_simulation_run<<<gs, bs>>>(
        end_epochs,
        start_epochs,
        termination_flags,
        simulation_ended_flags,
        backwards_flags,
        next_dts);
    check_cuda_error(cudaGetLastError(), "prepare simulation run kernel launch failed");

    bool reached_max_steps = true;
    for (auto step = 0; step < simulation.rkf_parameters.max_steps; ++step)
    {
        evaluate_ode<<<gs, bs>>>(
            states,
            epochs,
            next_dts,
            d_states,
            termination_flags,
            center_of_integration,
            active_bodies,
            constants,
            ephemeris);
        check_cuda_error(cudaGetLastError(), "evaluate ode kernel launch failed");

        advance_step<<<gs, bs>>>(
            d_states,
            end_epochs,
            start_epochs,
            states,
            next_dts,
            last_dts,
            termination_flags,
            epochs);
        check_cuda_error(cudaGetLastError(), "advance step kernel launch failed");

        if (all_terminated(termination_flags, reduction_buffer, gs, bs))
        {
            std::cout << "All simulations terminated at step " << step << std::endl;
            reached_max_steps = false;
            break;
        }
    }

    if (reached_max_steps)
    {
        std::cout << "Terminated because the simulation reached max step count" << std::endl;
    }
    else
    {
        std::cout << "Terminated because all samples terminated" << std::endl;
    }

    // TODO retrieve final results and return them
    // TODO find out what is actually being compared in the acceptance tests
}
