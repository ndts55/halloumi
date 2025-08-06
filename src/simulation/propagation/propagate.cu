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

#pragma region Math and Physics Helpers

__device__ Vec<Float, STATE_DIM> calculate_current_state(
    const int &stage,
    const DeviceArray2D<Float, STATE_DIM> &states,
    const DeviceArray3D<Float, STATE_DIM, RKF78::NStages> &d_states,
    const CudaIndex &index,
    const Float &dt)
{
    Vec<Float, STATE_DIM> state = {0.0};

    // sum intermediate d_states up to stage
    for (auto st = 0; st < stage; ++st)
    {
        auto coefficient = RKF78::coefficient(stage, st);
        // state += coefficient * d_states.at(st, index)
        for (auto dim = 0; dim < STATE_DIM; ++dim)
        {
            state[dim] += coefficient * d_states.at(dim, st, index);
        }
    }

    // add the current state
    // state = states.at(index) + dt * state
    for (auto dim = 0; dim < STATE_DIM; ++dim)
    {
        state[dim] *= dt;
        state[dim] += states.at(dim, index);
    }
    return state;
}

__device__ Vec<Float, STATE_DIM> calculate_state_delta(
    const DeviceConstants &constants,
    const DeviceEphemeris &ephemeris,
    DeviceArray1D<Integer> &active_bodies,
    const Integer &center_of_integration,
    const Integer &t,
    const Vec<Float, STATE_DIM> &state)
{
    const auto state_position = state.slice<POSITION_OFFSET, POSITION_DIM>();

    // velocity delta, i.e., acceleration
    Vec<Float, VELOCITY_DIM> velocity_delta{0.0};
    if (center_of_integration == 0)
    {
        for (const auto target : active_bodies)
        {
            auto body_position = ephemeris.calculate_position(t, target, center_of_integration);
            velocity_delta += three_body_barycentric(state_position, body_position, constants.gm_for(target));
        }
    }
    else
    {
        for (const auto target : active_bodies)
        {
            if (target != center_of_integration)
            {
                auto body_position = ephemeris.calculate_position(t, target, center_of_integration);
                velocity_delta += three_body_non_barycentric(state_position, body_position, constants.gm_for(target));
            }
            else
            {
                auto gm = constants.gm_for(target);
                velocity_delta += two_body(state_position, gm);
            }
        }
    }
    // Velocity of previous state becomes position delta
    return state.slice<VELOCITY_OFFSET, VELOCITY_DIM>().append(velocity_delta);
}

#pragma endregion

#pragma region Kernels

__global__ void prepare_simulation_run(
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

__global__ void evaluate_ode(
    DeviceArray1D<Float> epochs,
    DeviceArray2D<Float, STATE_DIM> states,
    DeviceArray3D<Float, STATE_DIM, RKF78::NStages> d_states,
    const DeviceArray1D<Float> next_dts,
    const DeviceArray1D<bool> termination_flags,
    const Integer center_of_integration,
    DeviceArray1D<Integer> active_bodies,
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

    // TODO optimization for stage = 0 where no call to calculate_current_state or RKF78::node is needed
    for (auto stage = 0; stage < RKF78::NStages; ++stage)
    {
        const auto current_state = calculate_current_state(stage, states, d_states, index, dt);
        auto state_delta = calculate_state_delta(
            constants,
            ephemeris,
            active_bodies,
            center_of_integration,
            /* t */ epoch + RKF78::node(stage) * dt,
            current_state);
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
    // prepare for continuation
    prepare_simulation_run<<<gs, bs>>>(
        simulation.propagation_context.propagation_state.terminated.get(),
        simulation.propagation_context.propagation_state.simulation_ended.get(),
        simulation.propagation_context.propagation_state.backwards.get(),
        simulation.propagation_context.propagation_state.dt_next.get(),
        simulation.propagation_context.samples_data.end_epochs.get(),
        simulation.propagation_context.samples_data.start_epochs.get());

    // set up bool reduction buffer for termination flag kernel
    CudaArray1D<bool> reduction_buffer(n, false);
    check_cuda_error(reduction_buffer.prefetch());

    const auto coi = simulation.propagation_context.samples_data.center_of_integration;

    for (auto step = 0; step < simulation.rkf_parameters.max_steps; ++step)
    {
        // TODO compute dstates of each sample
        // TODO advance steps using dstates, set termination flag for each sample
        // TODO check for termination condition on all samples
    }

    // TODO retrieve final results and return them
}

#pragma endregion