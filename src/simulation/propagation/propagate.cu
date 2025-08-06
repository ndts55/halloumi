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

__device__ Vec<Float, POSITION_DIM> interpolate_type_2_body_to_position(const DeviceEphemeris &eph, const Integer &body_index, const Float &epoch)
{
    auto nintervals = eph.nintervals_at(body_index);
    auto data_offset = eph.dataoffset_at(body_index);
    auto pdeg = eph.pdeg_at(body_index);

    // data = [ ...[other data; (data_offset)], interval radius, ...[intervals; (nintervals)], ...[coefficients; (nintervals * (pdeg + 1))] ]
    auto radius = eph.data_at(data_offset);
    DeviceArray1D<Float> intervals{/* data pointer */ eph.data.data + data_offset + 1, /* size */ (std::size_t)nintervals};
    DeviceArray1D<Float> coefficients{/* data pointer */ intervals.end(), /* size */ (std::size_t)nintervals * (pdeg + 1)};

    std::size_t idx = (epoch - intervals.at(0)) / (2 * radius);
    Float s = (epoch - intervals.at(idx)) / radius - 1.0;
    Vec<Float, POSITION_DIM> position = {0.0};
    Vec<Float, POSITION_DIM> w1 = {0.0};
    Vec<Float, POSITION_DIM> w2 = {0.0};
    Float s2 = 2 * s;
    for (auto i = pdeg; i > 0; --i)
    {
        w2 = w1;
        w1 = position;
        position = (w1 * s2 - w2) + coefficients.at(i * nintervals + idx);
    }
    return (position * s - w1) + coefficients.at(idx);
}

__device__ Vec<Float, POSITION_DIM> read_position(const DeviceEphemeris &eph, const Float &epoch, const Integer &target, const Integer &center)
{
    Vec<Float, POSITION_DIM> position = {0.0};
    if (target == center)
    {
        return position;
    }

    Integer t = target;
    while (t != center)
    {
        auto body_index = eph.index_of_target(t);
        // ! We only have type 2 bodies for now.
        position += interpolate_type_2_body_to_position(eph, body_index, epoch);
        t = eph.center_at(body_index);
    }
    return position;
}

__device__ Vec<Float, POSITION_DIM> calculate_position(const DeviceEphemeris &eph, const Float &epoch, const Integer &target, const Integer &center_of_integration)
{
    auto cc = eph.common_center(target, center_of_integration);
    auto xt = read_position(eph, epoch, target, cc);
    auto xc = read_position(eph, epoch, center_of_integration, cc);
    xt -= xc;
    return xt;
}

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

__device__ Vec<Float, VELOCITY_DIM> calculate_velocity_delta(
    const DeviceConstants &constants,
    const DeviceEphemeris &ephemeris,
    const Integer &epoch,
    DeviceArray1D<Integer> &active_bodies,
    const Integer &center_of_integration,
    const Vec<Float, POSITION_DIM> &state_position)
{
    Vec<Float, VELOCITY_DIM> vec{0.0};
    if (center_of_integration == 0)
    {
        for (const auto target : active_bodies)
        {
            auto body_position = calculate_position(ephemeris, epoch, target, center_of_integration);
            vec += three_body_barycentric(state_position, body_position, constants.gm_for(target));
        }
    }
    else
    {
        for (const auto target : active_bodies)
        {
            if (target != center_of_integration)
            {
                auto body_position = calculate_position(ephemeris, epoch, target, center_of_integration);
                vec += three_body_non_barycentric(state_position, body_position, constants.gm_for(target));
            }
            else
            {
                auto gm = constants.gm_for(target);
                vec += two_body(state_position, gm);
            }
        }
    }
    return vec;
}

__device__ Vec<Float, STATE_DIM> evaluate_ode_for_stage(
    const DeviceConstants &constants,
    const DeviceEphemeris &ephemeris,
    DeviceArray1D<Integer> &active_bodies,
    const Integer &center_of_integration,
    const Integer &t,
    const Vec<Float, STATE_DIM> &state)
{
    // velocity delta, i.e., acceleration
    auto velocity_delta = calculate_velocity_delta(constants, ephemeris, t, active_bodies, center_of_integration, state.slice<POSITION_OFFSET, POSITION_DIM>());
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

    for (auto stage = 0; stage < RKF78::NStages; ++stage)
    {
        const auto current_state = calculate_current_state(stage, states, d_states, index, dt);
        const Float t = epoch + RKF78::node(stage) * dt;
        auto state_delta = evaluate_ode_for_stage(constants, ephemeris, active_bodies, center_of_integration, t, current_state);
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