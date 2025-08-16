#pragma once
#include <cuda_runtime.h>
#include "cuda/vec.cuh"
#include "core/types.cuh"
#include "simulation/tableau.cuh"
#include "cuda/device_array.cuh"
#include "simulation/environment/ephemeris.cuh"
#include "simulation/environment/constants.cuh"

__device__ inline VelocityVector two_body(const PositionVector &position_delta, const Float &gm)
{
    return position_delta * -gm * position_delta.reciprocal_cubed_norm();
}

__device__ inline VelocityVector three_body_barycentric(const PositionVector &source_position, const PositionVector &body_position, const Float &gm)
{
    return two_body(source_position - body_position, gm);
}

__device__ inline VelocityVector three_body_non_barycentric(const PositionVector &source_position, const PositionVector &body_position, const Float &gm)
{
    return three_body_barycentric(source_position, body_position, gm) + two_body(body_position, gm);
}

__device__ StateVector calculate_current_state(
    // State data
    const DeviceStatesMatrix &states,
    const DeviceDerivativesTensor &d_states,
    // Computation coordinates
    const CudaIndex &index,
    const int &stage,
    const Float &dt)
{
    StateVector state{0.0};

    // sum intermediate d_states up to stage
    for (auto st = 0; st < stage; ++st)
    {
        Float coefficient = RKF78::coefficient(stage, st);
        state += d_states.vector_at(st, index) * coefficient;
    }

    // add the current state
    // state = states.at(index) + dt * state
    state *= dt;
    state += states.vector_at(index);

    return state;
}

__device__ StateVector calculate_state_derivative(
    // Primary inputes
    const StateVector &state,
    const Float &t,
    // Physics configs
    const Integer &center_of_integration,
    const DeviceIntegerArray &active_bodies,
    const DeviceConstants &constants,
    const DeviceEphemeris &ephemeris)
{
    const PositionVector state_position = state.slice<POSITION_OFFSET, POSITION_DIM>();

    // velocity delta, i.e., acceleration
    VelocityVector velocity_delta{0.0};
    if (center_of_integration == 0)
    {
        for (auto i = 0; i < active_bodies.n_vecs; ++i)
        {
            const Integer target = active_bodies.at(i);
            const PositionVector body_position = ephemeris.calculate_position(t, target, center_of_integration);
            velocity_delta += three_body_barycentric(state_position, body_position, constants.gm_for(target));
        }
    }
    else
    {
        for (auto i = 0; i < active_bodies.n_vecs; ++i)
        {
            const Integer target = active_bodies.at(i);
            if (target != center_of_integration)
            {
                const PositionVector body_position = ephemeris.calculate_position(t, target, center_of_integration);
                velocity_delta += three_body_non_barycentric(state_position, body_position, constants.gm_for(target));
            }
            else
            {
                velocity_delta += two_body(state_position, constants.gm_for(target));
            }
        }
    }
    // Velocity of previous state becomes position delta
    return state.slice<VELOCITY_OFFSET, VELOCITY_DIM>().append(velocity_delta);
}

__device__ StateVector calculate_componentwise_truncation_error(const DeviceDerivativesTensor &d_states, const CudaIndex &index)
{
    StateVector sum{0.0};
    for (auto stage = 0; stage < RKF78::NStages; ++stage)
    {
        sum += d_states.vector_at(stage, index) * RKF78::embedded_weight(stage);
    }

    return sum;
}

__device__ Float clamp_dt(const Float &dt)
{
    return min(device_rkf_parameters.max_dt_scale, max(device_rkf_parameters.min_dt_scale, dt));
}

__device__ StateVector calculate_final_state_derivative(const DeviceDerivativesTensor d_states, const CudaIndex &index)
{
    StateVector sum{0.0};

    for (auto stage = 0; stage < RKF78::NStages; ++stage)
    {
        sum += d_states.vector_at(stage, index) * RKF78::weight(stage);
    }
    return sum;
}
