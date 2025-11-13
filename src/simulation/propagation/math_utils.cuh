#pragma once
#include <cuda_runtime.h>
#include "cuda/vec.cuh"
#include "core/types.cuh"
#include "simulation/tableau.cuh"
#include "simulation/environment/ephemeris.cuh"
#include "simulation/environment/constants.cuh"
#include "simulation/rkf_parameters.cuh"

__device__ inline VelocityVector two_body(const PositionVector &position_delta, const double &gm)
{
    return position_delta * -gm * position_delta.reciprocal_cubed_norm();
}

__device__ inline VelocityVector three_body_barycentric(const PositionVector &source_position, const PositionVector &body_position, const double &gm)
{
    return two_body(source_position - body_position, gm);
}

__device__ inline VelocityVector three_body_non_barycentric(const PositionVector &source_position, const PositionVector &body_position, const double &gm)
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
    const double &dt)
{
    StateVector state{0.0};

    // sum intermediate d_states up to stage
    for (auto st = 0; st < stage; ++st)
    {
        double coefficient = RKF78::coefficient(stage, st);
        state += d_states.vector_at(st, index) * coefficient;
    }

    // add the current state
    // state = states.at(index) + dt * state
    state *= dt;
    state += states.vector_at(index);

    return state;
}

// Calculates the acceleration in the current position using point gravity
__device__ VelocityVector calculate_velocity_derivative(
    // Primary inputes
    const PositionVector &current_position,
    const double &epoch,
    // Physics configs
    const int &center_of_integration,
    const DeviceIntegerArray &active_bodies,
    const DeviceConstants &constants,
    const DeviceEphemeris &ephemeris)
{
    // velocity derivative, i.e., acceleration
    VelocityVector velocity_derivative{0.0};
    if (center_of_integration == 0)
    {
        for (std::size_t i = 0; i < active_bodies.n_elements; ++i)
        {
            const int target = active_bodies.at(i);
            const PositionVector body_position = ephemeris.calculate_position(epoch, target, center_of_integration);
            velocity_derivative += three_body_barycentric(current_position, body_position, constants.gm_for(target));
        }
    }
    else
    {
        for (std::size_t i = 0; i < active_bodies.n_elements; ++i)
        {
            const int target = active_bodies.at(i);
            if (target != center_of_integration)
            {
                const PositionVector body_position = ephemeris.calculate_position(epoch, target, center_of_integration);
                velocity_derivative += three_body_non_barycentric(current_position, body_position, constants.gm_for(target));
            }
            else
            {
                velocity_derivative += two_body(current_position, constants.gm_for(target));
            }
        }
    }
    return velocity_derivative;
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
