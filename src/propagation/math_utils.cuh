#pragma once
#include <cuda_runtime.h>
#include "core/types.cuh"
#include "simulation/ephemeris.cuh"
#include "simulation/constants.cuh"


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
    const double &dt);


// Calculates the acceleration in the current position using point gravity
__device__ VelocityVector calculate_velocity_derivative(
    // Primary inputes
    const PositionVector &current_position,
    const double &epoch,
    // Physics configs
    const int &center_of_integration,
    const DeviceIntegerArray &active_bodies,
    const DeviceConstants &constants,
    const DeviceEphemeris &ephemeris);


__device__ StateVector calculate_final_state_derivative(const DeviceDerivativesTensor d_states, const CudaIndex &index);
