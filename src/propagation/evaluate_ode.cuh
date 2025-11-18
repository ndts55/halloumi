#pragma once
#include <cuda_runtime.h>
#include "core/types.cuh"
#include "simulation/ephemeris.cuh"
#include "simulation/constants.cuh"

__global__ void evaluate_ode(
    // Input data
    const DeviceStatesMatrix states,
    const DeviceFloatArray epochs,
    const DeviceFloatArray next_dts,
    // Output data
    DeviceDerivativesTensor d_states,
    // Control flags
    const DeviceBoolArray termination_flags,
    // Physics configs
    const int center_of_integration,
    const DeviceIntegerArray active_bodies,
    const DeviceConstants constants,
    const DeviceEphemeris ephemeris);