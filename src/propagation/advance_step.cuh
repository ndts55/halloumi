#pragma once
#include <cuda_runtime.h>
#include "core/types.cuh"

__global__ void advance_step(
    // Input data
    const DeviceDerivativesTensor d_states,
    const DeviceFloatArray end_epochs,
    const DeviceFloatArray start_epochs,
    // Output data
    DeviceStatesMatrix states,
    DeviceFloatArray next_dts,
    DeviceFloatArray last_dts,
    DeviceBoolArray termination_flags,
    DeviceFloatArray epochs);