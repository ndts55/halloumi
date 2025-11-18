#pragma once
#include <cuda_runtime.h>
#include "core/types.cuh"

__global__ void prepare_simulation_run(
    // Input
    const DeviceFloatArray end_epochs,
    const DeviceFloatArray start_epochs,

    // Output
    DeviceBoolArray termination_flags,
    DeviceBoolArray simulation_ended,
    DeviceBoolArray backwards,
    DeviceFloatArray next_dts);