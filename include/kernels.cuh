#pragma once
#include "utils.hpp"
#include "types.hpp"
#include "tableau.hpp"
#include "device_array.cuh"
#include "rkf_parameters.hpp"
#include <cuda_runtime.h>

// Test kernel to verify that device_ephemeris and device_rkf_parameters are correctly copied
__global__ void test_device_constants();

__host__ void launch_test_device_constants();

// TODO Can this name be improved?
__global__ void prepare_for_continuation(DeviceArray2D<Float, STATE_DIM> samples);

__global__ void advance_step(
    DeviceArray2D<Float, STATE_DIM> states,
    DeviceArray1D<Float> next_dts,
    DeviceArray1D<Float> last_dts,
    DeviceArray1D<Float> epochs,
    const DeviceArray1D<Float> end_epochs,
    DeviceArray1D<bool> termination_flags);

__global__ void evaluate_ode(
    DeviceArray1D<Float> epochs,
    DeviceArray2D<Float, STATE_DIM> states,
    DeviceArray3D<Float, STATE_DIM, RKF78::NStages> d_states,
    const DeviceArray1D<Float> next_dts,
    const DeviceArray1D<bool> termination_flags);

__global__ void all_terminated(
    const DeviceArray1D<bool> termination_flags,
    DeviceArray1D<bool> result_buffer,
    bool initial_value);
