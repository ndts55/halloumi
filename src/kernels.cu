#include "kernels.cuh"
#include "types.hpp"
#include "device_array.cuh"
#include "rkf_parameters.hpp"
#include <cuda_runtime.h>
#include "constants.cuh"
#include <stdio.h>

#pragma region CUDA Helpers

using CudaIndex = unsigned int;

__device__ CudaIndex index_in_grid()
{
    return blockIdx.x * blockDim.x + threadIdx.x;
}
__device__ CudaIndex index_in_block()
{
    return threadIdx.x;
}

#pragma endregion

__global__ void test_device_constants()
{
    // Only execute on the first thread to avoid multiple prints
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        // Test RKF parameters
        printf("Device RKF parameters verification:\n");
        printf("  - abs_tol: %.10e\n", device_rkf_parameters.abs_tol);
        printf("  - rel_tol: %.10e\n", device_rkf_parameters.rel_tol);
        printf("  - initial_time_step: %.10e\n", device_rkf_parameters.initial_time_step);
        printf("  - min_time_step: %.10e\n", device_rkf_parameters.min_time_step);
        printf("  - max_steps: %llu\n", device_rkf_parameters.max_steps);
        printf("  - dt_safety: %.10e\n", device_rkf_parameters.dt_safety);
        printf("  - min_dt_scale: %.10e\n", device_rkf_parameters.min_dt_scale);
        printf("  - max_dt_scale: %.10e\n", device_rkf_parameters.max_dt_scale);

        // Test DeviceEphemeris - check if pointers are valid
        printf("\nDevice Ephemeris verification:\n");

        // Check integers array - using indices
        if (device_ephemeris.n_bodies() > 0)
        {
            printf("  - First body info:\n");
            printf("    - Frame: %ld\n", device_ephemeris.frame_at(0));
            printf("    - Target: %ld\n", device_ephemeris.target_at(0));
            printf("    - Center: %ld\n", device_ephemeris.center_at(0));
            printf("    - DType: %ld\n", device_ephemeris.dtype_at(0));
            printf("    - NIntervals: %ld\n", device_ephemeris.nintervals_at(0));
            printf("    - PDeg: %ld\n", device_ephemeris.pdeg_at(0));
            printf("    - DataOffset: %ld\n", device_ephemeris.dataoffset_at(0));
            printf("    - DataSize: %ld\n", device_ephemeris.datasize_at(0));

            // Check floats array
            printf("    - Initial Epoch: %.10e\n", device_ephemeris.initial_epoch_at(0));
            printf("    - Final Epoch: %.10e\n", device_ephemeris.final_epoch_at(0));

            // Check first data element if datasize > 0
            if (device_ephemeris.datasize_at(0) > 0)
            {
                printf("    - First Data Element: %.10e\n", device_ephemeris.data_at(device_ephemeris.dataoffset_at(0)));
            }
        }

        printf("\nNumber of bodies in ephemeris: %llu\n", device_ephemeris.n_bodies());
    }
}

__host__ void launch_test_device_constants()
{
    test_device_constants<<<1, 1>>>();
}

__global__ void prepare_for_continuation(
    DeviceArray2D<Float, STATE_DIM> samples)
{
    const auto global_index = index_in_grid();
    if (global_index >= samples.n_vecs)
    {
        return;
    }
    // TODO test kernel that tells me whether device_ephemeris and device_rkf_parameters were copied correctly
    // There seems to be a hidden 1D array called end_of_simulation

    // propagation::detail::prepare<typename ModelT::SamplesT>(i, samples, cfg.initialTimeStep());
    // ->
    // samples.metadata().base().resetEndOfSimulation(i);
    // samples.metadata().base().setInitialStep(i, dt);
    // ->
    // resetEndOfSimulation //
    // if (this->endOfSimulation()[idx]) {
    //     this->terminated()[idx]      = false;
    //     this->endOfSimulation()[idx] = false;
    // }
    // setInitialStep //
    // Real span = this->endEpochs()[idx] - this->startEpochs()[idx];
    // this->nextDts()[idx] = copysign(dt, span);
    // /* Update the backward flag if the sign of the span is negative */
    // this->backward()[idx] = span < 0;
}
