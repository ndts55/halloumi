#include "propagation/prepare_simulation_run.cuh"
#include "propagation/cuda_utils.cuh"

__global__ void prepare_simulation_run(
    // Input
    const DeviceFloatArray end_epochs,
    const DeviceFloatArray start_epochs,

    // Output
    DeviceBoolArray termination_flags,
    DeviceBoolArray simulation_ended,
    DeviceBoolArray backwards,
    DeviceFloatArray next_dts)
{
    const CudaIndex i = index_in_grid();
    if (i >= termination_flags.n_elements)
    {
        return;
    }

    if (simulation_ended.at(i))
    {
        termination_flags.at(i) = false;
        simulation_ended.at(i) = false;
    }

    const double span = end_epochs.at(i) - start_epochs.at(i);
    next_dts.at(i) = copysign(device_rkf_parameters.initial_time_step, span);
    backwards.at(i) = span < 0;
}