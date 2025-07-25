#include <iostream>
#include <optional>
#include "ephemeris.hpp"
#include "propagation_context.hpp"
#include "utils.hpp"
#include "simulation.hpp"
#include <cuda_runtime.h>
#include "kernels.cuh"
#include "constants.cuh"
#include "device_ephemeris.cuh"

void check_cuda_error(cudaError_t error, const std::string &message = "CUDA error")
{
    if (error != cudaSuccess)
    {
        throw std::runtime_error(message + ": " + cudaGetErrorString(error));
    }
}

void prepare_device_mem(const Simulation &simulation)
{
    check_cuda_error(
        cudaMemcpyToSymbolAsync(
            static_cast<const void *>(&device_rkf_parameters),
            &simulation.rkf_parameters,
            sizeof(RKFParameters),
            0, // offset
            cudaMemcpyHostToDevice),
        "Failed to copy RKF parameters to device");

    DeviceEphemeris de(simulation.ephemeris);
    check_cuda_error(
        cudaMemcpyToSymbolAsync(
            static_cast<const void *>(&device_ephemeris),
            &de,
            sizeof(DeviceEphemeris),
            0,
            cudaMemcpyHostToDevice),
        "Failed to copy DeviceEphemeris to device");

    check_cuda_error(simulation.ephemeris.data.prefetch());
    check_cuda_error(simulation.ephemeris.integers.prefetch());
    check_cuda_error(simulation.ephemeris.floats.prefetch());
    check_cuda_error(simulation.propagation_context.propagation_state.states.prefetch());
    check_cuda_error(simulation.propagation_context.propagation_state.epochs.prefetch());
    check_cuda_error(simulation.propagation_context.propagation_state.terminated.prefetch());
    check_cuda_error(simulation.propagation_context.propagation_state.dt_last.prefetch());
    check_cuda_error(simulation.propagation_context.propagation_state.dt_next.prefetch());
    check_cuda_error(simulation.propagation_context.samples_data.end_epochs.prefetch());
    check_cuda_error(simulation.propagation_context.samples_data.start_epochs.prefetch());

    check_cuda_error(cudaDeviceSynchronize());

    std::cout << "Successfully set up device" << std::endl;
}

int main()
{
    // TODO take config file as command line argument
    auto configuration = json_from_file("acceptance/acceptance.test.5-days.json");
    auto simulation = Simulation::from_json(configuration);

    std::cout << "Read ephemeris\n"
              << "Bodies: "
              << simulation.ephemeris.n_bodies() << std::endl;
    std::cout << "Read propagation context\n"
              << "States: "
              << simulation.propagation_context.samples_data.n_vecs << std::endl;
    std::cout << "Read RKF parameters\n"
              << "> abs tol: " << simulation.rkf_parameters.abs_tol << "\n"
              << "> rel tol: " << simulation.rkf_parameters.rel_tol << "\n"
              << "> initial time step: " << simulation.rkf_parameters.initial_time_step << "\n"
              << "> min time step: " << simulation.rkf_parameters.min_time_step << "\n"
              << "> max steps: " << simulation.rkf_parameters.max_steps << std::endl;

    prepare_device_mem(simulation);

    // Launch test kernel to verify device memory setup
    std::cout << "Launching test kernel to verify device memory setup..." << std::endl;
    launch_test_device_constants();
    check_cuda_error(cudaDeviceSynchronize(), "Error synchronizing after test kernel");
    check_cuda_error(cudaGetLastError(), "Error launching test kernel");
    std::cout << "Test kernel completed successfully" << std::endl;

    // TODO ensure coalesced access to data in global memory
    // TODO implement kernel prepare_for_continuation
    // TODO implement kernel for ode evaluation (all stages)
    // TODO implement kernel for step advancement (adaptive step size)
    // TODO implement kernel for determining whether all samples have finished
    // TODO ensure simulation is stopped if number of steps exceed given value
    // TODO run the simulation on CUDA device
    // TODO get output data from device
    // TODO write output data to some output file for easy comparison
    // TODO implement mathematical model
    // TODO implement physical model
    // TODO implement ode evaluation stages as for loop
    // TODO test register count for unrolled / not unrolled for loop
    return 0;
}
