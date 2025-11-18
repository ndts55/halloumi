#include "propagation/propagate.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <nlohmann/json.hpp>
#include "core/types.cuh"
#include "propagation/cuda_utils.cuh"
#include "propagation/all_terminated.cuh"
#include "propagation/prepare_simulation_run.cuh"
#include "propagation/evaluate_ode.cuh"
#include "propagation/advance_step.cuh"

__host__ void propagate(Simulation &simulation)
{
    // figure out grid size and block size
    auto n = simulation.n_samples();
    auto bs = 64; // block_size_from_env();
    auto gs = grid_size(bs, n);

    sync_to_device(simulation);
    // set up bool reduction buffer for termination flag kernel
    HostBoolArray host_reduction_buffer(gs, false); // One entry per block
    check_cuda_error(host_reduction_buffer.copy_to_device());
    HostDerivativesTensor host_d_states(n, 0.0);
    check_cuda_error(host_d_states.copy_to_device());
    check_cuda_error(cudaDeviceSynchronize());

    std::cout << "Preparing arrays" << std::endl;

    // 'get' device arrays
    const auto backwards_flags = simulation.propagation_state.backwards.get();
    const auto end_epochs = simulation.samples_data.end_epochs.get();
    const auto start_epochs = simulation.samples_data.start_epochs.get();

    auto next_dts = simulation.propagation_state.next_dts.get();
    auto last_dts = simulation.propagation_state.last_dts.get();
    auto reduction_buffer = host_reduction_buffer.get();
    auto termination_flags = simulation.propagation_state.terminated.get();
    auto simulation_ended_flags = simulation.propagation_state.simulation_ended.get();

    auto states = simulation.propagation_state.states.get();
    auto epochs = simulation.propagation_state.epochs.get();
    auto d_states = host_d_states.get();
    auto center_of_integration = simulation.samples_data.center_of_integration;
    auto active_bodies = simulation.active_bodies.get();
    auto constants = simulation.constants.get();
    auto ephemeris = simulation.ephemeris.get();

    prepare_simulation_run<<<gs, bs>>>(
        end_epochs,
        start_epochs,
        termination_flags,
        simulation_ended_flags,
        backwards_flags,
        next_dts);
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError(), "prepare simulation run kernel launch failed");

    bool reached_max_steps = true;
    for (auto step = 0; step < simulation.rkf_parameters.max_steps; ++step)
    {
        evaluate_ode<<<gs, bs>>>(
            states,
            epochs,
            next_dts,
            d_states,
            termination_flags,
            center_of_integration,
            active_bodies,
            constants,
            ephemeris);
#ifndef NDEBUG
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError(), "evaluate ode kernel launch failed");
#endif

        advance_step<<<gs, bs>>>(
            d_states,
            end_epochs,
            start_epochs,
            states,
            next_dts,
            last_dts,
            termination_flags,
            epochs);
#ifndef NDEBUG
        cudaDeviceSynchronize();
        check_cuda_error(cudaGetLastError(), "advance step kernel launch failed");
#endif

        if (all_terminated(termination_flags, reduction_buffer, gs, bs))
        {
            std::cout << "All simulations terminated at step " << step << std::endl;
            reached_max_steps = false;
            break;
        }
    }

    if (reached_max_steps)
    {
        std::cout << "Some samples did not terminate within the maximum number of steps" << std::endl;
    }
    else
    {
        std::cout << "Terminated because all samples terminated" << std::endl;
    }

    sync_to_host(simulation);

    cudaDeviceSynchronize();
    check_cuda_error(host_d_states.copy_to_host());
    // dump_d_states(host_d_states);
    // dump_states(simulation.propagation_state.states);
    // dump_array(simulation.propagation_state.next_dts, "next_dts.json");

    simulation.propagated = true;
}
