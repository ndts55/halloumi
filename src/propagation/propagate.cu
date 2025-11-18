#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <nlohmann/json.hpp>
#include "core/types.cuh"
#include "core/vec.cuh"
#include "simulation/ephemeris.cuh"
#include "propagation/propagate.cuh"
#include "simulation/simulation.cuh"
#include "simulation/tableau.cuh"
#include "simulation/rkf_parameters.cuh"
#include "propagation/cuda_utils.cuh"
#include "propagation/math_utils.cuh"
#include "propagation/time_step_criterion.cuh"
#include "utils.cuh"
#include <assert.h>
#include "propagation/all_terminated.cuh"
#include "propagation/prepare_simulation_run.cuh"

__global__ void prepare_simulation_run_test_kernel(
    const DeviceBoolArray termination_flags,
    const DeviceBoolArray simulation_ended,
    const DeviceBoolArray backwards,
    const DeviceFloatArray next_dts)
{
    const CudaIndex i = index_in_grid();
    if (i >= termination_flags.n_elements)
    {
        return;
    }

    assert(!termination_flags.at(i) && !simulation_ended.at(i) && !backwards.at(i) && (next_dts.at(i) == device_rkf_parameters.initial_time_step));
}

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
    const DeviceEphemeris ephemeris)
{
    const CudaIndex index = index_in_grid();
    if (index >= termination_flags.n_elements || termination_flags.at(index))
    {
        return;
    }

    const double dt = next_dts.at(index);
    const double epoch = epochs.at(index);

    { // ! Optimization for stage = 0;
        // Simply read out the state from states
        const StateVector current_state = states.vector_at(index);
        const VelocityVector velocity_derivative = calculate_velocity_derivative(
            current_state.slice<POSITION_OFFSET, POSITION_DIM>(),
            epoch, // this is where the optimization happens
            center_of_integration,
            active_bodies,
            constants,
            ephemeris);
        const StateVector state_derivative = current_state.slice<VELOCITY_OFFSET, VELOCITY_DIM>().append(velocity_derivative);
        d_states.set_vector_at(0, index, state_derivative);
    }

    // ! Starts at 1 due to optimization above
#pragma unroll
    for (auto stage = 1; stage < RKF78::NStages; ++stage)
    {
        const StateVector current_state = calculate_current_state(states, d_states, index, stage, dt);
        const PositionVector current_position = current_state.slice<POSITION_OFFSET, POSITION_DIM>();
        const VelocityVector velocity_derivative = calculate_velocity_derivative(
            current_position,
            /* epoch */ epoch + RKF78::node(stage) * dt,
            center_of_integration,
            active_bodies,
            constants,
            ephemeris);
        const StateVector state_derivative = current_state.slice<VELOCITY_OFFSET, VELOCITY_DIM>().append(velocity_derivative);
        d_states.set_vector_at(stage, index, state_derivative);
    }
}

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
    DeviceFloatArray epochs)
{
    const CudaIndex index = index_in_grid();

    if (index >= states.n_vecs || termination_flags.at(index))
    {
        return;
    }

    const double dt = next_dts.at(index);
    TimeStepCriterion criterion{};
    criterion.current_dt = dt;
    criterion.next_dt = device_rkf_parameters.max_dt_scale * dt;

    StateVector current_state_derivative = calculate_final_state_derivative(d_states, index);
    StateVector current_state = states.vector_at(index);
    StateVector next_state = current_state + (current_state_derivative * dt);

    criterion.evaluate_error(
        dt,
        current_state_derivative,
        current_state,
        next_state,
        d_states,
        index);
    // FIXME dt now is wrong after this call

    // check for end of simulation
    // if (!criterion.terminate && !criterion.reject)
    // {
    //     // FIXME the code that is actually executed does not do this! It looks at some events
    //     criterion.evaluate_simulation_end(
    //         criterion.current_dt,
    //         criterion.next_dt,
    //         epochs.at(index),
    //         start_epochs.at(index),
    //         end_epochs.at(index));
    // }

    // TODO evaluate adapting events

    if (!criterion.terminate && !criterion.refine)
    {
        // if next time step would be too small just terminate the sample
        criterion.terminate = (criterion.reject ? abs(criterion.current_dt) : abs(criterion.next_dt)) < device_rkf_parameters.min_time_step;
    }

    // Set termination flag, we already know what it ought to be
    termination_flags.at(index) = criterion.terminate;

    if (criterion.reject)
    {
        // reject the current time step
        // results are discarded and re-evaluated with shorter dt
        next_dts.at(index) = criterion.current_dt;
    }
    else
    {
        // no rejection, no termination
        // advance
        epochs.at(index) += dt;
        states.set_vector_at(index, next_state);
        last_dts.at(index) = dt;
        if (!criterion.terminate)
        {
            next_dts.at(index) = criterion.next_dt;
        }
    }
}



void dump_d_states(const HostDerivativesTensor &d_states, const std::string &filename = "d_states.json")
{
    auto array = nlohmann::json::array();
    for (auto index = 0; index < d_states.n_mats(); ++index)
    {
        auto states = nlohmann::json::array();
        for (auto stage = 0; stage < RKF78::NStages; ++stage)
        {
            auto state = nlohmann::json::array();
            for (auto dim = 0; dim < STATE_DIM; ++dim)
            {
                state.push_back(d_states.at(stage, index, dim));
            }
            states.push_back(state);
        }
        auto sample = nlohmann::json::object();
        sample["index"] = index;
        sample["states"] = states;
        array.push_back(sample);
    }

    json_to_file(array, filename);
}

void dump_states(const HostStatesMatrix &states, const std::string &filename = "states.json")
{
    auto array = nlohmann::json::array();
    for (auto index = 0; index < states.n_vecs(); ++index)
    {
        nlohmann::json state = nlohmann::json::array();
        for (auto dim = 0; dim < STATE_DIM; ++dim)
        {
            state.push_back(states.at(index, dim));
        }
        array.push_back(state);
    }
    json_to_file(array, filename);
}

template <typename T>
void dump_array(const HostArray<T> &array, const std::string &filename)
{
    auto json_array = nlohmann::json::array();
    for (std::size_t i = 0; i < array.size(); ++i)
    {
        json_array.push_back(array.at(i));
    }
    json_to_file(json_array, filename);
}

__host__ void propagate(Simulation &simulation)
{
    // figure out grid size and block size
    auto n = simulation.n_samples();
    auto bs = 64; // block_size_from_env();
    auto gs = grid_size(bs, n);

#ifndef NDEBUG
    // TODO move this to a unit test
    auto tf0 = HostBoolArray(n, 1);
    tf0.copy_to_device();
    auto buffer = HostBoolArray(n, 0);
    buffer.copy_to_device();
    cudaDeviceSynchronize();
    auto db = buffer.get();
    auto d0 = tf0.get();
    assert(all_terminated(d0, db, gs, bs));
    buffer.copy_to_device();
    std::vector<bool> vec(n, true);
    vec.at(n - 1) = false;
    auto tf1 = HostBoolArray(std::move(vec));
    tf1.copy_to_device();
    cudaDeviceSynchronize();
    auto d1 = tf1.get();
    assert(!all_terminated(d1, db, gs, bs));
#endif

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

#ifndef NDEBUG
    // TODO move this to a unit test
    prepare_simulation_run_test_kernel<<<gs, bs>>>(
        termination_flags,
        simulation_ended_flags,
        backwards_flags,
        next_dts);
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError(), "prepare simulation run test kernel failed");
#endif

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
