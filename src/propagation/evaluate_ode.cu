#include "propagation/evaluate_ode.cuh"
#include "propagation/cuda_utils.cuh"
#include "propagation/math_utils.cuh"
#include "simulation/tableau.cuh"

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