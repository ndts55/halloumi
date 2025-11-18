#include "propagation/advance_step.cuh"
#include "propagation/cuda_utils.cuh"
#include "propagation/math_utils.cuh"
#include "simulation/rkf_parameters.cuh"
#include "propagation/time_step_criterion.cuh"

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