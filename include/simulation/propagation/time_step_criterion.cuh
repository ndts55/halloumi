#pragma once
#include <cuda_runtime.h>
#include <limits>
#include "core/types.cuh"
#include "cuda/vec.cuh"
#include "cuda/device_array.cuh"
#include "simulation/propagation/math_utils.cuh"

__device__ inline StateVector calculate_desired_error_magnitude(
    const StateVector &current_state_derivative,
    const StateVector &current_state,
    const StateVector &next_state,
    const Float &dt_value)
{
    auto scaled_states_max = next_state.componentwise_abs().componentwise_max(current_state.componentwise_abs()) * device_rkf_parameters.scale_state;

    auto scaled_current_d_state = current_state_derivative.componentwise_abs() * device_rkf_parameters.scale_dstate * dt_value;

    auto error_scale_base = scaled_states_max + scaled_current_d_state;
    auto relative_error_tolerance = error_scale_base * device_rkf_parameters.rel_tol;

    auto desired_error_magnitude = relative_error_tolerance + device_rkf_parameters.abs_tol;

    return desired_error_magnitude;
}

struct TimeStepCriterion
{
    bool reject = false;
    bool terminate = false;
    bool refine = false;
    Float current_dt = std::numeric_limits<Float>::max();
    Float next_dt = std::numeric_limits<Float>::max();

    __device__ static TimeStepCriterion from_dts(Float current_dt, Float next_dt)
    {
        TimeStepCriterion criterion;
        criterion.current_dt = current_dt;
        criterion.next_dt = next_dt;
        return criterion;
    }

    __device__ void evaluate_error(
        const Float &current_dt,
        const StateVector &current_state_derivative,
        const StateVector &current_state,
        const StateVector &next_state,
        const DerivativesDeviceTensor &d_states,
        const CudaIndex &index)
    {
        const Float dt_value = fabs(current_dt);
        const StateVector desired_error_magnitude = calculate_desired_error_magnitude(
            current_state_derivative,
            current_state,
            next_state,
            dt_value);
        const StateVector error = calculate_componentwise_truncation_error(d_states, index) * dt_value;
        const Float error_ratio = (error / desired_error_magnitude).max_norm();

        if (error_ratio >= 1)
        {
            const Float dt_ratio = device_rkf_parameters.dt_safety * pow(error_ratio, -1.0 / RKF78::Order);
            this->current_dt = current_dt * clamp_dt(dt_ratio);
            this->reject = true;
        }
        else
        {
            const Float dt_ratio = device_rkf_parameters.dt_safety * pow(error_ratio, -1.0 / (RKF78::Order + 1));
            this->next_dt = current_dt * clamp_dt(dt_ratio);
        }
    }

    __device__ void evaluate_simulation_end(
        const Float &current_dt,
        const Float &next_dt,
        const Float &current_epoch,
        const Float &start_epoch,
        const Float &end_epoch)
    {
        auto forward = (end_epoch - start_epoch) > 0;
        const Float next_epoch = current_epoch + current_dt;
        this->current_dt = current_dt;
        this->next_dt = next_dt;
        constexpr Float tolerance = 1e-5;
        if (forward ? (next_epoch > end_epoch + tolerance) : (next_epoch < end_epoch - tolerance))
        {
            // this step is too far
            // reject and update current_dt
            this->current_dt = end_epoch - current_epoch;
            this->reject = true;
            this->refine = true;
        }
        else if (forward ? (next_epoch + next_dt > end_epoch + tolerance) : (next_epoch + next_dt < end_epoch - tolerance))
        {
            // next step is too far
            // update next_dt and refine
            this->next_dt = end_epoch - next_epoch;
            this->refine = true;
        }
    }

    __device__ inline TimeStepCriterion &operator|=(const TimeStepCriterion &other)
    {
        reject |= other.reject;
        terminate |= other.terminate;
        refine |= other.refine;
        current_dt = copysign(min(fabs(current_dt), fabs(other.current_dt)), current_dt);
        next_dt = copysign(min(fabs(next_dt), fabs(other.next_dt)), next_dt);
        return *this;
    }
};