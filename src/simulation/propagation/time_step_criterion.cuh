#pragma once
#include <cuda_runtime.h>
#include <limits>
#include "core/types.cuh"
#include "cuda/vec.cuh"
#include "simulation/propagation/math_utils.cuh"

__device__ inline StateVector calculate_desired_error_magnitude(
    const StateVector &current_state_derivative,
    const StateVector &current_state,
    const StateVector &next_state,
    const double &dt_value)
{
    const StateVector scaled_states_max = next_state.componentwise_abs().componentwise_max(current_state.componentwise_abs()) * device_rkf_parameters.scale_state;

    const double derivative_scale = dt_value * device_rkf_parameters.scale_dstate;
    const StateVector scaled_current_d_state = current_state_derivative.componentwise_abs() * derivative_scale;

    const StateVector error_scale_base = scaled_states_max + scaled_current_d_state;
    const StateVector relative_error_tolerance = error_scale_base * device_rkf_parameters.rel_tol;

    const StateVector desired_error_magnitude = relative_error_tolerance + device_rkf_parameters.abs_tol;

    return desired_error_magnitude;
}

__device__ StateVector calculate_componentwise_truncation_error(const DeviceDerivativesTensor &d_states, const CudaIndex &index)
{
    StateVector sum{};
    for (std::size_t stage = 0; stage < RKF78::NStages; ++stage)
    {
        sum += d_states.vector_at(stage, index) * RKF78::embedded_weight(stage);
    }

    return sum;
}

__device__ double clamp_dt(const double &dt)
{
    return min(device_rkf_parameters.max_dt_scale, max(device_rkf_parameters.min_dt_scale, dt));
}

struct TimeStepCriterion
{
    bool reject = false;
    bool terminate = false;
    bool refine = false;
    double current_dt = std::numeric_limits<double>::max();
    double next_dt = std::numeric_limits<double>::max();

    __device__ static TimeStepCriterion from_dts(double current_dt, double next_dt)
    {
        TimeStepCriterion criterion;
        criterion.current_dt = current_dt;
        criterion.next_dt = next_dt;
        return criterion;
    }

    __device__ void evaluate_error(
        const double &current_dt,
        const StateVector &current_state_derivative,
        const StateVector &current_state,
        const StateVector &next_state,
        const DeviceDerivativesTensor &d_states,
        const CudaIndex &index)
    {
        const double dt_value = fabs(current_dt);
        const StateVector desired_error_magnitude = calculate_desired_error_magnitude(
            current_state_derivative,
            current_state,
            next_state,
            dt_value);
        // FIXME these values are incorrect
        const StateVector error = calculate_componentwise_truncation_error(d_states, index) * dt_value;
        const double error_ratio = (error / desired_error_magnitude).max_norm();

        TimeStepCriterion criterion{};
        if (error_ratio >= 1)
        {
            const double dt_ratio = device_rkf_parameters.dt_safety * pow(error_ratio, -1. / RKF78::Order);
            criterion.current_dt = current_dt * clamp_dt(dt_ratio);
            criterion.reject = true;
        }
        else
        {
            const double dt_ratio = device_rkf_parameters.dt_safety * pow(error_ratio, -1. / (RKF78::Order + 1));
            criterion.next_dt = current_dt * clamp_dt(dt_ratio);
        }

        this->combine_with(criterion);
    }

    __device__ void evaluate_simulation_end(
        const double &current_dt,
        const double &next_dt,
        const double &current_epoch,
        const double &start_epoch,
        const double &end_epoch)
    {
        bool forward = (end_epoch - start_epoch) > 0;
        const double next_epoch = current_epoch + current_dt;
        TimeStepCriterion criterion{};
        criterion.current_dt = current_dt;
        criterion.next_dt = next_dt;
        constexpr double tolerance = 1e-5;
        if (forward ? (next_epoch > end_epoch + tolerance) : (next_epoch < end_epoch - tolerance))
        {
            // this step is too far
            // reject and update current_dt
            criterion.current_dt = end_epoch - current_epoch;
            criterion.reject = true;
            criterion.refine = true;
        }
        else if (forward ? (next_epoch + next_dt > end_epoch + tolerance) : (next_epoch + next_dt < end_epoch - tolerance))
        {
            // next step is too far
            // update next_dt and refine
            this->next_dt = end_epoch - next_epoch;
            this->refine = true;
        }
        this->combine_with(criterion);
    }

    __device__ inline TimeStepCriterion &operator|=(const TimeStepCriterion &other)
    {
        this->combine_with(other);
        return *this;
    }

    __device__ inline void combine_with(const TimeStepCriterion &other)
    {
        reject |= other.reject;
        terminate |= other.terminate;
        refine |= other.refine;
        current_dt = copysign(min(fabs(current_dt), fabs(other.current_dt)), current_dt);
        next_dt = copysign(min(fabs(next_dt), fabs(other.next_dt)), next_dt);
    }
};