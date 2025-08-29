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
    const Float &dt_value)
{
    const StateVector scaled_states_max = next_state.componentwise_abs().componentwise_max(current_state.componentwise_abs()) * device_rkf_parameters.scale_state;

    const Float derivative_scale = dt_value * device_rkf_parameters.scale_dstate;
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

__device__ Float clamp_dt(const Float &dt)
{
    return min(device_rkf_parameters.max_dt_scale, max(device_rkf_parameters.min_dt_scale, dt));
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
        const DeviceDerivativesTensor &d_states,
        const CudaIndex &index)
    {
        const Float dt_value = fabs(current_dt);
        if (index == 0)
        {
            printf("    > dt value: %.15e\n", dt_value);
        }
        const StateVector desired_error_magnitude = calculate_desired_error_magnitude(
            current_state_derivative,
            current_state,
            next_state,
            dt_value);
        if (index == 0)
        {
            printf("    > desired: [%.15e, %.15e, %.15e, %.15e, %.15e, %.15e]\n",
                   desired_error_magnitude[0],
                   desired_error_magnitude[1],
                   desired_error_magnitude[2],
                   desired_error_magnitude[3],
                   desired_error_magnitude[4],
                   desired_error_magnitude[5]);
        }
        // FIXME these values are incorrect
        const StateVector error = calculate_componentwise_truncation_error(d_states, index) * dt_value;
        const Float error_ratio = (error / desired_error_magnitude).max_norm();
        if (index == 0)
        {
            printf("    > trunc error: [%.15e, %.15e, %.15e, %.15e, %.15e, %.15e]\n",
                   error[0], error[1], error[2], error[3], error[4], error[5]);
            printf("    > error ratio: %.15e\n", error_ratio);
        }

        TimeStepCriterion criterion{};
        if (error_ratio >= 1)
        {
            const Float dt_ratio = device_rkf_parameters.dt_safety * pow(error_ratio, -1. / RKF78::Order);
            criterion.current_dt = current_dt * clamp_dt(dt_ratio);
            criterion.reject = true;
        }
        else
        {
            const Float dt_ratio = device_rkf_parameters.dt_safety * pow(error_ratio, -1. / (RKF78::Order + 1));
            criterion.next_dt = current_dt * clamp_dt(dt_ratio);
        }

        this->combine_with(criterion);
    }

    __device__ void evaluate_simulation_end(
        const Float &current_dt,
        const Float &next_dt,
        const Float &current_epoch,
        const Float &start_epoch,
        const Float &end_epoch)
    {
        bool forward = (end_epoch - start_epoch) > 0;
        const Float next_epoch = current_epoch + current_dt;
        TimeStepCriterion criterion{};
        criterion.current_dt = current_dt;
        criterion.next_dt = next_dt;
        constexpr Float tolerance = 1e-5;
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