#include <cuda_runtime.h>
#include "cuda/vec.cuh"
#include "core/types.cuh"
#include "simulation/tableau.cuh"
#include "cuda/device_array.cuh"
#include "simulation/environment/ephemeris.cuh"
#include "simulation/environment/constants.cuh"

__device__ inline Vec<Float, VELOCITY_DIM> two_body(const Vec<Float, POSITION_DIM> &position_delta, const Float &gm)
{
    return position_delta * -gm * position_delta.reciprocal_cubed_norm();
}

__device__ inline Vec<Float, VELOCITY_DIM> three_body_barycentric(const Vec<Float, POSITION_DIM> &source_position, const Vec<Float, POSITION_DIM> &body_position, const Float &gm)
{
    return two_body(source_position - body_position, gm);
}

__device__ inline Vec<Float, VELOCITY_DIM> three_body_non_barycentric(const Vec<Float, POSITION_DIM> &source_position, const Vec<Float, POSITION_DIM> &body_position, const Float &gm)
{
    return three_body_barycentric(source_position, body_position, gm) + two_body(body_position, gm);
}

__device__ Vec<Float, STATE_DIM> calculate_current_state(
    // State data
    const DeviceArray2D<Float, STATE_DIM> &states,
    const DeviceArray3D<Float, STATE_DIM, RKF78::NStages> &d_states,
    // Computation coordinates
    const CudaIndex &index,
    const int &stage,
    const Float &dt)
{
    Vec<Float, STATE_DIM> state = {0.0};

    // sum intermediate d_states up to stage
    for (auto st = 0; st < stage; ++st)
    {
        auto coefficient = RKF78::coefficient(stage, st);
        // state += coefficient * d_states.at(st, index)
        for (auto dim = 0; dim < STATE_DIM; ++dim)
        {
            state[dim] += coefficient * d_states.at(dim, st, index);
        }
    }

    // add the current state
    // state = states.at(index) + dt * state
    for (auto dim = 0; dim < STATE_DIM; ++dim)
    {
        state[dim] *= dt;
        state[dim] += states.at(dim, index);
    }
    return state;
}

__device__ Vec<Float, STATE_DIM> calculate_state_delta(
    // Primary inputes
    const Vec<Float, STATE_DIM> &state,
    const Float &t,
    // Physics configs
    const Integer &center_of_integration,
    const DeviceArray1D<Integer> &active_bodies,
    const DeviceConstants &constants,
    const DeviceEphemeris &ephemeris)
{
    const auto state_position = state.slice<POSITION_OFFSET, POSITION_DIM>();

    // velocity delta, i.e., acceleration
    Vec<Float, VELOCITY_DIM> velocity_delta{0.0};
    if (center_of_integration == 0)
    {
        for (auto i = 0; i < active_bodies.n_vecs; ++i)
        {
            const auto target = active_bodies.at(i);
            const auto body_position = ephemeris.calculate_position(t, target, center_of_integration);
            velocity_delta += three_body_barycentric(state_position, body_position, constants.gm_for(target));
        }
    }
    else
    {
        for (auto i = 0; i < active_bodies.n_vecs; ++i)
        {
            const auto target = active_bodies.at(i);
            if (target != center_of_integration)
            {
                const auto body_position = ephemeris.calculate_position(t, target, center_of_integration);
                velocity_delta += three_body_non_barycentric(state_position, body_position, constants.gm_for(target));
            }
            else
            {
                const auto gm = constants.gm_for(target);
                velocity_delta += two_body(state_position, gm);
            }
        }
    }
    // Velocity of previous state becomes position delta
    return state.slice<VELOCITY_OFFSET, VELOCITY_DIM>().append(velocity_delta);
}

__device__ Vec<Float, STATE_DIM> calculate_truncation_error(const DeviceArray3D<Float, STATE_DIM, RKF78::NStages> &d_states, const CudaIndex &index)
{
    Vec<Float, STATE_DIM> sum{0.0};
    for (auto stage = 0; stage < RKF78::NStages; ++stage)
    {
        sum += d_states.vector_at(stage, index) * RKF78::embedded_weight(stage);
    }

    return sum;
}

__device__ Float clamp_dt(const Float &dt)
{
    return min(device_rkf_parameters.max_dt_scale, max(device_rkf_parameters.min_dt_scale, dt));
}

__device__ Vec<Float, STATE_DIM> calculate_final_d_state(const DeviceArray3D<Float, STATE_DIM, RKF78::NStages> d_states, const CudaIndex &index)
{
    Vec<Float, STATE_DIM> sum{0.0};

    for (auto stage = 0; stage < RKF78::NStages; ++stage)
    {
        sum += d_states.vector_at(stage, index) * RKF78::weight(stage);
    }
    return sum;
}
