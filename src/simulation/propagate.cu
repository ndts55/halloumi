#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include "core/types.cuh"
#include "cuda/vec.cuh"
#include "simulation/environment/ephemeris.cuh"
#include "simulation/propagate.cuh"
#include "simulation/simulation.hpp"
#include "simulation/tableau.cuh"
#include "simulation/rkf_parameters.cuh"

#pragma region CUDA Helpers

using CudaIndex = unsigned int;

__device__ CudaIndex index_in_grid()
{
    return blockIdx.x * blockDim.x + threadIdx.x;
}
__device__ CudaIndex index_in_block()
{
    return threadIdx.x;
}

void check_cuda_error(cudaError_t error, const std::string &message = "CUDA error")
{
    if (error != cudaSuccess)
    {
        throw std::runtime_error(message + ": " + cudaGetErrorString(error));
    }
}

void prefetch_rkf_parameters(const RKFParameters &rkf_parameters)
{
    check_cuda_error(initialize_rkf_parameters_on_device(rkf_parameters),
                     "Failed to copy RKF parameters to device");
}

void prefetch_ephemeris(const Ephemeris &ephemeris)
{
    check_cuda_error(ephemeris.data.prefetch());
    check_cuda_error(ephemeris.integers.prefetch());
    check_cuda_error(ephemeris.floats.prefetch());
}

void prefetch_propagation_context(const PropagationContext &propagation_context)
{
    check_cuda_error(propagation_context.propagation_state.states.prefetch());
    check_cuda_error(propagation_context.propagation_state.epochs.prefetch());
    check_cuda_error(propagation_context.propagation_state.terminated.prefetch());
    check_cuda_error(propagation_context.propagation_state.dt_last.prefetch());
    check_cuda_error(propagation_context.propagation_state.dt_next.prefetch());
    check_cuda_error(propagation_context.propagation_state.simulation_ended.prefetch());
    check_cuda_error(propagation_context.propagation_state.backwards.prefetch());
    check_cuda_error(propagation_context.samples_data.end_epochs.prefetch());
    check_cuda_error(propagation_context.samples_data.start_epochs.prefetch());
}

void prefetch_constants(const Constants &constants)
{
    check_cuda_error(constants.body_ids.prefetch());
    check_cuda_error(constants.gms.prefetch());
}

void prefetch_active_bodies(const ActiveBodies &active_bodies)
{
    check_cuda_error(active_bodies.prefetch());
}

void prepare_device_memory(const Simulation &simulation)
{
    std::cout << "Preparing device memory for simulation..." << std::endl;

    prefetch_rkf_parameters(simulation.rkf_parameters);
    prefetch_ephemeris(simulation.ephemeris);
    prefetch_propagation_context(simulation.propagation_context);
    check_cuda_error(RKF78::initialize_device_tableau(), "Error initializing RKF78 tableau");
    prefetch_constants(simulation.constants);
    prefetch_active_bodies(simulation.active_bodies);

    check_cuda_error(cudaDeviceSynchronize(), "Error synchronizing after prefetching");
    std::cout << "Successfully set up device" << std::endl;
}

#pragma endregion

#pragma region Math and Physics Helpers

__device__ inline Float cubed_norm(const Vec<Float, STATE_DIM> &state_vector, const Integer &offset, const Integer &dimensionality)
{
    Float norm = 0.0;
    for (Integer i = 0; i < dimensionality; ++i)
    {
        auto value = state_vector[offset + i];
        norm += value * value;
    }
    return sqrtf(norm);
}

__device__ inline Float reciprocal_cubed_norm(const Vec<Float, STATE_DIM> &state_vector, const Integer &offset, const Integer &dimensionality)
{
    auto n = cubed_norm(state_vector, offset, dimensionality);
    return n != 0.0 ? 1.0 / (n * n * n) : 0.0;
}

__device__ inline Integer target_body_index(const DeviceEphemeris &eph, const Integer &target)
{
    Integer tbc = eph.size();
    for (Integer i = 0; i < tbc; ++i)
    {
        if (eph.target_at(i) == target)
        {
            tbc = i;
            break;
        }
    }
    return tbc;
}

// TODO profile with and without inline
__device__ Integer common_center(const DeviceEphemeris &eph, Integer tc, Integer cc)
{
    if (tc == 0 || cc == 0)
    {
        return 0;
    }

    Integer tcnew, ccnew;

    while (tc != cc && tc != 0 && cc != 0)
    {
        tcnew = eph.center_at(target_body_index(eph, tc));
        if (tcnew == cc)
        {
            return tcnew;
        }
        ccnew = eph.center_at(target_body_index(eph, cc));
        if (ccnew == tc)
        {
            return ccnew;
        }
        tc = tcnew;
        cc = ccnew;
    }
    if (tc == 0 || cc == 0)
    {
        return 0;
    }
    else
    {
        return tc;
    }
}

__device__ Vec<Float, POSITION_DIM> interpolate_type_2_body_to_position(const DeviceEphemeris &eph, const Integer &body_index, const Float &epoch)
{
    auto nintervals = eph.nintervals_at(body_index);
    auto data_offset = eph.dataoffset_at(body_index);
    auto pdeg = eph.pdeg_at(body_index);

    // data = [ ...[other data; (data_offset)], interval radius, ...[intervals; (nintervals)], ...[coefficients; (nintervals * (pdeg + 1))] ]
    auto radius = eph.data_at(data_offset);
    DeviceArray1D<Float> intervals{/* data pointer */ eph.data.data + data_offset + 1, /* size */ (std::size_t)nintervals};
    DeviceArray1D<Float> coefficients{/* data pointer */ intervals.end(), /* size */ (std::size_t)nintervals * (pdeg + 1)};

    std::size_t idx = (epoch - intervals.at(0)) / (2 * radius);
    Float s = (epoch - intervals.at(idx)) / radius - 1.0;
    Vec<Float, POSITION_DIM> position = {0.0};
    Vec<Float, POSITION_DIM> w1 = {0.0};
    Vec<Float, POSITION_DIM> w2 = {0.0};
    Float s2 = 2 * s;
    for (auto i = pdeg; i > 0; --i)
    {
        w2 = w1;
        w1 = position;
        position = (w1 * s2 - w2) + coefficients.at(i * nintervals + idx);
    }
    return (position * s - w1) + coefficients.at(idx);
}

__device__ Vec<Float, POSITION_DIM> read_position(const DeviceEphemeris &eph, const Float &epoch, const Integer &target, const Integer &center)
{
    Vec<Float, POSITION_DIM> position = {0.0};
    if (target == center)
    {
        return position;
    }

    Integer t = target;
    while (t != center)
    {
        auto body_index = target_body_index(eph, t);
        position += interpolate_type_2_body_to_position(eph, body_index, epoch);
        t = eph.center_at(body_index);
    }
    return position;
}

__device__ Vec<Float, POSITION_DIM> calculate_position(const DeviceEphemeris &eph, const Float &epoch, const Integer &target, const Integer &center_of_integration)
{
    auto cc = common_center(eph, target, center_of_integration);
    auto xt = read_position(eph, epoch, target, cc);
    auto xc = read_position(eph, epoch, center_of_integration, cc);
    xt -= xc;
    return xt;
}

#pragma endregion

#pragma region Kernels

__global__ void prepare_simulation_run(
    DeviceArray1D<bool> terminated,
    DeviceArray1D<bool> simulation_ended,
    DeviceArray1D<bool> backwards,
    DeviceArray1D<Float> dt_next,
    const DeviceArray1D<Float> end_epochs,
    const DeviceArray1D<Float> start_epochs)
{
    const auto i = index_in_grid();
    if (i >= terminated.n_vecs)
    {
        return;
    }

    if (simulation_ended.at(i))
    {
        terminated.at(i) = false;
        simulation_ended.at(i) = false;
    }

    auto span = end_epochs.at(i) - start_epochs.at(i);
    dt_next.at(i) = copysign(device_rkf_parameters.initial_time_step, span);
    backwards.at(i) = span < 0;
}

__global__ void evaluate_ode(
    DeviceArray1D<Float> epochs,
    DeviceArray2D<Float, STATE_DIM> states,
    DeviceArray3D<Float, STATE_DIM, RKF78::NStages> d_states,
    const DeviceArray1D<Float> next_dts,
    const DeviceArray1D<bool> termination_flags,
    const Integer center_of_integration,
    DeviceArray1D<Integer> active_bodies,
    const DeviceConstants constants,
    const DeviceEphemeris ephemeris)
{
    const auto index = index_in_grid();
    if (index >= termination_flags.n_vecs || termination_flags.at(index))
    {
        return;
    }

    const auto dt = next_dts.at(index);
    const auto epoch = epochs.at(index);

    for (auto stage = 0; stage < RKF78::NStages; ++stage)
    {
        const auto node_c = RKF78::node(stage);
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

        // TODO calculate the new velocity
        Float next_velocity[POSITION_DIM] = {0.0};
        if (center_of_integration == 0)
        {
            // TODO
            for (const auto target : active_bodies)
            {
                /*
                Vec3R bodyPos = env.ephemeris().getPosition(epoch, target, COI);
                acc += thirdBody<false>(pos, bodyPos, env.constants().body(target).gm());
                    return -gm * (scPos - bodyPos) * (scPos - bodyPos).rCubedNorm() + -gm * bodyPos * bodyPos.rCubedNorm()
                */
                auto body_position = calculate_position(ephemeris, epoch, target, center_of_integration);
            }
        }
        else
        {
            // TODO
            for (const auto target : active_bodies)
            {
                if (target != center_of_integration)
                {
                    /*
                    Vec3R bodyPos = env.ephemeris().getPosition(epoch, target, COI);
                    acc += thirdBody<false>(pos, bodyPos, env.constants().body(target).gm());
                        return -gm * (scPos - bodyPos) * (scPos - bodyPos).rCubedNorm() + -gm * bodyPos * bodyPos.rCubedNorm()
                    */
                    auto body_position = calculate_position(ephemeris, epoch, target, center_of_integration);
                }
                else
                {
                    auto gm = constants.gm_for(target);
                    auto rcn = reciprocal_cubed_norm(state, POSITION_OFFSET, POSITION_DIM);
                    for (auto i = 0; i < POSITION_DIM; ++i)
                    {
                        next_velocity[i] += -gm * state[POSITION_OFFSET + i] * rcn;
                    }
                }
            }
        }

        // dStates.template stage<stage>()[idx] = ode.eval(idx, state, t + c * dt);
        //     Vec3R physeval = physeval_(idx, state, t);
        //         (*this)[idx] = StateT::toPhysicalState(state); // What is `this` in this case?
        //         Real epoch   = StateT::getPhysicalTime(state, t);
        //         return this->physics_.eval(idx, (*this)[idx], epoch, this->getCOI(idx));
        //             Vec3R pos = state.pos();
        //             Vec3R vel = state.vel();
        //             Real t    = epoch;
        //             Vec3R acc = accs().eval(pos, vel, t, COI, this->env());
        //                         Vec3R acc;
        //                         if (COI == 0) {
        //                             for (mSize_t i = 0; i < env.ephemeris().activeBodies().size(); i++) {
        //                                 brie::NaifId target = env.ephemeris().activeBodies()[i];
        //                                 Vec3R bodyPos = env.ephemeris().getPosition(epoch, target, COI);
        //                                 acc += thirdBody<true>(
        //                                     pos, bodyPos, env.constants().body(target).gm());
        //                             }
        //                         } else {
        //                             for (mSize_t i = 0; i < env.ephemeris().activeBodies().size(); i++) {
        //                                 brie::NaifId target = env.ephemeris().activeBodies()[i];
        //                                 if (target != COI) {
        //                                     Vec3R bodyPos = env.ephemeris().getPosition(epoch, target, COI);
        //                                     acc += thirdBody<false>(pos, bodyPos, env.constants().body(target).gm());
        //                                         return -gm * (scPos - bodyPos) * (scPos - bodyPos).rCubedNorm() + -gm * bodyPos * bodyPos.rCubedNorm()
        //                                 } else {
        //                                     acc += twoBody(pos, env.constants().body(target).gm());
        //                                         return -gm * deltaPos * deltaPos.rCubedNorm();
        //                                 }
        //                             }
        //                         }
        //                         return acc;
        //             return acc;
        //     return StateT::template ODEcast<ORDER>(state, physeval);
        //         Vec6R out;
        //         out.head<3>() = state.vel();
        //         out.tail<3>() = physicalAcc;
        //         return out;
        // TODO put results in d_states
    }
}

// __global__ void advance_step(
//     DeviceArray2D<Float, STATE_DIM> states,
//     DeviceArray1D<Float> next_dts,
//     DeviceArray1D<Float> last_dts,
//     DeviceArray1D<Float> epochs,
//     const DeviceArray1D<Float> end_epochs,
//     DeviceArray1D<bool> termination_flags);

// __global__ void all_terminated(
//     const DeviceArray1D<bool> termination_flags,
//     DeviceArray1D<bool> result_buffer,
//     bool initial_value);

#pragma endregion

#pragma region Propagate

std::size_t block_size()
{
    const char *env_block = std::getenv("HALLOUMI_BLOCK_SIZE");
    return env_block ? std::stoi(env_block) : 128;
}

std::size_t grid_size(std::size_t block_size, std::size_t n_samples)
{
    return (n_samples + block_size - 1) / block_size;
}

__host__ void propagate(Simulation &simulation)
{
    prepare_device_memory(simulation);
    // figure out grid size and block size
    auto n = simulation.n_samples();
    auto bs = block_size();
    auto gs = grid_size(bs, n);
    // prepare for continuation
    prepare_simulation_run<<<gs, bs>>>(
        simulation.propagation_context.propagation_state.terminated.get(),
        simulation.propagation_context.propagation_state.simulation_ended.get(),
        simulation.propagation_context.propagation_state.backwards.get(),
        simulation.propagation_context.propagation_state.dt_next.get(),
        simulation.propagation_context.samples_data.end_epochs.get(),
        simulation.propagation_context.samples_data.start_epochs.get());

    // set up bool reduction buffer for termination flag kernel
    CudaArray1D<bool> reduction_buffer(n, false);
    check_cuda_error(reduction_buffer.prefetch());

    const auto coi = simulation.propagation_context.samples_data.center_of_integration;

    for (auto step = 0; step < simulation.rkf_parameters.max_steps; ++step)
    {
        // TODO compute dstates of each sample
        // TODO advance steps using dstates, set termination flag for each sample
        // TODO check for termination condition on all samples
    }

    // TODO retrieve final results and return them
}

#pragma endregion