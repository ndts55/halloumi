#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include "core/types.cuh"
#include "simulation/environment/ephemeris.hpp"
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

void prepare_device_memory(const Simulation &simulation)
{
    std::cout << "Preparing device memory for simulation..." << std::endl;
    check_cuda_error(initialize_rkf_parameters_on_device(simulation.rkf_parameters),
                     "Failed to copy RKF parameters to device");

    check_cuda_error(simulation.ephemeris.data.prefetch());
    check_cuda_error(simulation.ephemeris.integers.prefetch());
    check_cuda_error(simulation.ephemeris.floats.prefetch());
    check_cuda_error(simulation.propagation_context.propagation_state.states.prefetch());
    check_cuda_error(simulation.propagation_context.propagation_state.epochs.prefetch());
    check_cuda_error(simulation.propagation_context.propagation_state.terminated.prefetch());
    check_cuda_error(simulation.propagation_context.propagation_state.dt_last.prefetch());
    check_cuda_error(simulation.propagation_context.propagation_state.dt_next.prefetch());
    check_cuda_error(simulation.propagation_context.propagation_state.simulation_ended.prefetch());
    check_cuda_error(simulation.propagation_context.propagation_state.backwards.prefetch());
    check_cuda_error(simulation.propagation_context.samples_data.end_epochs.prefetch());
    check_cuda_error(simulation.propagation_context.samples_data.start_epochs.prefetch());

    check_cuda_error(cudaDeviceSynchronize(), "Error synchronizing after test kernel");
    check_cuda_error(cudaGetLastError(), "Error launching test kernel");

    check_cuda_error(RKF78::initialize_device_tableau(), "Error initializing RKF78 tableau");

    std::cout << "Successfully set up device" << std::endl;
}

#pragma endregion

#pragma region Vector Operations

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
    const Integer center_of_integration)
{
    auto index = index_in_grid();
    if (index >= termination_flags.n_vecs || termination_flags.at(index))
    {
        return;
    }

    // TODO bring tableau to constant memory

    const auto dt = next_dts.at(index);
    const auto epoch = epochs.at(index);

    for (auto stage = 0; stage < RKF78::NStages; ++stage)
    {
        const auto node_c = RKF78::node(stage);
        Float state[STATE_DIM] = {0.0f};

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
        //                                     acc += thirdBody<false>(
        //                                         pos, bodyPos, env.constants().body(target).gm());
        //                                 } else {
        //                                     acc += twoBody(pos, env.constants().body(target).gm());
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