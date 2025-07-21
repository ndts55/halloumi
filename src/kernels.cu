#include "kernels.cuh"
#include "types.hpp"
#include "device_array.cuh"
#include "rkf_parameters.hpp"
#include <cuda_runtime.h>

__device__ unsigned int index_in_grid()
{
    return blockIdx.x * blockDim.x + threadIdx.x;
}
__device__ unsigned int index_in_block()
{
    return threadIdx.x;
}

__global__ void prepare_for_continuation(const RKFParameters rkf_parameters,
                                         DeviceArray2D<Float, STATE_DIM> samples)
{
    const auto global_index = index_in_grid();
    if (global_index >= samples.n_vecs)
    {
        return;
    }
    // There seems to be a hidden 1D array called end_of_simulation

    // propagation::detail::prepare<typename ModelT::SamplesT>(i, samples, cfg.initialTimeStep());
    // ->
    // samples.metadata().base().resetEndOfSimulation(i);
    // samples.metadata().base().setInitialStep(i, dt);
    // ->
    // resetEndOfSimulation //
    // if (this->endOfSimulation()[idx]) {
    //     this->terminated()[idx]      = false;
    //     this->endOfSimulation()[idx] = false;
    // }
    // setInitialStep //
    // Real span = this->endEpochs()[idx] - this->startEpochs()[idx];
    // this->nextDts()[idx] = copysign(dt, span);
    // /* Update the backward flag if the sign of the span is negative */
    // this->backward()[idx] = span < 0;
}
