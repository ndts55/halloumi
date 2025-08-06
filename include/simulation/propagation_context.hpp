#pragma once
#include "core/types.cuh"
#include "cuda/cuda_array.hpp"
#include <nlohmann/json.hpp>

struct PropagationState
{
    CudaArray2D<Float, STATE_DIM> states;
    CudaArray1D<Float> epochs;
    CudaArray1D<bool> terminated;
    CudaArray1D<Float> last_dts;
    CudaArray1D<Float> next_dts;
    CudaArray1D<bool> simulation_ended;
    CudaArray1D<bool> backwards;
};

struct SamplesData
{
    std::size_t n_vecs;
    Integer center_of_integration;
    Float duration_in_days;
    CudaArray1D<Float> end_epochs;
    CudaArray1D<Float> start_epochs;
};

struct PropagationContext
{
    PropagationState propagation_state;
    SamplesData samples_data;

    PropagationContext(PropagationState &&propagation_state, SamplesData &&samples_data);
    static PropagationContext from_json(const nlohmann::json &json);
};
