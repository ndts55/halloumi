#pragma once
#include "types.hpp"
#include "cuda_array.hpp"
#include <nlohmann/json.hpp>

#pragma region Helper Structs

struct PropagationState
{
    CudaArray2D<Float, STATE_DIM> states;
    CudaArray1D<Float> epochs;
    CudaArray1D<bool> terminated;
    CudaArray1D<Float> dt_last;
    CudaArray1D<Float> dt_next;
};

struct SamplesData
{
    const std::size_t n_vecs;
    const Integer center_of_integration;
    const Float duration_in_days;
    const CudaArray1D<Float> end_epochs;
    const CudaArray1D<Float> start_epochs;
};

#pragma endregion

// PropagationContext SoA
class PropagationContext
{
private:
    PropagationState state_;
    const SamplesData data_;

public:
    PropagationContext() = default;
    PropagationContext(PropagationState &&propagation_state, SamplesData &&samples_data);
    static PropagationContext from_json(const nlohmann::json &json);
    inline std::size_t size() const { return data_.n_vecs; }
    inline std::size_t n_vecs() const { return data_.n_vecs; }
};
