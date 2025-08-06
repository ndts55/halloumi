#include "simulation/propagation_context.hpp"
#include <iostream>
#include <format>
#include "utils.hpp"
#include "cuda/cuda_array.hpp"

PropagationContext::PropagationContext(
    PropagationState &&propagation_state,
    SamplesData &&samples_data) : propagation_state(std::move(propagation_state)), samples_data(std::move(samples_data))
{
#ifndef NDEBUG
    auto expected_size = samples_data.n_vecs;

    // Check each array size against expected size
    bool states_check = (expected_size == propagation_state.states.n_vecs());
    bool epochs_check = (expected_size == propagation_state.epochs.n_elements());
    bool terminated_check = (expected_size == propagation_state.terminated.n_elements());
    bool dt_last_check = (expected_size == propagation_state.last_dts.n_elements());
    bool dt_next_check = (expected_size == propagation_state.next_dts.n_elements());
    bool end_epochs_check = (expected_size == samples_data.end_epochs.n_elements());
    bool start_epochs_check = (expected_size == samples_data.start_epochs.n_elements());

    // If any check fails, output details and throw exception
    if (!states_check || !epochs_check || !terminated_check || !dt_last_check ||
        !dt_next_check || !end_epochs_check || !start_epochs_check)
    {
        std::cerr << "Size validation failed. Expected size: " << expected_size << "\n";
        std::cerr << "┌───────────────────┬────────┬───────────────┐\n";
        std::cerr << "│ Array             │ Valid? │ Actual Size   │\n";
        std::cerr << "├───────────────────┼────────┼───────────────┤\n";
        std::cerr << "│ states            │ " << (states_check ? " Yes  " : " No   ") << "│ " << propagation_state.states.n_vecs() << "\n";
        std::cerr << "│ epochs            │ " << (epochs_check ? " Yes  " : " No   ") << "│ " << propagation_state.epochs.n_elements() << "\n";
        std::cerr << "│ terminated        │ " << (terminated_check ? " Yes  " : " No   ") << "│ " << propagation_state.terminated.n_elements() << "\n";
        std::cerr << "│ last_dts           │ " << (dt_last_check ? " Yes  " : " No   ") << "│ " << propagation_state.last_dts.n_elements() << "\n";
        std::cerr << "│ next_dts           │ " << (dt_next_check ? " Yes  " : " No   ") << "│ " << propagation_state.next_dts.n_elements() << "\n";
        std::cerr << "│ end_epochs        │ " << (end_epochs_check ? " Yes  " : " No   ") << "│ " << samples_data.end_epochs.n_elements() << "\n";
        std::cerr << "│ start_epochs      │ " << (start_epochs_check ? " Yes  " : " No   ") << "│ " << samples_data.start_epochs.n_elements() << "\n";
        std::cerr << "└───────────────────┴────────┴───────────────┘\n";

        throw std::runtime_error("Malformed states: array size mismatch");
    }
#endif
}

PropagationContext PropagationContext::from_json(const nlohmann::json &configuration)
{
    auto samples_json = json_from_file(configuration["samples"]);

    auto in_samples = samples_json["inSamples"];
    auto n_vecs = in_samples.size();

    // Mutable State
    CudaArray2D<Float, STATE_DIM> states(n_vecs);
    CudaArray1D<Float> epochs(n_vecs);
    CudaArray1D<bool> terminated(n_vecs, false);

    // Immutable state that has to be calculated
    Float duration_in_days = configuration["simConfig"]["durationDays"];
    CudaArray1D<Float> start_epochs(n_vecs);
    CudaArray1D<Float> end_epochs(n_vecs);

    std::size_t idx = 0;
    for (auto sample : in_samples)
    {
        Float e = sample["epoch"];
        epochs.at(idx) = e;
        end_epochs.at(idx) = e + duration_in_days; // TODO find out whether just adding the days is even correct
        auto cart = sample["stateCart"];
        for (auto dim = 0; dim < STATE_DIM; dim++)
        {
            // put each element of vec<6> in the correct position
            states.at(dim, idx) = cart[dim];
        }
        idx += 1;
    }

    return PropagationContext(
        PropagationState{
            std::move(states),
            std::move(epochs),
            std::move(terminated),
            /* last_dts */ CudaArray1D<Float>(n_vecs, 0.0f),
            /* next_dts */ CudaArray1D<Float>(n_vecs, 0.0f),
            /* simulation_ended */ CudaArray1D<bool>(n_vecs, false),
            /* backwards */ CudaArray1D<bool>(n_vecs, false)},
        SamplesData{
            n_vecs,
            /* center_of_integration: */ samples_json["centre"],
            duration_in_days,
            std::move(end_epochs),
            /* start_epochs: */ CudaArray1D<Float>(epochs)});
}
