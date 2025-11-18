#include <cuda_runtime.h>
#include "dump.cuh"

void dump_d_states(const HostDerivativesTensor &d_states, const std::string &filename)
{
    auto array = nlohmann::json::array();
    for (auto index = 0; index < d_states.n_mats(); ++index)
    {
        auto states = nlohmann::json::array();
        for (auto stage = 0; stage < RKF78::NStages; ++stage)
        {
            auto state = nlohmann::json::array();
            for (auto dim = 0; dim < STATE_DIM; ++dim)
            {
                state.push_back(d_states.at(stage, index, dim));
            }
            states.push_back(state);
        }
        auto sample = nlohmann::json::object();
        sample["index"] = index;
        sample["states"] = states;
        array.push_back(sample);
    }

    json_to_file(array, filename);
}

void dump_states(const HostStatesMatrix &states, const std::string &filename)
{
    auto array = nlohmann::json::array();
    for (auto index = 0; index < states.n_vecs(); ++index)
    {
        nlohmann::json state = nlohmann::json::array();
        for (auto dim = 0; dim < STATE_DIM; ++dim)
        {
            state.push_back(states.at(index, dim));
        }
        array.push_back(state);
    }
    json_to_file(array, filename);
}
