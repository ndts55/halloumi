#pragma once
#include "core/types.hpp"
#include <nlohmann/json.hpp>
#include <cstddef>
#include <cuda_runtime.h>

struct RKFParameters
{
    Float initial_time_step = 1;
    Float min_time_step = 1e-6;
    Float abs_tol = 1e-9;
    Float rel_tol = 1e-9;
    Float scale_state = 1;
    Float scale_dstate = 0;
    std::size_t max_steps = 1e6;

    static constexpr Float dt_safety = .9;
    static constexpr Float min_dt_scale = .2;
    static constexpr Float max_dt_scale = 5;

    static RKFParameters from_json(const nlohmann::json &exec_json);
};

extern __constant__ RKFParameters device_rkf_parameters;

cudaError_t initialize_rkf_parameters_on_device(const RKFParameters &rkf_parameters);
