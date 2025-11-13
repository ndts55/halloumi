#pragma once
#include "core/types.cuh"
#include <nlohmann/json.hpp>
#include <cstddef>
#include <cuda_runtime.h>

struct RKFParameters
{
    double initial_time_step = 1;
    double min_time_step = 1e-6;
    double abs_tol = 1e-9;
    double rel_tol = 1e-9;
    double scale_state = 1;
    double scale_dstate = 0;
    std::size_t max_steps = 1e6;

    static constexpr double dt_safety = .9;
    static constexpr double min_dt_scale = .2;
    static constexpr double max_dt_scale = 5;

    static RKFParameters from_json(const nlohmann::json &exec_json);
};

extern __constant__ RKFParameters device_rkf_parameters;

cudaError_t initialize_rkf_parameters_on_device(const RKFParameters &rkf_parameters);
