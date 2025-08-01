#include "simulation/rkf_parameters.hpp"
#include <nlohmann/json.hpp>
#include <cuda_runtime.h>

__host__ RKFParameters RKFParameters::from_json(const nlohmann::json &j)
{
    RKFParameters config;
    if (j.contains("initialstep"))
    {
        config.initial_time_step = j["initialstep"];
    }
    if (j.contains("minstep"))
    {
        config.min_time_step = j["minstep"];
    }
    if (j.contains("abstol"))
    {
        config.abs_tol = j["abstol"];
    }
    if (j.contains("reltol"))
    {
        config.rel_tol = j["reltol"];
    }
    if (j.contains("scalestate"))
    {
        config.scale_state = j["scalestate"];
    }
    if (j.contains("scaledstate"))
    {
        config.scale_dstate = j["scaledstate"];
    }
    if (j.contains("maxsteps"))
    {
        config.max_steps = j["maxsteps"];
    }
    return config;
}
