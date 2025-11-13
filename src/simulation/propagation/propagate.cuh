#pragma once
#include <cuda_runtime.h>
#include "simulation/simulation.hpp"

__host__ void propagate(Simulation &simulation);
