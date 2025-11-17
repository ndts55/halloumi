#pragma once
#include <cuda_runtime.h>
#include "simulation/simulation.cuh"

__host__ void propagate(Simulation &simulation);
