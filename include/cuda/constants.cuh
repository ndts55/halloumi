#pragma once
#include "simulation/rkf_parameters.hpp"
#include "cuda/device_ephemeris.cuh"
#include <cuda_runtime.h>

extern __constant__ RKFParameters device_rkf_parameters;
extern __device__ DeviceEphemeris device_ephemeris;
