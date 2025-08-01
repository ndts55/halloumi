#include "cuda/constants.cuh"
#include "simulation/rkf_parameters.hpp"
#include "cuda/device_ephemeris.cuh"

// Define the constant memory for RKF parameters
__constant__ RKFParameters device_rkf_parameters;

// Define the global device ephemeris
// NOTE: We use __device__ instead of __constant__ because DeviceEphemeris contains pointers
__device__ DeviceEphemeris device_ephemeris;
