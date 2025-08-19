#pragma once
#include "cuda/vec.cuh"
#include "cuda/device_array.cuh"
#include "simulation/tableau.cuh"
#include "cuda/cuda_array.hpp"

#pragma region State Vector Dimensions

constexpr std::size_t STATE_DIM = 6;
constexpr std::size_t POSITION_DIM = STATE_DIM / 2;
constexpr std::size_t POSITION_OFFSET = 0;
constexpr std::size_t VELOCITY_DIM = POSITION_DIM;
constexpr std::size_t VELOCITY_OFFSET = POSITION_DIM + POSITION_OFFSET;

#pragma endregion

#pragma region Type Definitions

using Integer = long int;
using Float = double;

using StateVector = Vec<Float, STATE_DIM>;
using PositionVector = Vec<Float, POSITION_DIM>;
using VelocityVector = Vec<Float, VELOCITY_DIM>;

using GlobalFloatArray = CudaArray1D<Float>;
using GlobalIntegerArray = CudaArray1D<Integer>;
using GlobalBoolArray = CudaArray1D<bool>;
using GlobalStatesMatrix = CudaArray2D<Float, STATE_DIM>;
using GlobalDerivativesTensor = CudaArray3D<Float, STATE_DIM, RKF78::NStages>;

using DeviceFloatArray = DeviceArray1D<Float>;
using DeviceIntegerArray = DeviceArray1D<Integer>;
using DeviceBoolArray = DeviceArray1D<bool>;
using DeviceStatesMatrix = DeviceArray2D<Float, STATE_DIM>;
using DeviceDerivativesTensor = DeviceArray3D<Float, STATE_DIM, RKF78::NStages>;

using CudaIndex = unsigned int;

#pragma endregion
