#pragma once
#include "cuda/vec.cuh"
#include "cuda/device_array.cuh"
#include "tableau.cuh"

#pragma region State Vector Dimensions

constexpr int STATE_DIM = 6;
constexpr int POSITION_DIM = STATE_DIM / 2;
constexpr int POSITION_OFFSET = 0;
constexpr int VELOCITY_DIM = POSITION_DIM;
constexpr int VELOCITY_OFFSET = POSITION_DIM + POSITION_OFFSET;

#pragma endregion

#pragma region Type Definitions

using Integer = long int;
using Float = double;

using StateVector = Vec<Float, STATE_DIM>;
using PositionVector = Vec<Float, POSITION_DIM>;
using VelocityVector = Vec<Float, VELOCITY_DIM>;
using StatesDeviceMatrix = DeviceArray2D<Float, STATE_DIM>;
using DerivativesDeviceTensor = DeviceArray3D<Float, STATE_DIM, RKF78::NStages>;

#pragma endregion
