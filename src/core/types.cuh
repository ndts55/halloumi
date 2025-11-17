#pragma once
#include "cuda/vec.cuh"
#include "simulation/tableau.cuh"
#include "cuda/array.cuh"
#include "cuda/matrix.cuh"
#include "cuda/tensor.cuh"

#pragma region State Vector Dimensions

constexpr std::size_t STATE_DIM = 6;
constexpr std::size_t POSITION_DIM = STATE_DIM / 2;
constexpr std::size_t POSITION_OFFSET = 0;
constexpr std::size_t VELOCITY_DIM = POSITION_DIM;
constexpr std::size_t VELOCITY_OFFSET = POSITION_DIM + POSITION_OFFSET;

#pragma endregion

#pragma region Type Definitions

using StateVector = Vec<double, STATE_DIM>;
using PositionVector = Vec<double, POSITION_DIM>;
using VelocityVector = Vec<double, VELOCITY_DIM>;

using HostFloatArray = HostArray<double>;
using HostIntegerArray = HostArray<int>;
using HostBoolArray = HostArray<bool>;
using HostStatesMatrix = HostMatrix<double, STATE_DIM>;
using HostDerivativesTensor = HostTensor<double, RKF78::NStages, STATE_DIM>;

using DeviceFloatArray = DeviceArray<double>;
using DeviceIntegerArray = DeviceArray<int>;
using DeviceBoolArray = DeviceArray<bool>;
using DeviceStatesMatrix = DeviceMatrix<double, STATE_DIM>;
using DeviceDerivativesTensor = DeviceTensor<double, RKF78::NStages, STATE_DIM>;

using CudaIndex = unsigned int;

#pragma endregion
