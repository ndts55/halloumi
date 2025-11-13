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

using Integer = long int;
using Float = double;
using Bool = uint8_t;

using StateVector = Vec<Float, STATE_DIM>;
using PositionVector = Vec<Float, POSITION_DIM>;
using VelocityVector = Vec<Float, VELOCITY_DIM>;

using HostFloatArray = HostArray<Float>;
using HostIntegerArray = HostArray<Integer>;
using HostBoolArray = HostArray<Bool>;
using HostStatesMatrix = HostMatrix<Float, STATE_DIM>;
using HostDerivativesTensor = HostTensor<Float, RKF78::NStages, STATE_DIM>;

using DeviceFloatArray = DeviceArray<Float>;
using DeviceIntegerArray = DeviceArray<Integer>;
using DeviceBoolArray = DeviceArray<Bool>;
using DeviceStatesMatrix = DeviceMatrix<Float, STATE_DIM>;
using DeviceDerivativesTensor = DeviceTensor<Float, RKF78::NStages, STATE_DIM>;

using CudaIndex = unsigned int;

#pragma endregion
