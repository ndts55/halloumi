#pragma once
#include "cuda/vec.cuh"

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

#pragma endregion
