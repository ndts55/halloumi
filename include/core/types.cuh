#pragma once

#pragma region State Vector Dimensions

#define STATE_DIM 6
#define POSITION_DIM (STATE_DIM / 2)
#define POSITION_OFFSET 0
#define VELOCITY_DIM (POSITION_DIM)
#define VELOCITY_OFFSET (POSITION_DIM + POSITION_OFFSET)

#pragma endregion

#pragma region Type Definitions

using Integer = long int;
using Float = double;

#pragma endregion
