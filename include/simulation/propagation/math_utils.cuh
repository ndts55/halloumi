#include <cuda_runtime.h>
#include "cuda/vec.cuh"
#include "core/types.cuh"

__device__ inline Vec<Float, VELOCITY_DIM> two_body(const Vec<Float, POSITION_DIM> &position_delta, const Float &gm)
{
    return position_delta * -gm * position_delta.reciprocal_cubed_norm();
}

__device__ inline Vec<Float, VELOCITY_DIM> three_body_barycentric(const Vec<Float, POSITION_DIM> &source_position, const Vec<Float, POSITION_DIM> &body_position, const Float &gm)
{
    return two_body(source_position - body_position, gm);
}

__device__ inline Vec<Float, VELOCITY_DIM> three_body_non_barycentric(const Vec<Float, POSITION_DIM> &source_position, const Vec<Float, POSITION_DIM> &body_position, const Float &gm)
{
    return three_body_barycentric(source_position, body_position, gm) + two_body(body_position, gm);
}