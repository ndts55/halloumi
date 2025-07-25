#pragma once
#include "types.hpp"
#include "device_array.cuh"
#include "cuda_array.hpp"
#include "ephemeris.hpp"

// Mirrors Ephemeris but specifically for device-side operations since access to integers, floats, and data is non-regular
struct DeviceEphemeris
{
private:
    DeviceArray1D<Float> data; // ? is this really a 1d array?
    DeviceArray2D<Integer, INTSIZE> integers;
    DeviceArray2D<Float, REALSIZE> floats;

public:
    DeviceEphemeris() = default;
    DeviceEphemeris(const Ephemeris &ephemeris)
        : data(ephemeris.data.get()), integers(ephemeris.integers.get()), floats(ephemeris.floats.get()) {}

    // Integers
    __device__ inline const Integer &frame_at(std::size_t idx) const
    {
        return integers.at(FRAME, idx);
    }

    __device__ inline const Integer &dtype_at(std::size_t idx) const
    {
        return integers.at(DTYPE, idx);
    }

    __device__ inline const Integer &target_at(std::size_t idx) const
    {
        return integers.at(TARGET, idx);
    }

    __device__ inline const Integer &center_at(std::size_t idx) const
    {
        return integers.at(CENTER, idx);
    }

    __device__ inline const Integer &nintervals_at(std::size_t idx) const
    {
        return integers.at(NINTERVALS, idx);
    }

    __device__ inline const Integer &pdeg_at(std::size_t idx) const
    {
        return integers.at(PDEG, idx);
    }

    __device__ inline const Integer &dataoffset_at(std::size_t idx) const
    {
        return integers.at(DATAOFFSET, idx);
    }

    __device__ inline const Integer &datasize_at(std::size_t idx) const
    {
        return integers.at(DATASIZE, idx);
    }

    // Floats
    __device__ inline const Float &initial_epoch_at(std::size_t idx) const
    {
        return floats.at(INITIALEPOCH, idx);
    }

    __device__ inline const Float &final_epoch_at(std::size_t idx) const
    {
        return floats.at(FINALEPOCH, idx);
    }

    // Data
    __device__ inline const Float &data_at(std::size_t index) const
    {
        return data.at(index);
    }

    // Get number of bodies
    __device__ inline std::size_t n_bodies() const
    {
        return integers.n_vecs;
    }
};