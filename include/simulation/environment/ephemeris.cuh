#pragma once
#include <nlohmann/json.hpp>
#include <cuda_runtime.h>
#include "core/types.cuh"
#include "cuda/cuda_array.hpp"
#include "cuda/vec.cuh"

enum IntMembers
{
    /* Integer members composing the Ephemeris metadata */
    FRAME,
    DTYPE,
    TARGET,
    CENTER,
    NINTERVALS,
    PDEG,
    DATAOFFSET,
    DATASIZE,
    /* Leave the following item as last -- it only acts as enum size */
    INTSIZE
};

enum RealMembers
{
    /* Real members composing the Ephemeris metadata */
    INITIALEPOCH,
    FINALEPOCH,
    /* Leave the following item as last -- it only act as enum size*/
    REALSIZE
};

struct DeviceEphemeris
{
    // TODO use pointer types directly and store n_vecs only once
    DeviceArray1D<Float> data; // ? is this really a 1d array?
    DeviceArray2D<Integer, INTSIZE> integers;
    DeviceArray2D<Float, REALSIZE> floats;

    DeviceEphemeris() = default;
    DeviceEphemeris(DeviceArray1D<Float> &&d, DeviceArray2D<Integer, INTSIZE> &&i, DeviceArray2D<Float, REALSIZE> &&f)
        : data(d), integers(i), floats(f) {}

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

    __device__ inline std::size_t size() const
    {
        return data.n_vecs;
    }

    // Helper methods
    __device__ inline Integer index_of_target(const Integer &target) const
    {
        for (auto i = 0; i < n_bodies(); ++i)
        {
            if (target_at(i) == target)
            {
                return i;
            }
        }
        return n_bodies();
    }

    // TODO profile with and without inline
    __device__ Integer common_center(Integer tc, Integer cc) const
    {
        if (tc == 0 || cc == 0)
        {
            return 0;
        }

        Integer tcnew, ccnew;

        while (tc != cc && tc != 0 && cc != 0)
        {
            tcnew = center_at(index_of_target(tc));
            if (tcnew == cc)
            {
                return tcnew;
            }
            ccnew = center_at(index_of_target(cc));
            if (ccnew == tc)
            {
                return ccnew;
            }
            tc = tcnew;
            cc = ccnew;
        }
        if (tc == 0 || cc == 0)
        {
            return 0;
        }
        else
        {
            return tc;
        }
    }

    __device__ Vec<Float, POSITION_DIM> calculate_position(const Float &epoch, const Integer &target, const Integer &center_of_integration) const;
};

struct Ephemeris
{
    CudaArray1D<Float> data;
    CudaArray2D<Integer, INTSIZE> integers;
    CudaArray2D<Float, REALSIZE> floats;

    static Ephemeris from_brie(const nlohmann::json &json);

    inline std::size_t n_bodies() const { return integers.n_vecs(); }

    DeviceEphemeris get() const
    {
        return DeviceEphemeris(data.get(), integers.get(), floats.get());
    }
};
