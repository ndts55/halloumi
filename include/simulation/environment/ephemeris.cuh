#pragma once
#include <nlohmann/json.hpp>
#include <cuda_runtime.h>
#include "core/types.cuh"
#include "cuda/vec.cuh"
#include "cuda/matrix.cuh"

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
    DeviceFloatArray data; // ? is this really a 1d array?
    DeviceMatrix<Integer, INTSIZE> integers;
    DeviceMatrix<Float, REALSIZE> floats;

    DeviceEphemeris() = default;
    DeviceEphemeris(DeviceFloatArray &&d, DeviceMatrix<Integer, INTSIZE> &&i, DeviceMatrix<Float, REALSIZE> &&f)
        : data(std::move(d)), integers(std::move(i)), floats(std::move(f)) {}

    // Integers
    __device__ inline const Integer &frame_at(std::size_t idx) const
    {
        return integers.at(idx, FRAME);
    }

    __device__ inline const Integer &dtype_at(std::size_t idx) const
    {
        return integers.at(idx, DTYPE);
    }

    __device__ inline const Integer &target_at(std::size_t idx) const
    {
        return integers.at(idx, TARGET);
    }

    __device__ inline const Integer &center_at(std::size_t idx) const
    {
        return integers.at(idx, CENTER);
    }

    __device__ inline const Integer &nintervals_at(std::size_t idx) const
    {
        return integers.at(idx, NINTERVALS);
    }

    __device__ inline const Integer &pdeg_at(std::size_t idx) const
    {
        return integers.at(idx, PDEG);
    }

    __device__ inline const Integer &dataoffset_at(std::size_t idx) const
    {
        return integers.at(idx, DATAOFFSET);
    }

    __device__ inline const Integer &datasize_at(std::size_t idx) const
    {
        return integers.at(idx, DATASIZE);
    }

    // Floats
    __device__ inline const Float &initial_epoch_at(std::size_t idx) const
    {
        return floats.at(idx, INITIALEPOCH);
    }

    __device__ inline const Float &final_epoch_at(std::size_t idx) const
    {
        return floats.at(idx, FINALEPOCH);
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
        return data.n_elements;
    }

    // Helper methods
    __device__ inline std::size_t index_of_target(const Integer &target) const
    {
#ifdef __CUDA_ARCH__
        // if (threadIdx.x == 0 && blockIdx.x == 0)
        // {
        //     printf("====\n");
        //     printf("size is %llu", n_bodies());
        //     // FIXME the sizes are different, I get 14 and CUDAjectory getts 4
        //     for (std::size_t i = 0; i < n_bodies(); ++i)
        //     {
        //         auto ti = target_at(i);
        //         printf("    Target %llu at %llu\n", ti, i);
        //     }
        // }
#endif
        for (std::size_t i = 0; i < n_bodies(); ++i)
        {
            auto ti = target_at(i);
            if (ti == target)
            {
                return i;
            }
        }

        return n_bodies();
    }

    __device__ Integer common_center(const Integer &target, const Integer &center) const
    {
        if (target == 0 || center == 0)
        {
            return 0;
        }

        Integer tc = target;
        Integer cc = center;
        Integer tcnew, ccnew;

        while (tc != cc && tc != 0 && cc != 0)
        {
            tcnew = center_at(index_of_target(tc));
            if (tcnew == cc)
            {
                // target center is center of old center
                return tcnew;
            }
            ccnew = center_at(index_of_target(cc));
            if (ccnew == tc)
            {
                // center center is center of old target
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

    __device__ PositionVector calculate_position(const Float &epoch, const Integer &target, const Integer &center_of_integration) const;

    __device__ PositionVector read_position(const Float &epoch, const Integer &target, const Integer &center) const;

    __device__ PositionVector interpolate_type_2_body_to_position(const std::size_t &body_index, const Float &epoch) const;
};

struct Ephemeris
{
    HostFloatArray data;
    HostMatrix<Integer, INTSIZE> integers;
    HostMatrix<Float, REALSIZE> floats;

    static Ephemeris from_brie(const nlohmann::json &json);

    inline std::size_t n_bodies() const { return integers.n_vecs(); }

    DeviceEphemeris get() const
    {
        return DeviceEphemeris(data.get(), integers.get(), floats.get());
    }
};
