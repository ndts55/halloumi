#pragma once
#include <nlohmann/json.hpp>
#include <cuda_runtime.h>
#include "core/types.cuh"
#include "cuda/vec.cuh"
#include "cuda/matrix.cuh"

enum IntMembers
{
    /* int members composing the Ephemeris metadata */
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
    DeviceFloatArray data;
    DeviceMatrix<int, INTSIZE> integers;
    DeviceMatrix<double, REALSIZE> floats;

    DeviceEphemeris() = default;
    DeviceEphemeris(DeviceFloatArray &&d, DeviceMatrix<int, INTSIZE> &&i, DeviceMatrix<double, REALSIZE> &&f)
        : data(std::move(d)), integers(std::move(i)), floats(std::move(f)) {}

    // Integers
    __device__ inline const int &frame_at(std::size_t idx) const
    {
        return integers.at(idx, FRAME);
    }

    __device__ inline const int &dtype_at(std::size_t idx) const
    {
        return integers.at(idx, DTYPE);
    }

    __device__ inline const int &target_at(std::size_t idx) const
    {
        return integers.at(idx, TARGET);
    }

    __device__ inline const int &center_at(std::size_t idx) const
    {
        return integers.at(idx, CENTER);
    }

    __device__ inline const int &nintervals_at(std::size_t idx) const
    {
        return integers.at(idx, NINTERVALS);
    }

    __device__ inline const int &pdeg_at(std::size_t idx) const
    {
        return integers.at(idx, PDEG);
    }

    __device__ inline const int &dataoffset_at(std::size_t idx) const
    {
        return integers.at(idx, DATAOFFSET);
    }

    __device__ inline const int &datasize_at(std::size_t idx) const
    {
        return integers.at(idx, DATASIZE);
    }

    // Floats
    __device__ inline const double &initial_epoch_at(std::size_t idx) const
    {
        return floats.at(idx, INITIALEPOCH);
    }

    __device__ inline const double &final_epoch_at(std::size_t idx) const
    {
        return floats.at(idx, FINALEPOCH);
    }

    // Data
    __device__ inline const double &data_at(std::size_t index) const
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
    __device__ inline std::size_t index_of_target(const int &target) const
    {
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

    __device__ int common_center(const int &target, const int &center) const
    {
        if (target == 0 || center == 0)
        {
            return 0;
        }

        int tc = target;
        int cc = center;
        int tcnew, ccnew;

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

    __device__ PositionVector calculate_position(const double &epoch, const int &target, const int &center_of_integration) const;

    __device__ PositionVector read_position(const double &epoch, const int &target, const int &center) const;

    __device__ PositionVector interpolate_type_2_body_to_position(const std::size_t &body_index, const double &epoch) const;
};

struct Ephemeris
{
    HostFloatArray data;
    HostMatrix<int, INTSIZE> integers;
    HostMatrix<double, REALSIZE> floats;

    static Ephemeris from_brie(const nlohmann::json &json);

    inline std::size_t n_bodies() const { return integers.n_vecs(); }

    DeviceEphemeris get() const
    {
        return DeviceEphemeris(data.get(), integers.get(), floats.get());
    }
};
