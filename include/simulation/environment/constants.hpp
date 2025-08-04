#pragma once
#include <nlohmann/json.hpp>
#include "core/types.hpp"
#include "cuda/cuda_array.hpp"
#include "cuda/device_array.cuh"
#include <cuda_runtime.h>

// ? How are body and universe constants accessed
// all threads access the same data since they loop through all active bodies
// seems like only the GM data is used

enum BodyConstants
{
    GM,
    NUM_BODY_CONSTANTS
};

struct DeviceConstants
{
private:
    DeviceArray1D<Integer> body_ids;
    DeviceArray2D<Float, NUM_BODY_CONSTANTS> data;

public:
    DeviceConstants(DeviceArray1D<Integer> &&ids, DeviceArray2D<Float, NUM_BODY_CONSTANTS> &&d)
        : body_ids(std::move(ids)), data(std::move(d)) {}

    __device__ inline std::size_t index_of(Integer body_id) const
    {
        for (std::size_t i = 0; i < body_ids.n_vecs; ++i)
        {
            if (body_ids.at(i) == body_id)
            {
                return i;
            }
        }
        return 0;
    }

    __device__ inline Float gm_for(Integer body_id) const
    {
        auto i = index_of(body_id);
        return data.at(GM, i);
    }
};

struct Constants
{
private:
    CudaArray1D<Integer> body_ids{10, 301, 399};
    CudaArray2D<Float, NUM_BODY_CONSTANTS> data{{1.3271244004193938E+11}, {4.9028000661637961E+03}, {3.9860043543609598E+05}};

public:
    DeviceConstants get() const
    {
        return DeviceConstants(body_ids.get(), data.get());
    }

    inline std::size_t n_bodies() const { return body_ids.n_elements(); }
};

/*
[
    {
        "CONSTANTS": "",
        "AU": 1.4959787070000000E+08,
        "CLIGHT": 2.9979245800000000E+05,
        "EMRAT": 8.1300568800524700E+01,
        "PS": 4.56e-9
    },
    {
        "NAME": "Sun",
        "NAIFID": 10,
        "GM": 1.3271244004193938E+11,
        "SOI": 10000000000,
        "R": 696342,
        "A_MEAN": 0,
        "J2": 0.0
    },
    {
        "NAME": "Moon",
        "NAIFID": 301,
        "MAIN_BODY": "Earth",
        "GM": 4.9028000661637961E+03,
        "SOI": 66194.377280492699,
        "R": 1737.5,
        "A_MEAN": 384466.5276989999,
        "J2": 203.43e-6
    },
    {
        "NAME": "Earth",
        "NAIFID": 399,
        "MAIN_BODY": "Sun",
        "GM": 3.9860043543609598E+05,
        "SOI": 924646.788599425,
        "R": 6378.1366,
        "A_MEAN": 149597870.7,
        "J2": 1082.636022984e-6
    }
]
*/