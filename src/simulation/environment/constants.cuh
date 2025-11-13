#pragma once
#include <nlohmann/json.hpp>
#include "core/types.cuh"
#include <cuda_runtime.h>

// ? How are body and universe constants accessed
// all threads access the same data since they loop through all active bodies
// seems like only the GM data is used

enum BodyConstants
{
    GM,
    NUM_BODY_CONSTANTS
};
namespace celestial_constants
{
    constexpr Integer SUN_ID = 10;
    constexpr Integer EARTH_ID = 399;
    constexpr Integer MOON_ID = 301;
    // constexpr Integer EARTH_BARYCENTER_ID = 3;

    constexpr std::size_t BODY_COUNT = 3;

    constexpr std::array<Integer, BODY_COUNT> BODY_IDS = {SUN_ID, EARTH_ID, MOON_ID};

    constexpr double SUN_GM = 1.3271244004193938E+11;
    constexpr double EARTH_GM = 3.9860043543609598E+05;
    constexpr double MOON_GM = 4.9028000661637961E+03;
    // constexpr double EARTH_BARYCENTER_GM = ? ;

    constexpr std::array<double, BODY_COUNT> BODY_GMS = {SUN_GM, EARTH_GM, MOON_GM};
    constexpr std::array<std::array<double, BODY_COUNT>, NUM_BODY_CONSTANTS> BODY_CONSTANTS = {BODY_GMS};

    constexpr double SUN_RADIUS = 696342.0;
    constexpr double MOON_RADIUS = 1737.5;
    constexpr double EARTH_RADIUS = 6378.1366;

    constexpr double AU = 1.4959787070000000E+08;
    constexpr double SPEED_OF_LIGHT = 2.9979245800000000E+05;
}

struct DeviceConstants
{
private:
    DeviceIntegerArray body_ids;
    DeviceFloatArray gms;

public:
    DeviceConstants(DeviceIntegerArray &&ids, DeviceFloatArray &&g)
        : body_ids(std::move(ids)), gms(std::move(g)) {}

    __device__ inline std::size_t index_of(Integer body_id) const
    {
        for (std::size_t i = 0; i < body_ids.n_elements; ++i)
        {
            if (body_ids.at(i) == body_id)
            {
                return i;
            }
        }
        return 0;
    }

    __device__ inline double gm_for(Integer body_id) const
    {
        return gms.at(index_of(body_id));
    }
};

struct Constants
{
    HostIntegerArray body_ids{celestial_constants::BODY_IDS};
    HostFloatArray gms{celestial_constants::BODY_GMS};

    DeviceConstants get() const
    {
        return DeviceConstants(body_ids.get(), gms.get());
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