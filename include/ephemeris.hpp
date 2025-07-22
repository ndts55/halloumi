#pragma once
#include <nlohmann/json.hpp>
#include "types.hpp"
#include "cuda_array.hpp"

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

struct Ephemeris
{
    CudaArray1D<Float> data;
    CudaArray2D<Integer, INTSIZE> integers;
    CudaArray2D<Float, REALSIZE> floats;

    static Ephemeris from_brie(const nlohmann::json &json);
};
