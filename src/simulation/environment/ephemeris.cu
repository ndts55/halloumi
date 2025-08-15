#include "simulation/environment/ephemeris.cuh"
#include <iostream>
#include <nlohmann/json.hpp>
#include <fstream>
#include "utils.hpp"
#include "cuda/cuda_array.hpp"
#include <cuda_runtime.h>
#include "cuda/vec.cuh"

#pragma region Layout v1
CudaArray1D<Float> load_ephemeris_data_v1(const nlohmann::json &core)
{
    std::size_t data_size = std::accumulate(core.begin(), core.end(), 0, [](const nlohmann::json &a, const nlohmann::json &b)
                                            { return a["data"].size() + b["data"].size(); });
    CudaArray1D<Float> bodies(data_size);
    auto bodies_iterator = bodies.data().get();
    for (auto body : core)
    {
        std::copy(body["data"].begin(), body["data"].end(), bodies_iterator);
        std::advance(bodies_iterator, body["data"].size());
    }
    return bodies;
}

Ephemeris load_brie_v1(const nlohmann::json &core)
{
    std::cout << "loading brie v1" << std::endl;

    // Calculate sizes
    auto n_bodies = core.size();
    std::size_t data_size = std::accumulate(core.begin(), core.end(), 0, [](const nlohmann::json &a, const nlohmann::json &b)
                                            { return a["data"].size() + b["data"].size(); });

    // Construct arrays
    CudaArray1D<Float> bodies(data_size);
    CudaArray2D<Integer, INTSIZE> integers(n_bodies);
    CudaArray2D<Float, REALSIZE> floats(n_bodies);

    // Fill arrays
    std::size_t idx = 0;
    auto bodies_iterator = bodies.data().get();
    for (auto body : core)
    {
        // Data
        std::copy(body["data"].begin(), body["data"].end(), bodies_iterator);

        // Integers
        integers.at(FRAME, idx) = body["metadata"]["frame"];
        integers.at(DTYPE, idx) = body["metadata"]["dtype"];
        integers.at(TARGET, idx) = body["metadata"]["target"];
        integers.at(CENTER, idx) = body["metadata"]["center"];
        integers.at(NINTERVALS, idx) = body["metadata"]["nintervals"];
        integers.at(PDEG, idx) = body["metadata"]["pdeg"];
        integers.at(DATASIZE, idx) = body["data"].size();
        // set data offset
        if (idx == 0)
        {
            integers.at(DATAOFFSET, idx) = 0;
        }
        else
        {
            integers.at(DATAOFFSET, idx) = integers.at(DATAOFFSET, idx - 1) + integers.at(DATASIZE, idx - 1);
        }

        // Floats
        floats.at(INITIALEPOCH, idx) = body["metadata"]["startEpoch"];
        floats.at(FINALEPOCH, idx) = body["metadata"]["finalEpoch"];

        // Set up next iteration
        std::advance(bodies_iterator, body["data"].size());
        idx += 1;
    }

    // Construct result
    return Ephemeris{std::move(bodies), std::move(integers), std::move(floats)};
}
#pragma endregion

#pragma region Layout v2
Ephemeris load_brie_v2(const nlohmann::json &core)
{
    auto metadata = core["metadata"];
    std::size_t n_bodies = metadata["nBodyUnits"];

    CudaArray2D<Integer, INTSIZE> integers(n_bodies);
    CudaArray2D<Float, REALSIZE> floats(n_bodies);
    CudaArray1D<Float> data(core["data"].size());

    std::copy(metadata["intMetadata"].begin(), metadata["intMetadata"].end(), integers.data().get());
    std::copy(metadata["doubleMetadata"].begin(), metadata["doubleMetadata"].end(), floats.data().get());

    auto data_json = core["data"];
    std::copy(data_json.begin(), data_json.end(), data.data().get());

    return Ephemeris{std::move(data), std::move(integers), std::move(floats)};
}
#pragma endregion

#pragma region Ephemeris::from_brie

std::string get_brie_file(const nlohmann::json &json)
{
    if (json.is_string())
    {
        return json;
    }
    else if (json.is_array())
    {
        return json[0];
    }
    print_json(json);
    throw std::invalid_argument("Received invalid ephemeris data json");
}

Ephemeris Ephemeris::from_brie(const nlohmann::json &json)
{
    auto file = get_brie_file(json);

    // load cbor from given file
    auto cbor_json = json_from_cbor(file);

    auto layout = cbor_json["layout"];
    if (!layout.is_number() || (layout != 1 && layout != 2))
    {
        throw std::invalid_argument("brie data is neither v1 nor v2");
    }

    auto core = cbor_json["core"];

    if (layout == 1)
    {
        return load_brie_v1(core);
    }

    return load_brie_v2(core);
}

#pragma endregion

#pragma region DeviceEphemeris::calculate_position

/*
Reconstructs a continuous position vector from discrete Chebyshev coefficients stored in the ephemeris data.
*/
__device__ PositionVector interpolate_type_2_body_to_position(const DeviceEphemeris &eph, const Integer &body_index, const Float &epoch)
{
    auto nintervals = eph.nintervals_at(body_index);
    auto data_offset = eph.dataoffset_at(body_index);
    auto pdeg = eph.pdeg_at(body_index);

    // data = [ ...[other data; (data_offset)], interval radius, ...[intervals; (nintervals)], ...[coefficients; (nintervals * (pdeg + 1))] ]
    auto radius = eph.data_at(data_offset);
    DeviceArray1D<Float> intervals{/* data pointer */ eph.data.data + data_offset + 1, /* size */ (std::size_t)nintervals};
    DeviceArray1D<Float> coefficients{/* data pointer */ intervals.end(), /* size */ (std::size_t)nintervals * (pdeg + 1)};

    std::size_t idx = (epoch - intervals.at(0)) / (2 * radius); // interval selection
    Float s = (epoch - intervals.at(idx)) / radius - 1.0;       // normalized  time coordinate
    // use clenshaw recurrence relation to efficiently calculate chebyshev polynomials
    PositionVector position = {0.0};
    PositionVector w1 = {0.0};
    PositionVector w2 = {0.0};
    Float s2 = 2 * s;
    for (auto i = pdeg; i > 0; --i)
    {
        w2 = w1;
        w1 = position;
        position = (w1 * s2 - w2) + coefficients.at(i * nintervals + idx);
    }
    return (position * s - w1) + coefficients.at(idx);
}

__device__ PositionVector read_position(const DeviceEphemeris &eph, const Float &epoch, const Integer &target, const Integer &center)
{
    PositionVector position = {0.0};
    if (target == center)
    {
        return position;
    }

    Integer t = target;
    while (t != center)
    {
        auto body_index = eph.index_of_target(t);
        // ! We only have type 2 bodies for now.
        position += interpolate_type_2_body_to_position(eph, body_index, epoch);
        t = eph.center_at(body_index);
    }
    return position;
}

__device__ PositionVector DeviceEphemeris::calculate_position(const Float &epoch, const Integer &target, const Integer &center_of_integration) const
{
    auto cc = common_center(target, center_of_integration);
    auto xt = read_position(*this, epoch, target, cc);
    auto xc = read_position(*this, epoch, center_of_integration, cc);
    xt -= xc;
    return xt;
}

#pragma endregion