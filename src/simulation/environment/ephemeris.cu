#include "simulation/environment/ephemeris.cuh"
#include <iostream>
#include <nlohmann/json.hpp>
#include <fstream>
#include "utils.hpp"
#include "cuda/cuda_array.hpp"
#include <cuda_runtime.h>
#include "cuda/vec.cuh"

#pragma region Layout v1
GlobalFloatArray load_ephemeris_data_v1(const nlohmann::json &core)
{
    std::size_t data_size = std::accumulate(core.begin(), core.end(), 0, [](const nlohmann::json &a, const nlohmann::json &b)
                                            { return a["data"].size() + b["data"].size(); });
    GlobalFloatArray bodies(data_size);
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
    GlobalFloatArray bodies(data_size);
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
    GlobalFloatArray data(core["data"].size());

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
__device__ PositionVector DeviceEphemeris::interpolate_type_2_body_to_position(
    const std::size_t &body_index,
    const Float &epoch) const
{
    auto nintervals = nintervals_at(body_index);
    auto data_offset = dataoffset_at(body_index);
    auto pdeg = pdeg_at(body_index);

    // data = [ ...[other data; (data_offset)], interval radius, ...[intervals; (nintervals)], ...[coefficients; (nintervals * (pdeg + 1))] ]
    auto radius = data_at(data_offset);
    DeviceFloatArray intervals{
        .data = data.data + data_offset + 1,
        .n_vecs = (std::size_t)nintervals};
    // TODO create dynamic device array where vector size does not have to be a compile-time constant
    DeviceFloatArray coefficients{
        .data = intervals.end(),
        .n_vecs = (std::size_t)nintervals * (pdeg + 1)};

    std::size_t idx = (epoch - intervals.at(0)) / (2 * radius); // interval selection
    Float s = (epoch - intervals.at(idx)) / radius - 1.;        // normalized  time coordinate
    // use clenshaw recurrence relation to efficiently calculate chebyshev polynomials
    PositionVector position{0.0};
    PositionVector w1{0.0};
    PositionVector w2{0.0};
    Float s2 = 2 * s;
    // highestDegree = numIndexes - 1 = degree - 1 + 1 = pdeg - 1 + 1 = pdeg
    for (auto degree = pdeg; degree > 0; --degree)
    {
        w2 = w1;
        w1 = position;
        position = (w1 * s2 - w2) + coefficients.at(get_2d_index(nintervals, degree, idx));
    }
    return (position * s - w1) + coefficients.at(/* get_2d_index(nintervals, 0, idx) = */ idx);
}

__device__ PositionVector DeviceEphemeris::read_position(
    const Float &epoch,
    const Integer &target,
    const Integer &center) const
{
    PositionVector position{0.0};
    if (target == center)
    {
        return position;
    }

    Integer t = target;
    while (t != center)
    {
        std::size_t body_index = index_of_target(t);
        // ! We only have type 2 bodies for now.
        position += interpolate_type_2_body_to_position(body_index, epoch);
        t = center_at(body_index);
    }
    return position;
}

__device__ PositionVector DeviceEphemeris::calculate_position(const Float &epoch, const Integer &target, const Integer &center_of_integration) const
{
    Integer cc = common_center(target, center_of_integration);
    PositionVector xc = read_position(epoch, center_of_integration, cc);
    return read_position(epoch, target, cc) - xc;
}

#pragma endregion