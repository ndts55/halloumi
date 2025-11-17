#include "simulation/environment/ephemeris.cuh"
#include <iostream>
#include <nlohmann/json.hpp>
#include <fstream>
#include "utils.cuh"
#include <cuda_runtime.h>
#include "cuda/vec.cuh"
#include "core/types.cuh"

#pragma region Layout v1
HostFloatArray load_ephemeris_data_v1(const nlohmann::json &core)
{
    std::size_t data_size = std::accumulate(core.begin(), core.end(), 0, [](const nlohmann::json &a, const nlohmann::json &b)
                                            { return a["data"].size() + b["data"].size(); });
    HostFloatArray bodies(data_size);
    auto bodies_iterator = bodies.begin();
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
    HostFloatArray bodies(data_size);
    HostMatrix<int, INTSIZE> integers(n_bodies);
    HostMatrix<double, REALSIZE> floats(n_bodies);

    // Fill arrays
    std::size_t idx = 0;
    auto bodies_iterator = bodies.begin();
    for (auto body : core)
    {
        // Data
        std::copy(body["data"].begin(), body["data"].end(), bodies_iterator);

        // Integers
        integers.at(idx, FRAME) = body["metadata"]["frame"];
        integers.at(idx, DTYPE) = body["metadata"]["dtype"];
        integers.at(idx, TARGET) = body["metadata"]["target"];
        integers.at(idx, CENTER) = body["metadata"]["center"];
        integers.at(idx, NINTERVALS) = body["metadata"]["nintervals"];
        integers.at(idx, PDEG) = body["metadata"]["pdeg"];
        integers.at(idx, DATASIZE) = body["data"].size();
        // set data offset
        if (idx == 0)
        {
            integers.at(idx, DATAOFFSET) = 0;
        }
        else
        {
            integers.at(idx, DATAOFFSET) = integers.at(idx - 1, DATAOFFSET) + integers.at(idx - 1, DATASIZE);
        }

        // Floats
        floats.at(idx, INITIALEPOCH) = body["metadata"]["startEpoch"];
        floats.at(idx, FINALEPOCH) = body["metadata"]["finalEpoch"];

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

    HostMatrix<int, INTSIZE> integers(n_bodies);
    HostMatrix<double, REALSIZE> floats(n_bodies);
    HostFloatArray data(core["data"].size());

    std::copy(metadata["intMetadata"].begin(), metadata["intMetadata"].end(), integers.begin());
    std::copy(metadata["doubleMetadata"].begin(), metadata["doubleMetadata"].end(), floats.begin());

    auto data_json = core["data"];
    std::copy(data_json.begin(), data_json.end(), data.begin());

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
    std::string file = get_brie_file(json);

    // load cbor from given file
    nlohmann::json cbor_json = json_from_cbor(file);

    nlohmann::json layout = cbor_json["layout"];
    if (!layout.is_number() || (layout != 1 && layout != 2))
    {
        throw std::invalid_argument("brie data is neither v1 nor v2");
    }

    nlohmann::json core = cbor_json["core"];

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
    const double &epoch) const
{
    const int nintervals = nintervals_at(body_index);
    const int data_offset = dataoffset_at(body_index);
    const int pdeg = pdeg_at(body_index);
    const std::size_t n_indexes = pdeg + 1;

    // data = [ ...[other data; (data_offset)], interval radius, ...[intervals; (nintervals)], ...[coefficients; (nintervals * (pdeg + 1))] ]
    const double radius = data_at(data_offset);
    DeviceFloatArray intervals{
        .n_elements = (std::size_t)nintervals,
        .data = data.data + data_offset + 1,
    };
    const std::size_t n_coeff_vectors = (std::size_t)nintervals * n_indexes;
    const DeviceFloatArray coefficients{
        .n_elements = POSITION_DIM * n_coeff_vectors,
        .data = intervals.end(),
    };

    std::size_t idx = (std::size_t)((epoch - intervals.at(0)) / (2. * radius)); // interval selection
    double s = (epoch - intervals.at(idx)) / radius - 1.;                        // normalized  time coordinate
    // use clenshaw recurrence relation to efficiently calculate chebyshev polynomials
    PositionVector position{0.0};
    PositionVector w1{0.0};
    PositionVector w2{0.0};
    double s2 = 2. * s;
    // highestDegree = numIndexes - 1 = degree - 1 + 1 = pdeg - 1 + 1 = pdeg
    for (std::size_t degree = pdeg; degree > 0; --degree)
    {
        w2 = w1;
        w1 = position;
        PositionVector coeff_vector{};
        std::size_t offset = degree * nintervals;
        std::size_t coeff_index = offset + idx;
        for (std::size_t dim = 0; dim < POSITION_DIM; ++dim)
        {
            coeff_vector[dim] = coefficients.at(n_coeff_vectors * dim + coeff_index);
        }
        position = (w1 * s2 - w2) + coeff_vector;
    }
    { // degree = 0
        PositionVector coeff_vector{};
        // std::size_t offset = degree * nintervals = 0 * nintervals = 0;
        // std::size_t coeff_index = offset + idx = 0 + idx = idx;
        for (std::size_t dim = 0; dim < POSITION_DIM; ++dim)
        {
            coeff_vector[dim] = coefficients.at(n_coeff_vectors * dim + /* coeff_index */ idx);
        }
        position = (position * s - w1) + coeff_vector;
    }

    return position;
}

__device__ PositionVector DeviceEphemeris::read_position(
    const double &epoch,
    const int &target,
    const int &center) const
{
    PositionVector position{0.0};
    if (target == center)
    {
        return position;
    }

    int t = target;
    while (t != center)
    {
        std::size_t body_index = index_of_target(t);
        // ! We only have type 2 bodies for now.
        position += interpolate_type_2_body_to_position(body_index, epoch);
        t = center_at(body_index);
    }
    return position;
}

__device__ PositionVector DeviceEphemeris::calculate_position(const double &epoch, const int &target, const int &center_of_integration) const
{
    const int cc = common_center(target, center_of_integration);
    PositionVector xc = read_position(epoch, center_of_integration, cc);
    PositionVector xt = read_position(epoch, target, cc);
    PositionVector result = xt - xc;
    return result;
}

#pragma endregion