#include "simulation/ephemeris.hpp"
#include <iostream>
#include <nlohmann/json.hpp>
#include <fstream>
#include "utils.hpp"
#include "cuda/cuda_array.hpp"

#pragma region Layout v1
CudaArray1D<Float> load_ephemeris_data_v1(const nlohmann::json &core)
{
    std::size_t data_size = std::accumulate(core.begin(), core.end(), 0, [](const nlohmann::json &a, const nlohmann::json &b)
                                            { return a["data"].size() + b["data"].size(); });
    CudaArray1D<Float> bodies(data_size);
    std::size_t idx = 0;
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