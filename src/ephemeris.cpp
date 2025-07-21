#include <iostream>
#include <nlohmann/json.hpp>
#include <fstream>
#include "utils.hpp"
#include "ephemeris.hpp"

std::size_t Ephemeris::n_bodies() const
{
    return this->metadata.n_bodies();
}
std::size_t EphemerisMetadata::n_bodies() const
{
    return this->integers.size();
}

// TODO copy ephemeris data to device
// TODO provide access to ephemeris data on device

#pragma region Layout v1
EphemerisMetadata load_ephemeris_metadata_v1(const nlohmann::json &core)
{
    auto n_bodies = core.size();
    // create std::vector with correct size
    std::vector<Integer> integers(n_bodies * INTSIZE, 0);
    std::vector<Float> floats(n_bodies * REALSIZE, 0.0);
    // load metadata from AoS to SoA using IntMembers and RealMembers enums
    std::size_t idx = 0;
    for (auto body : core)
    {
        auto metadata = body["metadata"];
        integers[get_2d_index(n_bodies, FRAME, idx)] = metadata["frame"];
        integers[get_2d_index(n_bodies, DTYPE, idx)] = metadata["dtype"];
        integers[get_2d_index(n_bodies, TARGET, idx)] = metadata["target"];
        integers[get_2d_index(n_bodies, CENTER, idx)] = metadata["center"];
        integers[get_2d_index(n_bodies, NINTERVALS, idx)] = metadata["nintervals"];
        integers[get_2d_index(n_bodies, PDEG, idx)] = metadata["pdeg"];
        integers[get_2d_index(n_bodies, DATASIZE, idx)] = body["data"].size();
        // set data offset
        if (idx == 0)
        {
            integers[get_2d_index(n_bodies, DATAOFFSET, idx)] = 0;
        }
        else
        {
            integers[get_2d_index(n_bodies, DATAOFFSET, idx)] = integers[get_2d_index(n_bodies, DATAOFFSET, idx - 1)] + integers[get_2d_index(n_bodies, DATASIZE, idx - 1)];
        }
        floats[get_2d_index(n_bodies, INITIALEPOCH, idx)] = metadata["startEpoch"];
        floats[get_2d_index(n_bodies, FINALEPOCH, idx)] = metadata["finalEpoch"];
        idx += 1;
    }

    return EphemerisMetadata(std::move(integers), std::move(floats));
}

std::vector<Float> load_ephemeris_data_v1(const nlohmann::json &core)
{
    std::size_t data_size = std::accumulate(core.begin(), core.end(), 0, [](const nlohmann::json &a, const nlohmann::json &b)
                                            { return a["data"].size() + b["data"].size(); });
    std::vector<Float> data(data_size, 0.0);
    std::size_t idx = 0;
    auto data_iterator = data.begin();
    for (auto body : core)
    {
        std::copy(body["data"].begin(), body["data"].end(), data_iterator);
        std::advance(data_iterator, body["data"].size());
    }
    return data;
}

Ephemeris load_brie_v1(const nlohmann::json &core)
{
    std::cout << "loading brie v1" << std::endl;
    auto metadata = load_ephemeris_metadata_v1(core);
    auto data = load_ephemeris_data_v1(core);
    return Ephemeris(std::move(metadata), std::move(data));
}
#pragma endregion

#pragma region Layout v2
Ephemeris load_brie_v2(const nlohmann::json &core)
{
    auto metadata = core["metadata"];
    std::size_t n_bodies = metadata["nBodyUnits"];

    std::vector<Integer> integers(n_bodies * INTSIZE, 0);
    std::vector<Float> floats(n_bodies * REALSIZE, 0.0);
    std::vector<Float> data(core["data"].size(), 0.0);

    std::copy(metadata["intMetadata"].begin(), metadata["intMetadata"].end(), integers.begin());
    std::copy(metadata["doubleMetadata"].begin(), metadata["doubleMetadata"].end(), floats.begin());

    auto data_json = core["data"];
    std::copy(data_json.begin(), data_json.end(), data.begin());

    return Ephemeris(EphemerisMetadata(std::move(integers), std::move(floats)), std::move(data));
}
#pragma endregion

#pragma region Single Brie File
Ephemeris ephemeris_from_brie_file(const std::string &file)
{
    // load cbor from given file
    auto cbor_json = json_from_cbor(file);

    std::ofstream o("output.json");
    o << cbor_json << std::endl;

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

#pragma region Brie File Array
Ephemeris ephemeris_from_brie_file_array(const nlohmann::json &array)
{
    auto ephemeris = ephemeris_from_brie_file(array[0]);
    // TODO re-write with std::reduce because that's actually what we're doing here
    for (auto i = 1; i < array.size(); i++)
    {
        auto other_ephemeris = ephemeris_from_brie_file(array[i]);
        ephemeris = ephemeris.merge_with(other_ephemeris);
    }
    return ephemeris;
}
Ephemeris Ephemeris::merge_with(const Ephemeris &other)
{
    // TODO merge ephemeris data, if reading in multiple brie files
    throw std::runtime_error("Not implemented");
    /*
    __host__ AugEphUnit merge(AugEphUnit& other, const cudaStream_t& stream = 0)
    {

        /* Create actual merge subset and extract the missing metadata
        BodyArrayT otherIDs = other.metadata_.getNaifIDs();
        BodyArrayT missingIDs = this->metadata_.getMissingNaifIDs(otherIDs);
        /* Create the merged array of IDs
        AugEphUnit<UseTexture> missing = other.makeSubset(std::move(missingIDs));

        /* Create new metadata
        MetadataT metadata = this->metadata_.merge(missing.metadata_);

        /* Create new empty data
        DataT data(metadata.totalDataSize());

        /* Create new active bodies
        /* Get missing active bodies
        BodyArrayT mabodies = this->missingActiveBodies(other.activeBodies_);
        /* Create new active bodies array and fill it with memcpy
        BodyArrayT newActiveBodies(
            this->activeBodies_.size() + mabodies.size());
        /* this active bodies part
        cudaMemcpyAsync(newActiveBodies.getHostData(),
                        this->activeBodies_.getHostData(),
                        this->activeBodies_.size() * sizeof(NaifId), cudaMemcpyHostToHost,
                        stream);
        /* Missing active bodies part
        cudaMemcpyAsync(
            newActiveBodies.getHostData() + this->activeBodies_.size(),
            mabodies.getHostData(), mabodies.size() * sizeof(NaifId),
            cudaMemcpyHostToHost, stream);

        /* Now create new AugEphUnit
        AugEphUnit<UseTexture> ephs(
            std::move(metadata), std::move(data), std::move(newActiveBodies));

        /* Copy in the data batches
        cudaMemcpyAsync(ephs.data_.getHostData(), this->data_.getHostData(),
                        this->data_.size() * sizeof(Real), cudaMemcpyHostToHost, stream);
        cudaMemcpyAsync(ephs.data_.getHostData() + this->data_.size(),
                        missing.data_.getHostData(), missing.data_.size() * sizeof(Real),
                        cudaMemcpyHostToHost, stream);

        /* Finally return
        return ephs;
    }
    */
}

EphemerisMetadata EphemerisMetadata::merge_with(const EphemerisMetadata &other) const
{
    throw std::runtime_error("Not implemented");
    // auto missing_metadata = other.make_subset(this->calculate_missing_naif_ids(other.naif_ids()));
    // concatenate integers
    // std::vector<Integer> integers;
    // integers.reserve(this->integers.size() + other.integers.size());
    // integers.insert(integers.end(), this->integers.begin(), this->integers.end());
    // integers.insert(integers.end(), other.integers.begin(), other.integers.end());

    // concatenate floats
    // std::vector<Float> floats;
    // floats.reserve(this->floats.size() + other.floats.size());
    // floats.insert(floats.end(), this->floats.begin(), this->floats.end());
    // floats.insert(floats.end(), other.floats.begin(), other.floats.end());

    // TODO re-caclulate data offsets
    /*
    std::size_t offset = 0;
    for (idx_t i = 0; i < this->size(); i++) {
        this->getDataOffset(i) = offset;
        offset += this->getDataSize(i);
    }
    */
}
std::vector<Integer> EphemerisMetadata::naif_ids() const
{
    throw std::runtime_error("Not implemented");
}
EphemerisMetadata EphemerisMetadata::make_subset(const std::vector<Integer> naif_ids) const
{
    throw std::runtime_error("Not implemented");
}
#pragma endregion

Ephemeris Ephemeris::from_brie(const nlohmann::json &json)
{
    // TODO handle paths and naif ids
    if (json.is_string())
    {
        return ephemeris_from_brie_file(json);
    }
    else if (json.is_array())
    {
        return ephemeris_from_brie_file_array(json);
    }
    throw std::invalid_argument("received json that is object instead of string or array");
}