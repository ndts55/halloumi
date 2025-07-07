#pragma once
#include <nlohmann/json.hpp>
#include "types.hpp"

// TODO implement some kind of mirror type that manages data on device
// TODO convert Ephemeris to RAII class

#pragma region Enums
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
#pragma endregion

#pragma region Metadata
class EphemerisMetadata
{
private:
    std::vector<Integer> integers;
    std::vector<Float> floats;

    std::vector<Integer> calculate_missing_naif_ids(const std::vector<Integer> other_naif_ids) const;

public:
    EphemerisMetadata(std::vector<Integer> ints, std::vector<Float> flts) : integers(ints), floats(flts) {}
    EphemerisMetadata(EphemerisMetadata &emd) : integers(emd.integers), floats(emd.floats) {}
    EphemerisMetadata merge_with(const EphemerisMetadata &other) const;
    std::vector<Integer> naif_ids() const;
    EphemerisMetadata make_subset(const std::vector<Integer> naif_ids) const;
    size_t n_bodies() const;

    friend EphemerisMetadata operator+(const EphemerisMetadata &lhs, const EphemerisMetadata &rhs)
    {
        return lhs.merge_with(rhs);
    }
};
#pragma endregion

#pragma region Ephemeris
class Ephemeris
{
private:
    EphemerisMetadata metadata;
    std::vector<Float> data;

public:
    static Ephemeris from_brie(const nlohmann::json &json);

    Ephemeris merge_with(const Ephemeris &other_ephemeris);

    Ephemeris(EphemerisMetadata md, std::vector<Float> d) : metadata(md), data(d) {}

    size_t n_bodies() const;
};

// struct Ephemeris
// {
//     EphemerisMetadata metadata;
//     std::vector<Float> data;
// };
#pragma endregion
