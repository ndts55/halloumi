#pragma once
#include <nlohmann/json.hpp>
#include "types.hpp"

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
    // TODO convert to cuda arrays
    std::vector<Integer> integers;
    std::vector<Float> floats;

    std::vector<Integer> calculate_missing_naif_ids(const std::vector<Integer> other_naif_ids) const;

public:
    EphemerisMetadata(const EphemerisMetadata &&emd) : integers(emd.integers), floats(emd.floats) {}
    EphemerisMetadata(std::vector<Integer> &&ints, std::vector<Float> &&flts) : integers(ints), floats(flts) {}

    EphemerisMetadata make_subset(const std::vector<Integer> naif_ids) const;
    EphemerisMetadata merge_with(const EphemerisMetadata &other) const;
    std::vector<Integer> naif_ids() const;
    inline std::size_t n_bodies() const { return integers.size(); }

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
    // TODO convert to cuda array
    std::vector<Float> data;
    EphemerisMetadata metadata;

public:
    Ephemeris(const Ephemeris &&e) : metadata(std::move(e.metadata)), data(std::move(e.data)) {}
    Ephemeris(EphemerisMetadata &&md, std::vector<Float> &&d) : metadata(std::move(md)), data(std::move(d)) {}

    static Ephemeris from_brie(const nlohmann::json &json);

    Ephemeris merge_with(const Ephemeris &other_ephemeris);
    inline std::size_t n_bodies() const { return metadata.n_bodies(); }

    Ephemeris &operator=(const Ephemeris &&e);
};
#pragma endregion
