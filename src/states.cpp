#include <nlohmann/json.hpp>
#include "utils.hpp"
#include "states.hpp"

StatesArray::StatesArray(const std::vector<Float> &&d)
{
    this->data = std::move(d);
    this->n_vecs = this->data.size() / STATE_DIM;
    if ((this->n_vecs * STATE_DIM) != this->data.size())
    {
        throw std::runtime_error("what is this?");
    }
}

size_t StatesArray::size() const
{
    return this->n_vecs;
}

StatesArray StatesArray::from_json(const nlohmann::json &json)
{
    auto in_samples = json["inSamples"];
    auto n_vecs = in_samples.size();

    std::vector<Float> data(n_vecs * STATE_DIM, 0.0);

    size_t idx = 0;
    for (auto sample : in_samples)
    {
        auto cart = sample["stateCart"];
        for (auto dim = 0; dim < STATE_DIM; dim++)
        {
            // put each element of vec<6> in the correct position
            data[get_index(n_vecs, dim, idx)] = cart[dim];
        }
        // TODO set metadata as COI (even though it is always 399 (Earth))
    }

    return StatesArray(std::move(data));
}

template <size_t dim>
const Float &StatesArray::operator[](const size_t &idx) const
{
    return this->data[get_index(this->n_vecs, dim, idx)];
}

template <size_t dim>
Float &StatesArray::operator[](const size_t &idx)
{

    return this->data[get_index(this->n_vecs, dim, idx)];
}