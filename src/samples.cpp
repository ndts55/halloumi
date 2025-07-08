#include <iostream>
#include <format>
#include <nlohmann/json.hpp>
#include "utils.hpp"
#include "samples.hpp"

Samples::Samples(const std::vector<Float> &&d, const std::vector<Float> &&e)
{
    this->data = std::move(d);
    this->epochs = std::move(e);
    this->n_vecs = this->data.size() / STATE_DIM;
    if ((this->n_vecs * STATE_DIM) != this->data.size() || this->n_vecs != this->epochs.size())
    {
        throw std::runtime_error(
            "You gave me some malformed data:\n Expected data size:" + std::to_string(this->n_vecs) + "\tActual data size:" + std::to_string(this->data.size()) + "\nExpected epochs size:" + std::to_string(this->n_vecs) + "\tActual epochs size:" + std::to_string(this->epochs.size()) + "\n");
    }
}

size_t Samples::size() const
{
    return this->n_vecs;
}

Samples Samples::from_json(const nlohmann::json &json)
{
    auto in_samples = json["inSamples"];
    auto n_vecs = in_samples.size();

    std::vector<Float> data(n_vecs * STATE_DIM, 0.0);
    std::vector<Float> epochs;
    epochs.reserve(n_vecs);

    size_t idx = 0;
    for (auto sample : in_samples)
    {
        epochs.push_back(sample["epoch"]);
        auto cart = sample["stateCart"];
        for (auto dim = 0; dim < STATE_DIM; dim++)
        {
            // put each element of vec<6> in the correct position
            data[get_index(n_vecs, dim, idx)] = cart[dim];
        }
        // TODO set metadata as COI (even though it is always 399 (Earth))
    }

    return Samples(std::move(data), std::move(epochs));
}
