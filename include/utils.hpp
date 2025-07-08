#pragma once
#include <nlohmann/json.hpp>
#include <fstream>
#include <string>

nlohmann::json json_from_file(const std::string &path);

nlohmann::json json_from_cbor(const std::string &file);

size_t get_index(const size_t &n_vecs, const size_t &dim, const size_t &index);
