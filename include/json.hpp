#pragma once

#include <nlohmann/json.hpp>
#include <fstream>
#include <string>

namespace json
{

    inline nlohmann::json from_file(const std::string &path)
    {
        std::ifstream file(path);
        if (!file.is_open())
        {
            throw std::invalid_argument("File is not open: " + path);
        }
        if (!file.good())
        {
            throw std::runtime_error("File is not good: " + path);
        }

        return nlohmann::json::parse(file);
    }

    inline nlohmann::json from_cbor(const std::string &file)
    {
        // construct path, read from file, parse to json, success
        std::filesystem::path file_path = file;

        std::ifstream ifs(file_path, std::ios::binary);
        if (!ifs.is_open())
        {
            throw std::runtime_error("ifs is not open: " + file_path.string());
        }
        if (!ifs.good())
        {
            throw std::runtime_error("ifs not good: " + file_path.string());
        }

        std::vector<unsigned char> buffer(std::istreambuf_iterator<char>(ifs), {});

        return nlohmann::json::from_cbor(buffer);
    }

} // namespace json
