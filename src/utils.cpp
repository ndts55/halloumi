#include "utils.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>


nlohmann::json json_from_file(const std::string &path)
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

nlohmann::json json_from_cbor(const std::string &file)
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

void print_json(const nlohmann::json &json)
{
    if (json.is_string())
    {
        std::cout << json << std::endl;
    }
    else if (json.is_array())
    {
        std::cout << "array of size " << json.size() << std::endl;
    }
    else if (json.is_object())
    {
        for (auto it = json.begin(); it != json.end(); ++it)
        {
            std::cout << "key: " << it.key() << std::endl;
        }
    }
    else if (json.is_number())
    {
        std::cout << "Number: " << json << std::endl;
    }
    else if (json.is_boolean())
    {
        std::cout << "bool: " << json << std::endl;
    }
    else if (json.is_null())
    {
        std::cout << "Null" << std::endl;
    }
    else if (json.is_binary())
    {
        std::cout << "Binary with size " << json.size() << std::endl;
    }
    else
    {
        std::cout << "unsupported json type " << std::endl;
    }
}