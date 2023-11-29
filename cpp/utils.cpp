#include "utils.h"

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>

#include "tensor.h"

using json = nlohmann::json;

WeightsMap loadModelWeights(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Error: Unable to open file " + file_path);
    }

    nlohmann::json param_map;
    try {
        file >> param_map;
    } catch (const std::exception& e) {
        throw std::runtime_error("Error while reading JSON data: " + std::string(e.what()));
    }

    WeightsMap weights_map;
    for (auto& [name, param] : param_map.items()) {
        std::vector<size_t> shape;
        for (const auto& val : param["shape"]) {
            shape.push_back(val.get<size_t>());
        }

        std::vector<float> data;
        for (const auto& val : param["data"]) {
            data.push_back(val.get<float>());
        }

        Tensor tensor(shape, data);
        weights_map.emplace(name, std::move(tensor));
    }

    return weights_map;
}
