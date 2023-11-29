#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include "tensor.h"
#include <string>
#include <unordered_map>

struct Weight {
    std::vector<size_t> shape;
    std::vector<float> data;
};

using WeightsMap = std::unordered_map<std::string, Tensor>;

WeightsMap loadModelWeights(const std::string& file_path);

#endif // UTILS_H
