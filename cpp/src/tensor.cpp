#include "tensor.h"
#include <cmath>
#include <tuple>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>  // for std::out_of_range
#include <string>
#include <vector>

Tensor::Tensor(const std::vector<size_t>& shape, const std::vector<float>& data) {
    this->shape = shape;
    this->data = data;
    this->strides = std::vector<size_t>(shape.size());
    this->strides[shape.size() - 1] = 1;
    for (int i = shape.size() - 2; i >= 0; i--) {
        this->strides[i] = this->strides[i + 1] * shape[i + 1];
    }
}

Tensor::Tensor(const std::vector<size_t>& shape, const std::vector<float>& data, const std::vector<size_t>& strides) {
    this->shape = shape;
    this->data = data;
    this->strides = strides;
}

// stuff for json weights
Tensor::Tensor() : shape{}, data{}, strides{} {}
Tensor::Tensor(const Tensor& other) : shape(other.shape), data(other.data), strides(other.strides) {}
Tensor::Tensor(Tensor&& other) noexcept : shape(std::move(other.shape)), data(std::move(other.data)), strides(std::move(other.strides)) {}
Tensor& Tensor::operator=(Tensor other) {
    std::swap(shape, other.shape);
    std::swap(data, other.data);
    std::swap(strides, other.strides);
    return *this;
}
Tensor::~Tensor() {}

// Utility function to calculate the flat index
const size_t Tensor::calculate_flat_index(const std::vector<size_t>& index) const {
    size_t flat_index = 0;
    for (size_t i = 0; i < index.size(); ++i) {
        flat_index += index[i] * strides[i];
    }
    return flat_index;
}

// Overload for const indexing
const float& Tensor::operator[](const std::vector<size_t>& index) const {
    size_t flat_index = calculate_flat_index(index);
    if (flat_index >= data.size()) {
        throw std::out_of_range("Index out of bounds");
    }
    return data[flat_index];
}

// Overload for mutable indexing
float& Tensor::operator[](const std::vector<size_t>& index) {
    size_t flat_index = calculate_flat_index(index);
    if (flat_index >= data.size()) {
        throw std::out_of_range("Index out of bounds");
    }
    return data[flat_index];
}

Tensor Tensor::clone() const { return Tensor(shape, data); }

Tensor operator+(const Tensor& lhs, const Tensor& rhs) {
    assert(lhs.shape == rhs.shape && "Shapes of tensors must match for addition");

    std::vector<float> data;
    data.reserve(lhs.data.size());

    for (size_t i = 0; i < lhs.data.size(); ++i) {
        data.push_back(lhs.data[i] + rhs.data[i]);
    }

    return Tensor(lhs.shape, data, lhs.strides);
}

Tensor operator-(const Tensor& lhs, const Tensor& rhs) {
    assert(lhs.shape == rhs.shape && "Shapes of tensors must match for addition");

    std::vector<float> data;
    data.reserve(lhs.data.size());

    for (size_t i = 0; i < lhs.data.size(); ++i) {
        data.push_back(lhs.data[i] - rhs.data[i]);
    }

    return Tensor(lhs.shape, data, lhs.strides);
}

// Additional member functions and methods...
Tensor Tensor::arange(size_t start, size_t end, size_t step) {
    std::vector<float> data;
    for (size_t i = start; i < end; i += step) {
        data.push_back(static_cast<float>(i));
    }
    return Tensor({data.size()}, data);
}

Tensor Tensor::reshape(const std::vector<size_t>& new_shape) const {
    size_t total_size = 1;
    for (const auto& dim : new_shape) {
        total_size *= dim;
    }
    assert(total_size == data.size());

    std::vector<size_t> new_strides(new_shape.size());
    size_t stride = 1;
    for (int i = new_shape.size() - 1; i >= 0; --i) {
        new_strides[i] = stride;
        stride *= new_shape[i];
    }

    return Tensor(new_shape, data, new_strides);
}

Tensor Tensor::transpose() const {
    assert(shape.size() == 2);
    std::vector<float> new_data(shape[0] * shape[1], 0);
    Tensor output({shape[1], shape[0]}, new_data);

    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            output.data[j * shape[0] + i] = data[i * shape[1] + j];
        }
    }
    return output;
}

Tensor Tensor::permute_multihead() const {
    assert(shape.size() == 4);
    std::vector<float> new_data(shape[0] * shape[1] * shape[2] * shape[3], 0);
    Tensor output({shape[0], shape[2], shape[1], shape[3]}, new_data);

    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            for (size_t k = 0; k < shape[2]; ++k) {
                for (size_t l = 0; l < shape[3]; ++l) {
                    size_t original_index = i * strides[0] + j * strides[1] + k * strides[2] + l * strides[3];
                    size_t new_index = i * output.strides[0] + k * output.strides[1] + j * output.strides[2] + l * output.strides[3];
                    output.data[new_index] = data[original_index];
                }
            }
        }
    }
    return output;
}


std::tuple<Tensor, Tensor, Tensor> Tensor::split3(size_t dim) const {
    assert(dim < shape.size());
    size_t bsz = shape[0], seq = shape[1], hidden = shape[2];

    // Adjust the sizes for the output tensors
    std::vector<size_t> new_shape = {bsz, seq, hidden / 3};
    std::vector<float> output_data1(bsz * seq * hidden / 3, 0);
    std::vector<float> output_data2(bsz * seq * hidden / 3, 0);
    std::vector<float> output_data3(bsz * seq * hidden / 3, 0);

    Tensor output1(new_shape, output_data1);
    Tensor output2(new_shape, output_data2);
    Tensor output3(new_shape, output_data3);

    for (size_t i = 0; i < bsz; ++i) {
        for (size_t j = 0; j < seq; ++j) {
            for (size_t k = 0; k < hidden; ++k) {
                if (k < hidden / 3) {
                    output1.data[i * seq * (hidden / 3) + j * (hidden / 3) + k] = data[i * seq * hidden + j * hidden + k];
                } else if (k < 2 * hidden / 3) {
                    output2.data[i * seq * (hidden / 3) + j * (hidden / 3) + k - hidden / 3] = data[i * seq * hidden + j * hidden + k];
                } else {
                    output3.data[i * seq * (hidden / 3) + j * (hidden / 3) + k - 2 * hidden / 3] = data[i * seq * hidden + j * hidden + k];
                }
            }
        }
    }

    return std::make_tuple(output1, output2, output3);
}

// reduction functions
Tensor Tensor::max4d(size_t dim, bool keepdim) const {
    if (dim >= shape.size() || shape.size() != 4) {
        throw std::out_of_range("Invalid dimension or tensor is not 4D");
    }

    // Extract dimensions
    size_t B = shape[0], H = shape[1], T = shape[2], D = shape[3];

    std::vector<size_t> output_shape = shape;
    output_shape[dim] = 1;

    Tensor output(output_shape, std::vector<float>(std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<size_t>()), 0.0f));

    for (size_t i = 0; i < B; ++i) {
        for (size_t j = 0; j < H; ++j) {
            for (size_t k = 0; k < T; ++k) {
                for (size_t l = 0; l < D; ++l) {
                    size_t index = i * strides[0] + j * strides[1] + k * strides[2] + l * strides[3];
                    switch (dim) {
                        case 0:
                            output[{0, j, k, l}] = std::max(output[{0, j, k, l}], data[index]);
                            break;
                        case 1:
                            output[{i, 0, k, l}] = std::max(output[{i, 0, k, l}], data[index]);
                            break;
                        case 2:
                            output[{i, j, 0, l}] = std::max(output[{i, j, 0, l}], data[index]);
                            break;
                        case 3:
                            output[{i, j, k, 0}] = std::max(output[{i, j, k, 0}], data[index]);
                            break;
                        default:
                            throw std::out_of_range("Invalid dimension");
                    }
                }
            }
        }
    }

    if (!keepdim) {
        output_shape.erase(output_shape.begin() + dim);
        output = output.reshape(output_shape);
    }

    return output;
}

Tensor Tensor::sum4d(size_t dim, bool keepdim) const {
    if (dim >= shape.size() || shape.size() != 4) {
        throw std::out_of_range("Invalid dimension or tensor is not 4D");
    }

    size_t B = shape[0], H = shape[1], T = shape[2], D = shape[3];

    std::vector<size_t> output_shape = shape;
    output_shape[dim] = 1;

    Tensor output(output_shape, std::vector<float>(std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<size_t>()), 0.0f));

    for (size_t i = 0; i < B; ++i) {
        for (size_t j = 0; j < H; ++j) {
            for (size_t k = 0; k < T; ++k) {
                for (size_t l = 0; l < D; ++l) {
                    size_t index = i * strides[0] + j * strides[1] + k * strides[2] + l * strides[3];
                    switch (dim) {
                        case 0:
                            output[{0, j, k, l}] += data[index];
                            break;
                        case 1:
                            output[{i, 0, k, l}] += data[index];
                            break;
                        case 2:
                            output[{i, j, 0, l}] += data[index];
                            break;
                        case 3:
                            output[{i, j, k, 0}] += data[index];
                            break;
                        default:
                            throw std::out_of_range("Invalid dimension");
                    }
                }
            }
        }
    }

    if (!keepdim) {
        output_shape.erase(output_shape.begin() + dim);
        output = output.reshape(output_shape);
    }

    return output;
}

Tensor Tensor::mean(size_t dim, bool keepdim) const {
    if (dim >= shape.size() || shape.size() != 3) {
        throw std::out_of_range("Invalid dimension or tensor is not 3D");
    }

    size_t B = shape[0], T = shape[1], D = shape[2];

    std::vector<size_t> output_shape = shape;
    output_shape[dim] = 1;

    Tensor output(output_shape, std::vector<float>(std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<size_t>()), 0.0f));

    for (size_t i = 0; i < B; ++i) {
        for (size_t j = 0; j < T; ++j) {
            for (size_t k = 0; k < D; ++k) {
                size_t index = i * strides[0] + j * strides[1] + k * strides[2];
                switch (dim) {
                    case 0:
                        output[{0, j, k}] += data[index];
                        break;
                    case 1:
                        output[{i, 0, k}] += data[index];
                        break;
                    case 2:
                        output[{i, j, 0}] += data[index];
                        break;
                    default:
                        throw std::out_of_range("Invalid dimension");
                }
            }
        }
    }

    // Divide each element by the size of the dimension to get the mean
    for (auto& val : output.data) {
        val /= shape[dim];
    }

    if (!keepdim) {
        output_shape.erase(output_shape.begin() + dim);
        output = output.reshape(output_shape);
    }

    return output;
}

Tensor Tensor::variance(size_t dim, bool keepdim) const {
    Tensor mean_tensor = mean(dim, true);  // Keep dimension for subtraction

    // Subtract the mean and square the result
    Tensor x_minus_mean = *this;  // Assuming you have an appropriate copy constructor
    for (size_t i = 0; i < data.size(); ++i) {
        x_minus_mean.data[i] = std::pow(data[i] - mean_tensor.data[i], 2);
    }

    // Compute the mean of the squared differences
    Tensor var = x_minus_mean.mean(dim, keepdim);

    return var;
}

// unary ops
Tensor Tensor::exp() const {
    std::vector<float> result(data.size());
    for (int i = 0; i < data.size(); i++) {
        result[i] = std::exp(data[i]);
    }
    return Tensor(shape, result);
}

Tensor Tensor::sqrt() const {
    std::vector<float> result(data.size());
    for (int i = 0; i < data.size(); i++) {
        result[i] = std::sqrt(data[i]);
    }
    return Tensor(shape, result);
}

Tensor Tensor::add_scalar(float scalar) const {
    std::vector<float> result(data.size());
    for (int i = 0; i < data.size(); i++) {
        result[i] = data[i] + scalar;
    }
    return Tensor(shape, result);
}

// Tensor sub(float scalar) const {
//     std::vector<float> result(data.size());
//     for (int i = 0; i < data.size(); i++) {
//         result[i] = data[i] - scalar;
//     }
//     return Tensor(shape, result);
// }

// Tensor mul(float scalar) const {
//     std::vector<float> result(data.size());
//     for (int i = 0; i < data.size(); i++) {
//         result[i] = data[i] * scalar;
//     }
//     return Tensor(shape, result);
// }

Tensor Tensor::div_scalar(float scalar) const {
    std::vector<float> result(data.size());
    for (int i = 0; i < data.size(); i++) {
        result[i] = data[i] / scalar;
    }
    return Tensor(shape, result);
}

// elementwise operations
Tensor Tensor::add(const Tensor& other) const {
    if (this->shape == other.shape) {
        // Direct addition
        Tensor output(this->shape, std::vector<float>(this->data.size(), 0.0f));
        for (size_t i = 0; i < this->data.size(); ++i) {
            output.data[i] = this->data[i] + other.data[i];
        }
        return output;
    } else if (this->shape.size() == 2 && other.shape.size() == 2) {
        // 2D broadcasting
        Tensor output(this->shape, std::vector<float>(this->data.size(), 0.0f));
        for (size_t i = 0; i < this->shape[0]; ++i) {
            for (size_t j = 0; j < this->shape[1]; ++j) {
                size_t index_self = i * this->strides[0] + j * this->strides[1];
                size_t index_other = (i % other.shape[0]) * other.strides[0] + (j % other.shape[1]) * other.strides[1];
                output.data[index_self] = this->data[index_self] + other.data[index_other];
            }
        }
        return output;
    } else if (this->shape.size() == 2 && other.shape.size() == 1) {
        // 2D broadcasting with 1D
        Tensor output(this->shape, std::vector<float>(this->data.size(), 0.0f));
        for (size_t i = 0; i < this->shape[0]; ++i) {
            for (size_t j = 0; j < this->shape[1]; ++j) {
                size_t index_self = i * this->strides[0] + j * this->strides[1];
                size_t index_other = j % other.shape[0] * other.strides[0];
                output.data[index_self] = this->data[index_self] + other.data[index_other];
            }
        }
        return output;
    } else if (this->shape.size() == 3 && other.shape.size() == 3) {
        // 3D broadcasting
        Tensor output(this->shape, std::vector<float>(this->data.size(), 0.0f));
        for (size_t i = 0; i < this->shape[0]; ++i) {
            for (size_t j = 0; j < this->shape[1]; ++j) {
                for (size_t k = 0; k < this->shape[2]; ++k) {
                    size_t index_self = i * this->strides[0] + j * this->strides[1] + k * this->strides[2];
                    size_t index_other =
                        (i % other.shape[0]) * other.strides[0] + (j % other.shape[1]) * other.strides[1] + (k % other.shape[2]) * other.strides[2];
                    output.data[index_self] = this->data[index_self] + other.data[index_other];
                }
            }
        }
        return output;
    } else {
        throw std::invalid_argument("Tensor shapes do not match and are not suitable for broadcasting");
    }
}

Tensor Tensor::sub(const Tensor& other) const {
    if (this->shape == other.shape) {
        // Direct subtraction
        Tensor output(this->shape, std::vector<float>(this->data.size(), 0.0f));
        for (size_t i = 0; i < this->data.size(); ++i) {
            output.data[i] = this->data[i] - other.data[i];
        }
        return output;
    } else if (this->shape.size() == 2 && other.shape.size() == 2) {
        // 2D broadcasting
        Tensor output(this->shape, std::vector<float>(this->data.size(), 0.0f));
        for (size_t i = 0; i < this->shape[0]; ++i) {
            for (size_t j = 0; j < this->shape[1]; ++j) {
                size_t index_self = i * this->strides[0] + j * this->strides[1];
                size_t index_other = (i % other.shape[0]) * other.strides[0] + (j % other.shape[1]) * other.strides[1];
                output.data[index_self] = this->data[index_self] - other.data[index_other];
            }
        }
        return output;
    } else if (this->shape.size() == 2 && other.shape.size() == 1) {
        // 2D broadcasting with 1D
        Tensor output(this->shape, std::vector<float>(this->data.size(), 0.0f));
        for (size_t i = 0; i < this->shape[0]; ++i) {
            for (size_t j = 0; j < this->shape[1]; ++j) {
                size_t index_self = i * this->strides[0] + j * this->strides[1];
                size_t index_other = j % other.shape[0] * other.strides[0];
                output.data[index_self] = this->data[index_self] - other.data[index_other];
            }
        }
        return output;
    } else if (this->shape.size() == 3 && other.shape.size() == 3) {
        // 3D broadcasting
        Tensor output(this->shape, std::vector<float>(this->data.size(), 0.0f));
        for (size_t i = 0; i < this->shape[0]; ++i) {
            for (size_t j = 0; j < this->shape[1]; ++j) {
                for (size_t k = 0; k < this->shape[2]; ++k) {
                    size_t index_self = i * this->strides[0] + j * this->strides[1] + k * this->strides[2];
                    size_t index_other =
                        (i % other.shape[0]) * other.strides[0] + (j % other.shape[1]) * other.strides[1] + (k % other.shape[2]) * other.strides[2];
                    output.data[index_self] = this->data[index_self] - other.data[index_other];
                }
            }
        }
        return output;
    } else {
        throw std::invalid_argument("Tensor shapes do not match and are not suitable for broadcasting");
    }
}

Tensor Tensor::mul(const Tensor& other) const {
    if (this->shape == other.shape) {
        // Direct multiplication
        Tensor output(this->shape, std::vector<float>(this->data.size(), 0.0f));
        for (size_t i = 0; i < this->data.size(); ++i) {
            output.data[i] = this->data[i] * other.data[i];
        }
        return output;
    } else if (this->shape.size() == 2 && other.shape.size() == 2) {
        // 2D broadcasting
        Tensor output(this->shape, std::vector<float>(this->data.size(), 0.0f));
        for (size_t i = 0; i < this->shape[0]; ++i) {
            for (size_t j = 0; j < this->shape[1]; ++j) {
                size_t index_self = i * this->strides[0] + j * this->strides[1];
                size_t index_other = (i % other.shape[0]) * other.strides[0] + (j % other.shape[1]) * other.strides[1];
                output.data[index_self] = this->data[index_self] * other.data[index_other];
            }
        }
        return output;
    } else if (this->shape.size() == 3 && other.shape.size() == 3) {
        // 3D broadcasting
        Tensor output(this->shape, std::vector<float>(this->data.size(), 0.0f));
        for (size_t i = 0; i < this->shape[0]; ++i) {
            for (size_t j = 0; j < this->shape[1]; ++j) {
                for (size_t k = 0; k < this->shape[2]; ++k) {
                    size_t index_self = i * this->strides[0] + j * this->strides[1] + k * this->strides[2];
                    size_t index_other =
                        (i % other.shape[0]) * other.strides[0] + (j % other.shape[1]) * other.strides[1] + (k % other.shape[2]) * other.strides[2];
                    output.data[index_self] = this->data[index_self] * other.data[index_other];
                }
            }
        }
        return output;
    } else {
        throw std::invalid_argument("Tensor shapes do not match and are not suitable for broadcasting");
    }
}

Tensor Tensor::div(const Tensor& other) const {
    if (this->shape == other.shape) {
        // Direct division
        Tensor output(this->shape, std::vector<float>(this->data.size(), 0.0f));
        for (size_t i = 0; i < this->data.size(); ++i) {
            output.data[i] = this->data[i] / other.data[i];
        }
        return output;
    } else if (this->shape.size() == 2 && other.shape.size() == 2) {
        // 2D broadcasting
        Tensor output(this->shape, std::vector<float>(this->data.size(), 0.0f));
        for (size_t i = 0; i < this->shape[0]; ++i) {
            for (size_t j = 0; j < this->shape[1]; ++j) {
                size_t index_self = i * this->strides[0] + j * this->strides[1];
                size_t index_other = (i % other.shape[0]) * other.strides[0] + (j % other.shape[1]) * other.strides[1];
                output.data[index_self] = this->data[index_self] / other.data[index_other];
            }
        }
        return output;
    } else if (this->shape.size() == 3 && other.shape.size() == 3) {
        // 3D broadcasting
        Tensor output(this->shape, std::vector<float>(this->data.size(), 0.0f));
        for (size_t i = 0; i < this->shape[0]; ++i) {
            for (size_t j = 0; j < this->shape[1]; ++j) {
                for (size_t k = 0; k < this->shape[2]; ++k) {
                    size_t index_self = i * this->strides[0] + j * this->strides[1] + k * this->strides[2];
                    size_t index_other =
                        (i % other.shape[0]) * other.strides[0] + (j % other.shape[1]) * other.strides[1] + (k % other.shape[2]) * other.strides[2];
                    output.data[index_self] = this->data[index_self] / other.data[index_other];
                }
            }
        }
        return output;
    } else {
        throw std::invalid_argument("Tensor shapes do not match and are not suitable for broadcasting");
    }
}

// matmuls
Tensor Tensor::matmul(const Tensor& other) const {
    assert(shape.size() == 2);
    assert(other.shape.size() == 2);
    assert(other.shape[0] == shape[1]);

    std::vector<float> output_data(shape[0] * other.shape[1], 0);
    Tensor output({shape[0], other.shape[1]}, output_data);

    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < other.shape[1]; ++j) {
            for (size_t k = 0; k < shape[1]; ++k) {
                output.data[i * shape[1] + j] += data[i * shape[1] + k] * other.data[k * other.shape[1] + j];
            }
        }
    }

    return output;
}

// debug prints
std::string format_1d(const Tensor& tensor, size_t each_side_count) {
    std::stringstream ss;
    size_t size = tensor.data.size();

    if (size < 2 * each_side_count + 1) {
        for (float value : tensor.data) {
            ss << std::fixed << std::setprecision(4) << value << ", ";
        }
    } else {
        for (size_t i = 0; i < each_side_count; ++i) {
            ss << std::fixed << std::setprecision(4) << tensor.data[i] << ", ";
        }
        ss << "..., ";
        for (size_t i = size - each_side_count; i < size; ++i) {
            ss << std::fixed << std::setprecision(4) << tensor.data[i] << ", ";
        }
    }

    return "[" + ss.str() + "]";
}

std::string format_2d(const Tensor& tensor, size_t each_side_count) {
    std::stringstream ss;
    size_t rows = tensor.shape[0];
    Tensor row_tensor({tensor.shape[1]}, {});

    if (rows <= 2 * each_side_count) {
        for (size_t row = 0; row < rows; ++row) {
            row_tensor.data = std::vector<float>(tensor.data.begin() + row * tensor.strides[0], tensor.data.begin() + (row + 1) * tensor.strides[0]);
            ss << format_1d(row_tensor, each_side_count) << ", ";
        }
    } else {
        // Handle the first few rows
        for (size_t row = 0; row < each_side_count; ++row) {
            row_tensor.data = std::vector<float>(tensor.data.begin() + row * tensor.strides[0], tensor.data.begin() + (row + 1) * tensor.strides[0]);
            ss << format_1d(row_tensor, each_side_count) << ", ";
        }
        ss << "..., ";
        // Handle the last few rows
        for (size_t row = rows - each_side_count; row < rows; ++row) {
            row_tensor.data = std::vector<float>(tensor.data.begin() + row * tensor.strides[0], tensor.data.begin() + (row + 1) * tensor.strides[0]);
            ss << format_1d(row_tensor, each_side_count) << ", ";
        }
    }

    return "[" + ss.str() + "]";
}

std::string format_3d(const Tensor& tensor, size_t each_side_count) {
    std::stringstream ss;
    size_t batches = tensor.shape[0];
    Tensor batch_tensor({tensor.shape[1], tensor.shape[2]}, {});

    if (batches <= 2 * each_side_count) {
        for (size_t batch = 0; batch < batches; ++batch) {
            batch_tensor.data =
                std::vector<float>(tensor.data.begin() + batch * tensor.strides[0], tensor.data.begin() + (batch + 1) * tensor.strides[0]);
            ss << format_2d(batch_tensor, each_side_count) << ", ";
        }
    } else {
        // Handle the first few batches
        for (size_t batch = 0; batch < each_side_count; ++batch) {
            batch_tensor.data =
                std::vector<float>(tensor.data.begin() + batch * tensor.strides[0], tensor.data.begin() + (batch + 1) * tensor.strides[0]);
            ss << format_2d(batch_tensor, each_side_count) << ", ";
        }
        ss << "..., ";
        // Handle the last few batches
        for (size_t batch = batches - each_side_count; batch < batches; ++batch) {
            batch_tensor.data =
                std::vector<float>(tensor.data.begin() + batch * tensor.strides[0], tensor.data.begin() + (batch + 1) * tensor.strides[0]);
            ss << format_2d(batch_tensor, each_side_count) << ", ";
        }
    }

    return "[" + ss.str() + "]";
}

std::string format_4d(const Tensor& tensor, size_t each_side_count) {
    std::stringstream ss;
    size_t batches = tensor.shape[0];
    Tensor batch_tensor({tensor.shape[1], tensor.shape[2], tensor.shape[3]}, {});

    if (batches <= 2 * each_side_count) {
        for (size_t batch = 0; batch < batches; ++batch) {
            batch_tensor.data =
                std::vector<float>(tensor.data.begin() + batch * tensor.strides[0], tensor.data.begin() + (batch + 1) * tensor.strides[0]);
            ss << format_3d(batch_tensor, each_side_count) << ", ";
        }
    } else {
        // Handle the first few batches
        for (size_t batch = 0; batch < each_side_count; ++batch) {
            batch_tensor.data =
                std::vector<float>(tensor.data.begin() + batch * tensor.strides[0], tensor.data.begin() + (batch + 1) * tensor.strides[0]);
            ss << format_3d(batch_tensor, each_side_count) << ", ";
        }
        ss << "..., ";
        // Handle the last few batches
        for (size_t batch = batches - each_side_count; batch < batches; ++batch) {
            batch_tensor.data =
                std::vector<float>(tensor.data.begin() + batch * tensor.strides[0], tensor.data.begin() + (batch + 1) * tensor.strides[0]);
            ss << format_3d(batch_tensor, each_side_count) << ", ";
        }
    }

    return "[" + ss.str() + "]";
}

std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    std::string tensor_repr;
    switch (tensor.shape.size()) {
        case 1:
            tensor_repr = format_1d(tensor, 2);
            break;
        case 2:
            tensor_repr = format_2d(tensor, 2);
            break;
        case 3:
            tensor_repr = format_3d(tensor, 2);
            break;
        case 4:
            tensor_repr = format_4d(tensor, 2);
            break;
        default:
            // print the tensor shape
            std::cout << "Tensor shape: [";
            for (const auto& s : tensor.shape) std::cout << s << ", ";
            std::cout << "]\n";

            throw std::runtime_error("Unsupported tensor shape for printing");
    }

    os << "Tensor { data: " << tensor_repr << ", shape: [";
    for (const auto& s : tensor.shape) os << s << ", ";
    os << "], strides: [";
    for (const auto& s : tensor.strides) os << s << ", ";
    os << "] }";
    return os;
}