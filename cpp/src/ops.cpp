#include "ops.h"

#include <cassert>
#include <cmath>
#include <vector>

#include "tensor.h"

Tensor relu(const Tensor& x) {
    Tensor output = x.clone();
    for (int i = 0; i < x.data.size(); i++) {
        output.data[i] = x.data[i] > 0.0 ? x.data[i] : 0.0;
    }
    return output;
}

Tensor gelu(const Tensor& x) {
    Tensor output = x.clone();
    for (size_t i = 0; i < x.data.size(); i++) {
        output.data[i] = 0.5 * x.data[i] * (1.0 + std::tanh((2.0 / std::sqrt(M_PI)) * (x.data[i] + 0.044715 * std::pow(x.data[i], 3))));
    }
    return output;
}

Tensor softmax(const Tensor& x, size_t dim) {
    // only implemented for 4D tensors and softmax along the last dimension
    assert(dim == 3);
    assert(x.shape.size() == 4);

    Tensor output = x.clone();
    Tensor x_max = output.max4d(dim, true);
    Tensor x_exp = (output - x_max).exp();
    Tensor x_exp_sum = x_exp.sum4d(dim, true);

    for (size_t i = 0; i < x.shape[0]; ++i) {
        for (size_t j = 0; j < x.shape[1]; ++j) {
            for (size_t k = 0; k < x.shape[2]; ++k) {
                for (size_t l = 0; l < x.shape[3]; ++l) {
                    size_t index = i * x.strides[0] + j * x.strides[1] + k * x.strides[2] + l * x.strides[3];
                    size_t sum_index = i * x_exp_sum.strides[0] + j * x_exp_sum.strides[1] + k * x_exp_sum.strides[2];
                    output.data[index] = x_exp.data[index] / x_exp_sum.data[sum_index];
                }
            }
        }
    }

    return output;
}

Tensor last_token_slice(const Tensor& x) {
    assert(x.shape.size() == 3);  // Ensure it's a 3D tensor
    std::vector<size_t> new_shape = {x.shape[0], 1, x.shape[2]};
    Tensor new_data(new_shape, std::vector<float>(new_shape[0] * new_shape[1] * new_shape[2], 0.0));

    for (size_t i = 0; i < x.shape[0]; ++i) {
        for (size_t k = 0; k < x.shape[2]; ++k) {
            size_t index = i * x.strides[0] + (x.shape[1] - 1) * x.strides[1] + k * x.strides[2];
            new_data[{i, 0, k}] = x.data[index];
        }
    }

    return new_data;
}

size_t argmax(const Tensor& logits) {
    float max = logits.data[0];
    size_t max_index = 0;

    for (size_t i = 1; i < logits.data.size(); ++i) {
        if (logits.data[i] > max) {
            max = logits.data[i];
            max_index = i;
        }
    }

    return max_index;
}
