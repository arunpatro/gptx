#include "einops.h"

#include <cassert>
#include <stdexcept>
// #include <omp.h>

Tensor einsum(const std::string& expr, const Tensor& t1, const Tensor& t2) {
    if (expr == "b h i d, b h j d -> b h i j") {
        // this is attention x value matrix multiplication
        assert(t1.shape.size() == 4 && t2.shape.size() == 4);
        size_t B = t1.shape[0];
        size_t H = t1.shape[1];
        size_t I = t1.shape[2];
        size_t D = t1.shape[3];

        size_t _B = t2.shape[0];
        size_t _H = t2.shape[1];
        size_t J = t2.shape[2];
        size_t _D = t2.shape[3];

        // Ensuring the dimensions are compatible for matrix multiplication
        assert(B == _B && H == _H && D == _D);

        Tensor output({B, H, I, J}, std::vector<float>(B * H * I * J, 0.0));

        // for (size_t b = 0; b < B; b++) {
        //     for (size_t h = 0; h < H; h++) {
        //         for (size_t i = 0; i < I; i++) {
        //             for (size_t j = 0; j < J; j++) {
        //                 float sum = 0.0;
        //                 for (size_t d = 0; d < D; d++) {
        //                     size_t index1 = b * t1.strides[0] + h * t1.strides[1] + i * t1.strides[2] + d * t1.strides[3];
        //                     size_t index2 = b * t2.strides[0] + h * t2.strides[1] + j * t2.strides[2] + d * t2.strides[3];
        //                     sum += t1.data[index1] * t2.data[index2];
        //                 }
        //                 size_t output_index = b * output.strides[0] + h * output.strides[1] + i * output.strides[2] + j * output.strides[3];
        //                 output.data[output_index] = sum;
        //             }
        //         }
        //     }
        // }
        // #pragma omp parallel for
        for (size_t index = 0; index < B * H * I * J; ++index) {
            size_t b = index / (H * I * J);
            size_t h = (index / (I * J)) % H;
            size_t i = (index / J) % I;
            size_t j = index % J;

            float sum = 0.0;
            for (size_t d = 0; d < D; d++) {
                size_t index1 = b * t1.strides[0] + h * t1.strides[1] + i * t1.strides[2] + d * t1.strides[3];
                size_t index2 = b * t2.strides[0] + h * t2.strides[1] + j * t2.strides[2] + d * t2.strides[3];
                sum += t1.data[index1] * t2.data[index2];
            }
            output.data[index] = sum;
        }

        return output;

    } else if (expr == "b h i j, b h j d -> b h i d") {
        // Attention x value matrix multiplication
        assert(t1.shape.size() == 4 && t2.shape.size() == 4);
        size_t B = t1.shape[0];
        size_t H = t1.shape[1];
        size_t I = t1.shape[2];
        size_t J = t1.shape[3];

        size_t _B = t2.shape[0];
        size_t _H = t2.shape[1];
        size_t _J = t2.shape[2];
        size_t D = t2.shape[3];

        // Ensure that the dimensions are compatible
        assert(B == _B && H == _H && J == _J);

        Tensor output({B, H, I, D}, std::vector<float>(B * H * I * D, 0.0));

        // for (size_t b = 0; b < B; b++) {
        //     for (size_t h = 0; h < H; h++) {
        //         for (size_t i = 0; i < I; i++) {
        //             for (size_t d = 0; d < D; d++) {
        //                 float sum = 0.0;
        //                 for (size_t j = 0; j < J; j++) {
        //                     size_t index1 = b * t1.strides[0] + h * t1.strides[1] + i * t1.strides[2] + j * t1.strides[3];
        //                     size_t index2 = b * t2.strides[0] + h * t2.strides[1] + j * t2.strides[2] + d * t2.strides[3];
        //                     sum += t1.data[index1] * t2.data[index2];
        //                 }
        //                 size_t output_index = b * output.strides[0] + h * output.strides[1] + i * output.strides[2] + d * output.strides[3];
        //                 output.data[output_index] = sum;
        //             }
        //         }
        //     }
        // }
        // #pragma omp parallel for
        for (size_t index = 0; index < B * H * I * D; ++index) {
            size_t b = index / (H * I * D);
            size_t h = (index / (I * D)) % H;
            size_t i = (index / D) % I;
            size_t d = index % D;

            float sum = 0.0;
            for (size_t j = 0; j < J; j++) {
                size_t index1 = b * t1.strides[0] + h * t1.strides[1] + i * t1.strides[2] + j * t1.strides[3];
                size_t index2 = b * t2.strides[0] + h * t2.strides[1] + j * t2.strides[2] + d * t2.strides[3];
                sum += t1.data[index1] * t2.data[index2];
            }
            output.data[index] = sum;
        }

        return output;

    } else if (expr == "b i j, j k -> b i k") {
        // this is linear layer multiplication
        assert(t1.shape.size() == 3 && t2.shape.size() == 2);
        size_t B = t1.shape[0];
        size_t I = t1.shape[1];
        size_t J = t1.shape[2];

        size_t _J = t2.shape[0];
        size_t K = t2.shape[1];
        assert(J == _J);

        Tensor output({B, I, K}, std::vector<float>(B * I * K, 0.0));
        // for (size_t b = 0; b < B; b++) {
        //     for (size_t i = 0; i < I; i++) {
        //         for (size_t k = 0; k < K; k++) {
        //             float sum = 0.0;
        //             for (size_t j = 0; j < J; j++) {
        //                 size_t index1 = b * t1.strides[0] + i * t1.strides[1] + j * t1.strides[2];
        //                 size_t index2 = j * t2.strides[0] + k * t2.strides[1];
        //                 sum += t1.data[index1] * t2.data[index2];
        //             }
        //             size_t output_index = b * output.strides[0] + i * output.strides[1] + k * output.strides[2];
        //             output.data[output_index] = sum;
        //         }
        //     }
        // }
        // #pragma omp parallel for
        for (size_t index = 0; index < B * I * K; ++index) {
            size_t b = index / (I * K);
            size_t i = (index / K) % I;
            size_t k = index % K;

            float sum = 0.0;
            for (size_t j = 0; j < J; j++) {
                size_t index1 = b * t1.strides[0] + i * t1.strides[1] + j * t1.strides[2];
                size_t index2 = j * t2.strides[0] + k * t2.strides[1];
                sum += t1.data[index1] * t2.data[index2];
            }
            output.data[index] = sum;
        }
        return output;

    } else {
        throw std::runtime_error("Unsupported einsum expression");
    }
};