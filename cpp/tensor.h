#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <string>
#include <iostream>

class Tensor {
public:
    std::vector<float> data;
    std::vector<size_t> shape;
    std::vector<size_t> strides;

    // Constructor, Destructor, and other member functions
    Tensor(const std::vector<size_t>& shape, const std::vector<float>& data);
    Tensor(const std::vector<size_t>& shape, const std::vector<float>& data, const std::vector<size_t>& strides);

    // Stuff for loading the json weights
    Tensor(); // Default Constructor
    Tensor(const Tensor& other);  // Copy Constructor
    Tensor(Tensor&& other) noexcept; // Move Constructor
    Tensor& operator=(Tensor other); // Assignment Operator
    ~Tensor(); // Destructor


    // Utils
    const size_t calculate_flat_index(const std::vector<size_t>& index) const;
    const float& operator[](const std::vector<size_t>& index) const;
    float& operator[](const std::vector<size_t>& index);

    // stuff
    Tensor clone() const;
    static Tensor arange(size_t start, size_t end, size_t step); // static because it doesn't depend on the object
    Tensor reshape(const std::vector<size_t>& new_shape) const;
    Tensor transpose() const;
    Tensor permute_multihead() const;
    std::tuple<Tensor, Tensor, Tensor> split3(size_t dim) const;

    // reductions
    Tensor max4d(size_t dim, bool keepdim) const;
    Tensor sum4d(size_t dim, bool keepdim) const;
    Tensor mean(size_t dim, bool keepdim) const;
    Tensor variance(size_t dim, bool keepdim) const;

    // unary ops
    Tensor exp() const;
    Tensor sqrt() const;
    Tensor add_scalar(float scalar) const;
    Tensor div_scalar(float scalar) const;

    // elementwise ops
    Tensor add(const Tensor& other) const;
    Tensor sub(const Tensor& other) const;
    Tensor mul(const Tensor& other) const;
    Tensor div(const Tensor& other) const;

    // matmuls
    Tensor matmul(const Tensor& other) const;

    // formating
    std::string format_1d(size_t each_side_count) const;
    std::string format_2d(size_t each_side_count) const;
    std::string format_3d(size_t each_side_count) const;
    std::string format_4d(size_t each_side_count) const;

    // Overloaded operator<< declaration
    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
    friend Tensor operator+(const Tensor& lhs, const Tensor& rhs);
    friend Tensor operator-(const Tensor& lhs, const Tensor& rhs);
};

#endif // TENSOR_H
