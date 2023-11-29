#ifndef EINOPS_H
#define EINOPS_H

#include "tensor.h"

Tensor einsum(const std::string& expr, const Tensor& a, const Tensor& b);

#endif  // EINOPS_H
