#ifndef OPS_H
#define OPS_H

#include "tensor.h"

Tensor relu(const Tensor& x);
Tensor gelu(const Tensor& x);
Tensor softmax(const Tensor& x, size_t dim);
Tensor last_token_slice(const Tensor& x);
size_t argmax(const Tensor& logits);

#endif  // OPS_H
