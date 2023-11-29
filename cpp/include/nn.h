#ifndef NN_H
#define NN_H

#include "ops.h"
#include "tensor.h"

class Linear {
   public:
    Tensor weight;
    Tensor bias;

    Linear(const Tensor& weight, const Tensor& bias);
    Tensor operator()(const Tensor& x) const;
};

class Embedding {
   public:
    Tensor weight;

    Embedding(const Tensor& weight);
    Tensor operator()(const Tensor& x) const;
};

class LayerNorm {
   public:
    Tensor weight;
    Tensor bias;

    LayerNorm(const Tensor& weight, const Tensor& bias);
    Tensor operator()(const Tensor& x) const;
};

class MLP {
   public:
    Linear fc1;
    Linear fc2;

    MLP(const Linear& fc1, const Linear& fc2);
    Tensor operator()(const Tensor& x) const;
};

class CausalSelfAttention {
   public:
    Linear fc_qkv;
    Linear fc_o;

    CausalSelfAttention(const Linear& fc_qkv, const Linear& fc_o);
    Tensor operator()(const Tensor& x) const;
};

class Block {
   public:
    LayerNorm ln1;
    CausalSelfAttention csa;
    LayerNorm ln2;
    MLP mlp;

    Block(const LayerNorm& ln1, const CausalSelfAttention& csa, const LayerNorm& ln2, const MLP& mlp);
    Tensor operator()(const Tensor& x) const;
};

class GPT2 {
   public:
    Embedding wte;
    Embedding wpe;
    std::vector<Block> blocks;
    LayerNorm ln_f;
    Linear lm_head;

    GPT2(const Embedding& wte, const Embedding& wpe, const std::vector<Block>& blocks, const LayerNorm& ln_f, const Linear& lm_head);

    // Static factory method to create a GPT2 instance from JSON
    static GPT2 from_json(const std::string& file_path);
    
    Tensor operator()(const std::vector<size_t>& input_tokens) const;
    std::vector<size_t> generate(const std::vector<size_t>& input_tokens, size_t max_tokens);
};

#endif  // NN_H
