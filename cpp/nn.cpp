#include "nn.h"

#include <limits>

#include "einops.h"
#include "tensor.h"
#include "utils.h"
#include <cmath>

Linear::Linear(const Tensor& weight, const Tensor& bias) : weight(weight), bias(bias) {}
Tensor Linear::operator()(const Tensor& x) const {
    switch (x.shape.size()) {
        case 2:
            return x.matmul(weight.transpose()).add(bias);
        case 3:
            return einsum("b i j, j k -> b i k", x, weight).add(bias.reshape({1, 1, bias.shape[0]}));
        default:
            throw std::runtime_error("Unsupported tensor shape for linear forward");
    }
}

Embedding::Embedding(const Tensor& weight) : weight(weight) {}
Tensor Embedding::operator()(const Tensor& x) const {
    size_t b = x.shape[0];
    size_t t = x.shape[1];
    size_t d = weight.shape[1];
    Tensor output({b, t, d}, std::vector<float>(b * t * d, 0.0));
    for (size_t i = 0; i < b; i++) {
        for (size_t j = 0; j < t; j++) {
            size_t index = x[{i, j}];  // TODO: check this
            size_t output_index = i * output.strides[0] + j * output.strides[1];
            for (size_t k = 0; k < d; k++) {
                output.data[output_index] = weight.data[index * weight.strides[0] + k * weight.strides[1]];
                output_index += output.strides[2];
            }
        }
    }
    return output;
}

MLP::MLP(const Linear& fc1, const Linear& fc2) : fc1(fc1), fc2(fc2) {}
Tensor MLP::operator()(const Tensor& x) const { 
    Tensor _y = fc1(x);
    // std::cout << "After fc1: " << _y << std::endl;
    Tensor y = gelu(_y);
    // std::cout << "After gelu: " << y << std::endl;
    Tensor output = fc2(y);
    return output;
 }

CausalSelfAttention::CausalSelfAttention(const Linear& fc_qkv, const Linear& fc_o) : fc_qkv(fc_qkv), fc_o(fc_o) {}
Tensor CausalSelfAttention::operator()(const Tensor& x) const {
    Tensor qkv = fc_qkv(x);

    auto qkv_split = qkv.split3(2);
    Tensor q = std::get<0>(qkv_split);
    Tensor k = std::get<1>(qkv_split);
    Tensor v = std::get<2>(qkv_split);

    size_t B = q.shape[0];
    size_t T = q.shape[1];
    size_t D = q.shape[2];

    q = q.reshape({B, T, 12, 64}).permute_multihead();
    k = k.reshape({B, T, 12, 64}).permute_multihead();
    v = v.reshape({B, T, 12, 64}).permute_multihead();

    Tensor logits = einsum("b h i d, b h j d -> b h i j", q, k).div_scalar(std::sqrt(64.0));
    // causal mask
    for (size_t i = 0; i < logits.shape[0]; i++) {
        for (size_t j = 0; j < logits.shape[1]; j++) {
            for (size_t k = 0; k < logits.shape[2]; k++) {
                for (size_t l = 0; l < logits.shape[3]; l++) {
                    if (l > k) {
                        size_t index = i * logits.strides[0] + j * logits.strides[1] + k * logits.strides[2] + l * logits.strides[3];
                        logits.data[index] = -std::numeric_limits<float>::infinity();
                    }
                }
            }
        }
    }

    Tensor attn = softmax(logits, 3);
    // std::cout << "attn: " << attn << std::endl;
    Tensor y = einsum("b h i j, b h j d -> b h i d", attn, v);
    // std::cout << "y: " << y << std::endl;
    Tensor y1 = y.permute_multihead().reshape({B, T, 768});
    // std::cout << "y1: " << y1 << std::endl;
    Tensor output = einsum("b i j, j k -> b i k", y1, fc_o.weight).add(fc_o.bias.reshape({1, 1, fc_o.bias.shape[0]}));
    // std::cout << "output: " << output << std::endl;
    // exit(0);
    return output;
}

LayerNorm::LayerNorm(const Tensor& weight, const Tensor& bias) : weight(weight), bias(bias) {}
Tensor LayerNorm::operator()(const Tensor& x) const {
    Tensor mean = x.mean(2, true);
    Tensor variance = x.variance(2, true);
    Tensor output = x.sub(mean).div(variance.add_scalar(1e-5).sqrt());
    output = output.mul(weight.reshape({1, 1, weight.shape[0]})).add(bias.reshape({1, 1, bias.shape[0]}));
    return output;
}

Block::Block(const LayerNorm& ln1, const CausalSelfAttention& csa, const LayerNorm& ln2, const MLP& mlp) : ln1(ln1), csa(csa), ln2(ln2), mlp(mlp) {}
Tensor Block::operator()(const Tensor& x) const {
    Tensor _y = ln1(x);
    // std::cout << "After ln1: " << _y << std::endl;
    Tensor _y1 = csa(_y);
    // std::cout << "After csa: " << _y1 << std::endl;
    Tensor y = x.add(csa(ln1(x)));
    Tensor _y2 = ln2(y);
    // std::cout << "After ln2: " << _y2 << std::endl;
    Tensor _y3 = mlp(_y2);
    // std::cout << "After mlp: " << _y3 << std::endl;
    Tensor output = y.add(mlp(ln2(y)));
    return output;
}

GPT2::GPT2(const Embedding& wte, const Embedding& wpe, const std::vector<Block>& blocks, const LayerNorm& ln_f, const Linear& lm_head)
    : wte(wte), wpe(wpe), blocks(blocks), ln_f(ln_f), lm_head(lm_head) {}

GPT2 GPT2::from_json(const std::string& file_path) {
    auto weights_map = loadModelWeights(file_path);
    Embedding wte = Embedding(weights_map["wte.weight"]);
    Embedding wpe = Embedding(weights_map["wpe.weight"]);
    std::vector<Block> blocks;
    for (size_t i = 0; i < 12; i++) {
        std::string prefix = "h." + std::to_string(i) + ".";
        LayerNorm ln1 = LayerNorm(weights_map[prefix + "ln_1.weight"], weights_map[prefix + "ln_1.bias"]);
        Linear attn_fc_qkv = Linear(weights_map[prefix + "attn.c_attn.weight"], weights_map[prefix + "attn.c_attn.bias"]);
        Linear attn_fc_o = Linear(weights_map[prefix + "attn.c_proj.weight"], weights_map[prefix + "attn.c_proj.bias"]);
        CausalSelfAttention csa = CausalSelfAttention(attn_fc_qkv, attn_fc_o);
        LayerNorm ln2 = LayerNorm(weights_map[prefix + "ln_2.weight"], weights_map[prefix + "ln_2.bias"]);
        Linear mlp_fc_h = Linear(weights_map[prefix + "mlp.c_fc.weight"], weights_map[prefix + "mlp.c_fc.bias"]);
        Linear mlp_fc_o = Linear(weights_map[prefix + "mlp.c_proj.weight"], weights_map[prefix + "mlp.c_proj.bias"]);
        MLP mlp = MLP(mlp_fc_h, mlp_fc_o);
        Block block = Block(ln1, csa, ln2, mlp);
        blocks.push_back(block);
    }
    LayerNorm ln_f = LayerNorm(weights_map["ln_f.weight"], weights_map["ln_f.bias"]);

    // Transposing wte_weight for lm_head
    // Note: You need to implement the transpose functionality as per your Tensor class
    Tensor lm_head_weight = weights_map["wte.weight"].transpose();

    // Initializing zero bias for lm_head
    std::vector<float> lm_head_bias_values(weights_map["wte.weight"].shape[0], 0.0f);
    Tensor lm_head_bias = Tensor({weights_map["wte.weight"].shape[0]}, lm_head_bias_values);

    Linear lm_head = Linear(lm_head_weight, lm_head_bias);

    return GPT2(wte, wpe, blocks, ln_f, lm_head);
}

Tensor GPT2::operator()(const std::vector<size_t>& input_tokens) const {
    auto tokens = std::vector<float>(input_tokens.begin(), input_tokens.end());
    Tensor x = Tensor({1, tokens.size()}, tokens);
    Tensor x_id = Tensor::arange(0, input_tokens.size(), 1).reshape({1, input_tokens.size()});
    Tensor _y = wte(x);
    // std::cout << "After wte: " << _y << std::endl;
    Tensor _y1 = wpe(x_id);
    // std::cout << "After wpe: " << _y1 << std::endl;
    Tensor y = wte(x).add(wpe(x_id));
    // std::cout << "After add: " << y << std::endl;
    for (size_t i = 0; i < blocks.size(); i++) {
        y = blocks[i](y);
        // std::cout << "After block " << i << ": " << y << std::endl;
        // exit(0);
    }
    Tensor y1 = ln_f(y);
    // std::cout << "After ln_f: " << y1 << std::endl;
    Tensor y_slice = last_token_slice(y1);
    Tensor output = lm_head(y_slice);
    return output;
}

std::vector<size_t> GPT2::generate(const std::vector<size_t>& input_tokens, size_t max_tokens) {
    auto tokens = std::vector<size_t>(input_tokens.begin(), input_tokens.end());
    for (size_t i = 0; i < max_tokens; i++) {
        Tensor output = (*this)(tokens);
        Tensor probs = output.exp();
        Tensor probs_slice = last_token_slice(probs);
        float next_token = static_cast<float>(argmax(probs_slice));
        tokens.push_back(next_token);
    }
    return tokens;
}