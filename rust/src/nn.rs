use crate::einops::einsum;
use crate::ops;
use crate::tensor::Tensor;
use crate::utils;

pub struct Linear {
    pub weight: Tensor, // (Out x In)
    pub bias: Tensor,   // (Out)
}

impl Linear {
    pub fn new(weight: Tensor, bias: Tensor) -> Self {
        Self { weight, bias }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        match input.shape.len() {
            3 => {
                let output = einsum("b i j, j k -> b i k", input, &self.weight)
                    .add(&self.bias.reshape(vec![1, 1, self.bias.shape[0]]));
                output
            }
            2 => {
                let output = input.matmul(&self.weight.transpose()).add(&self.bias);
                output
            }
            _ => panic!("Invalid input shape"),
        }
    }
}

pub struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f32,
}

impl LayerNorm {
    pub fn new(weight: Tensor, bias: Tensor) -> Self {
        Self {
            weight,
            bias,
            eps: 1e-5,
        }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        let mean = input.mean(2, true);
        let variance = input.variance(2, true);
        let norm = input.sub(&mean).div(&variance.add_scalar(self.eps).sqrt());
        let output = norm
            .mul(&self.weight.reshape(vec![1, 1, self.weight.shape[0]]))
            .add(&self.bias.reshape(vec![1, 1, self.bias.shape[0]]));
        output
    }
}

pub struct Embedding {
    weight: Tensor,
}
// not an efficience indexing - lots of unnecessary floating point ops
// can replace with einops
impl Embedding {
    pub fn new(weight: Tensor) -> Self {
        Self { weight }
    }
    pub fn forward(&self, input: &Tensor) -> Tensor {
        // input is (batch_size, seq_len)
        let (B, T) = (input.shape[0], input.shape[1]);
        let D = self.weight.shape[1];
        let mut output = Tensor::new(vec![B, T, D], vec![0.; B * T * D]);
        for i in 0..B {
            for j in 0..T {
                let index = input.data[i * input.strides[0] + j * input.strides[1]] as usize;
                let mut output_index = i * output.strides[0] + j * output.strides[1];
                for k in 0..self.weight.shape[1] {
                    output.data[output_index] = self.weight.data
                        [index * self.weight.strides[0] + k * self.weight.strides[1]];
                    output_index += output.strides[2];
                }
            }
        }
        output
    }
}

// has an inbuilt ReLU activation
pub struct MLP {
    fc1: Linear,
    fc2: Linear,
}

impl MLP {
    pub fn new(fc1: Linear, fc2: Linear) -> Self {
        Self { fc1, fc2 }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        let x = self.fc1.forward(input);
        // println!("after fc1: {:?}", x);
        let x = ops::gelu(&x);
        // println!("after gelu: {:?}", x);
        let x = self.fc2.forward(&x);
        x
    }
}

// multi-head attention
pub struct CausalSelfAttention {
    fc_qkv: Linear,
    fc_o: Linear,
}

impl CausalSelfAttention {
    pub fn new(fc_qkv: Linear, fc_o: Linear) -> Self {
        Self { fc_qkv, fc_o }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        let qkv = einsum("b i j, j k -> b i k", input, &self.fc_qkv.weight).add(
            &self
                .fc_qkv
                .bias
                .reshape(vec![1, 1, self.fc_qkv.bias.shape[0]]),
        );

        let (q, k, v) = qkv.split3(2);
        let (B, T, D) = (q.shape[0], q.shape[1], q.shape[2]);

        let q = q.reshape(vec![B, T, 12, 64]).permute_multihead();
        let k = k.reshape(vec![B, T, 12, 64]).permute_multihead();
        let v = v.reshape(vec![B, T, 12, 64]).permute_multihead();

        // println!("q: {:?}", q);
        // println!("k: {:?}", k);
        // println!("v: {:?}", v);
        // println!("============");
        let att = einsum("b h i d, b h j d -> b h i j", &q, &k);
        let mut att = att.div_scalar((k.shape[3] as f32).sqrt());
        // println!("logits: {:?}", att);

        // causal mask
        for i in 0..att.shape[0] {
            for j in 0..att.shape[1] {
                for k in 0..att.shape[2] {
                    for l in 0..att.shape[3] {
                        if l > k {
                            let index = i * att.strides[0]
                                + j * att.strides[1]
                                + k * att.strides[2]
                                + l * att.strides[3];
                            att.data[index] = std::f32::NEG_INFINITY;
                        }
                    }
                }
            }
        }

        att = ops::softmax(&att, 3);
        let y = einsum("b h i j, b h j d -> b h i d", &att, &v);
        // println!("y after einsum: {:?}", y);
        let y = y.permute_multihead().reshape(vec![B, T, 768]);
        // println!("y after permute: {:?}", y);
        // projection layer
        let output = einsum("b i j, j k -> b i k", &y, &self.fc_o.weight)
            .add(&self.fc_o.bias.reshape(vec![1, 1, self.fc_o.bias.shape[0]]));
        // println!("output: {:?}", output);
        output
    }
}

pub struct Block {
    ln1: LayerNorm,
    csa: CausalSelfAttention,
    ln2: LayerNorm,
    mlp: MLP,
}

impl Block {
    pub fn new(ln1: LayerNorm, csa: CausalSelfAttention, ln2: LayerNorm, mlp: MLP) -> Self {
        Self { ln1, csa, ln2, mlp }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        let x = self.ln1.forward(input);
        // println!("after ln1: {:?}", x);
        let x = self.csa.forward(&x);
        // println!("after csa: {:?}", x);
        let x = x.add(input);
        let x1 = self.ln2.forward(&x);
        // println!("after ln2: {:?}", x1);
        let x1 = self.mlp.forward(&x1);
        // println!("after mlp: {:?}", x1);
        let output = x.add(&x1);
        output
    }
}

pub struct GPT2 {
    wte: Embedding,
    wpe: Embedding,
    blocks: Vec<Block>,
    ln_f: LayerNorm,
    lm_head: Linear,
}

impl GPT2 {
    pub fn new(
        wte: Embedding,
        wpe: Embedding,
        blocks: Vec<Block>,
        ln_f: LayerNorm,
        lm_head: Linear,
    ) -> Self {
        Self {
            wte,
            wpe,
            blocks,
            ln_f,
            lm_head,
        }
    }

    pub fn from_json() -> Self {
        let wmap = utils::load_model_weights("model_weights.json").unwrap();
        let wte_weight = wmap.get("wte.weight").unwrap();
        let wte = Embedding::new(wte_weight.clone());
        let wpe_weight = wmap.get("wpe.weight").unwrap();
        let wpe = Embedding::new(wpe_weight.clone());
        let mut blocks = Vec::new();
        for i in 0..12 {
            let ln1_weight = wmap.get(&format!("h.{}.ln_1.weight", i)).unwrap();
            let ln1_bias = wmap.get(&format!("h.{}.ln_1.bias", i)).unwrap();
            let ln1 = LayerNorm::new(ln1_weight.clone(), ln1_bias.clone());
            let attn_fc_qkv_weight = wmap.get(&format!("h.{}.attn.c_attn.weight", i)).unwrap();
            let attn_fc_qkv_bias = wmap.get(&format!("h.{}.attn.c_attn.bias", i)).unwrap();
            let attn_fc_o_weight = wmap.get(&format!("h.{}.attn.c_proj.weight", i)).unwrap();
            let attn_fc_o_bias = wmap.get(&format!("h.{}.attn.c_proj.bias", i)).unwrap();
            let attn_fc_qkv = Linear::new(attn_fc_qkv_weight.clone(), attn_fc_qkv_bias.clone());
            let attn_fc_o = Linear::new(attn_fc_o_weight.clone(), attn_fc_o_bias.clone());
            let csa = CausalSelfAttention::new(attn_fc_qkv, attn_fc_o);
            let ln2_weight = wmap.get(&format!("h.{}.ln_2.weight", i)).unwrap();
            let ln2_bias = wmap.get(&format!("h.{}.ln_2.bias", i)).unwrap();
            let ln2 = LayerNorm::new(ln2_weight.clone(), ln2_bias.clone());
            let mlp_fc_h_weight = wmap.get(&format!("h.{}.mlp.c_fc.weight", i)).unwrap();
            let mlp_fc_h_bias = wmap.get(&format!("h.{}.mlp.c_fc.bias", i)).unwrap();
            let mlp_fc_o_weight = wmap.get(&format!("h.{}.mlp.c_proj.weight", i)).unwrap();
            let mlp_fc_o_bias = wmap.get(&format!("h.{}.mlp.c_proj.bias", i)).unwrap();
            let mlp_fc_h = Linear::new(mlp_fc_h_weight.clone(), mlp_fc_h_bias.clone());
            let mlp_fc_o = Linear::new(mlp_fc_o_weight.clone(), mlp_fc_o_bias.clone());
            let mlp = MLP::new(mlp_fc_h, mlp_fc_o);
            let block = Block::new(ln1, csa, ln2, mlp);
            blocks.push(block);
        }
        let ln_f_weight = wmap.get("ln_f.weight").unwrap();
        let ln_f_bias = wmap.get("ln_f.bias").unwrap();
        let ln_f = LayerNorm::new(ln_f_weight.clone(), ln_f_bias.clone());
        let fc_weight = wte_weight.transpose(); // this is ideally incorrect and instead the weights of the attn layer should be transposed
        let fc_bias = Tensor::new(vec![wte_weight.shape[0]], vec![0.; wte_weight.shape[0]]); // zero bias
        let lm_head = Linear::new(fc_weight.clone(), fc_bias.clone());
        Self::new(wte, wpe, blocks, ln_f, lm_head)
    }

    pub fn forward(&self, input_tokens: &Vec<usize>) -> Tensor {
        let tokens: Vec<f32> = input_tokens.iter().map(|x| *x as f32).collect();
        let input = Tensor::new(vec![1, tokens.len()], tokens);

        let x = self.wte.forward(&input);
        // println!("after wte: {:?}", x);
        let pos = Tensor::arange(0, input.shape[1], 1).reshape(vec![1, input.shape[1]]);
        let pos_emb = self.wpe.forward(&pos);
        // println!("after wpe: {:?}", pos_emb);
        let mut x = x.add(&pos_emb);
        // println!("after add: {:?}", x);
        for block in &self.blocks {
            x = block.forward(&x);
            // println!("after block: {:?}", x);
        }
        let x = self.ln_f.forward(&x);
        // println!("after ln_f: {:?}", x);
        let x_slice = ops::last_token_slice(&x);
        let logits = self.lm_head.forward(&x_slice);
        logits
    }

    pub fn generate(&self, input_tokens: &Vec<usize>, max_tokens: usize) -> Vec<usize> {
        let mut tokens = input_tokens.clone();
        for _ in 0..max_tokens {
            let logits = self.forward(&tokens);
            let output_token = ops::argmax(&logits);
            tokens.push(output_token);
        }
        tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    #[test]
    fn test_linear() {
        let weight = Tensor::new(
            vec![5, 3],
            vec![
                6.0, 8.0, 2.0, 5.0, 8.0, 3.0, 9.0, 5.0, 2.0, 0.0, 4.0, 6.0, 9.0, 3.0, 7.0,
            ],
        );
        let bias = Tensor::new(vec![5], vec![0., 2., 9., 5., 0.]);
        let linear = Linear::new(weight, bias);
        let input = Tensor::new(vec![1, 3], vec![5.0, 1.0, 4.0]);
        let output = linear.forward(&input);
        assert_eq!(output.data, vec![46.0, 47.0, 67.0, 33.0, 76.0]);
    }

    // #[test]
    // fn test_layer_norm() {
    //     let w1 = Tensor::new(vec![7], vec![4., 0., 3., 8., 4., 0., 4.]);
    //     let b1 = Tensor::new(vec![7], vec![1., 2., 5., 5., 7., 6., 9.]);
    //     let ln = LayerNorm::new(w1, b1);
    //     let x = Tensor::new(vec![7], vec![2., 7., 6., 4., 6., 5., 0.]);
    //     let output = ln.forward(&x);
    //     ops::assert_approx_eq(&output.data, &vec![-2.9539,  2.0000,  7.2241,  4.0115,  9.9654,  6.0000,  1.5864], 1e-4);
    // }

    #[test]
    fn test_mlp() {
        let w1 = Tensor::new(
            vec![5, 3],
            vec![
                6.0, 9.0, 0.0, 5.0, 6.0, 3.0, 4.0, 4.0, 1.0, 8.0, 0.0, 3.0, 9.0, 8.0, 4.0,
            ],
        );
        let b1 = Tensor::new(vec![5], vec![5.0, 0.0, 8.0, 1.0, 0.0]);
        let w2 = Tensor::new(
            vec![3, 5],
            vec![
                1.0, 1.0, 9.0, 3.0, 7.0, 5.0, 2.0, 8.0, 7.0, 9.0, 8.0, 3.0, 3.0, 2.0, 9.0,
            ],
        );
        let b2 = Tensor::new(vec![3], vec![3.0, 0.0, 7.0]);
        let mlp = MLP::new(Linear::new(w1, b1), Linear::new(w2, b2));
        let input = Tensor::new(vec![1, 3], vec![0., 4., 3.]);
        let output = mlp.forward(&input);
        assert_eq!(output.data, vec![658., 953., 931.]);
    }
}
