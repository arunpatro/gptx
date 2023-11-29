use crate::tensor::Tensor;

pub fn relu(x: &Tensor) -> Tensor {
    let mut output = x.clone();
    for i in 0..x.data.len() {
        output.data[i] = if x.data[i] > 0.0 { x.data[i] } else { 0.0 };
    }
    output
}

pub fn gelu(x: &Tensor) -> Tensor {
    let mut output = x.clone();
    for i in 0..x.data.len() {
        output.data[i] = 0.5
            * x.data[i]
            * (1.0
                + ((2.0 / std::f32::consts::PI).sqrt()
                    * (x.data[i] + 0.044715 * x.data[i].powi(3)))
                .tanh());
    }
    output
}

pub fn softmax(x: &Tensor, dim: usize) -> Tensor {
    assert!(dim == 3);
    assert!(x.shape.len() == 4);
    let mut output = x.clone();
    let x_max = x.max4d(dim, true);
    let x_exp = x.sub(&x_max).exp();
    let x_exp_sum = x_exp.sum4d(dim, true);
    for i in 0..x.shape[0] {
        for j in 0..x.shape[1] {
            for k in 0..x.shape[2] {
                for l in 0..x.shape[3] {

                    output[vec![i, j, k, l]] =
                        x_exp[vec![i, j, k, l]] / x_exp_sum[vec![i, j, k, 0]];
                }
            }
        }
    }
    output
}

pub fn last_token_slice(x: &Tensor) -> Tensor {
    assert_eq!(x.shape.len(), 3); // Ensure it's a 3D tensor
    let new_shape = vec![x.shape[0], 1, x.shape[2]];
    let mut new_data = Tensor::new(new_shape.clone(), vec![0.0; new_shape.iter().product()]);

    for i in 0..x.shape[0] {
        for k in 0..x.shape[2] {
            let index = i * x.strides[0] + (x.shape[1] - 1) * x.strides[1] + k * x.strides[2];
            new_data[vec![i, 0, k]] = x.data[index];
        }
    }

    Tensor::new(new_shape, new_data.data)
}

pub fn argmax(logits: &Tensor) -> usize {
    let mut max = logits.data[0];
    let mut max_index = 0;
    for i in 1..logits.data.len() {
        if logits.data[i] > max {
            max = logits.data[i];
            max_index = i;
        }
    }
    max_index
}

pub fn assert_approx_eq(a: &[f32], b: &[f32], epsilon: f32) {
    assert_eq!(a.len(), b.len(), "Vectors are of different lengths");
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() < epsilon, "Values {:?} and {:?} are not approximately equal", x, y);
    }
}
