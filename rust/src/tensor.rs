use serde::{Deserialize, Serialize};
use std::ops::{Index, IndexMut};

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

impl std::ops::Add for Tensor {
    type Output = Self;

    fn add(self, other: Tensor) -> Self::Output {
        assert!(
            self.shape == other.shape,
            "Shapes of tensors must match for addition"
        );
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        let output = Tensor {
            data,
            shape: self.shape,
            strides: self.strides,
        };
        output
    }
}

impl std::ops::Sub for Tensor {
    type Output = Self;

    fn sub(self, other: Tensor) -> Self::Output {
        assert!(
            self.shape == other.shape,
            "Shapes of tensors must match for Subtraction"
        );
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a - b)
            .collect();
        let output = Tensor {
            data,
            shape: self.shape,
            strides: self.strides,
        };
        output
    }
}

impl Tensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Tensor {
        let mut strides = Vec::with_capacity(shape.len());
        let mut stride = 1;
        for &dimension in shape.iter().rev() {
            strides.insert(0, stride);
            stride *= dimension;
        }

        Tensor {
            data,
            shape,
            strides,
        }
    }

    pub fn arange(start: usize, end: usize, step: usize) -> Tensor {
        let data: Vec<f32> = (start..end).step_by(step).map(|x| x as f32).collect();
        Tensor {
            shape: vec![data.len()],
            strides: vec![1],
            data,
        }
    }

    pub fn reshape(&self, shape: Vec<usize>) -> Tensor {
        assert_eq!(self.data.len(), shape.iter().product::<usize>());
        let mut strides = Vec::with_capacity(shape.len());
        let mut stride = 1;
        for &dimension in shape.iter().rev() {
            strides.insert(0, stride);
            stride *= dimension;
        }
        Tensor {
            data: self.data.clone(),
            shape,
            strides,
        }
    }

    pub fn transpose(&self) -> Tensor {
        // this is only for 2d tensors
        assert_eq!(self.shape.len(), 2);
        let mut output = Tensor::new(
            vec![self.shape[1], self.shape[0]],
            vec![0.; self.shape[0] * self.shape[1]],
        );
        for i in 0..self.shape[0] {
            for j in 0..self.shape[1] {
                output[vec![j, i]] = self[vec![i, j]];
            }
        }
        output
    }

    pub fn permute_multihead(&self) -> Tensor {
        // hardcoding for now
        // for a 4d tensor, permute the dims 1 and 2
        assert!(self.shape.len() == 4);
        let mut output = Tensor::new(
            vec![self.shape[0], self.shape[2], self.shape[1], self.shape[3]],
            vec![0.; self.shape[0] * self.shape[1] * self.shape[2] * self.shape[3]],
        );

        for i in 0..self.shape[0] {
            for j in 0..self.shape[1] {
                for k in 0..self.shape[2] {
                    for l in 0..self.shape[3] {
                        output[vec![i, k, j, l]] = self[vec![i, j, k, l]];
                    }
                }
            }
        }
        output
    }

    pub fn split3(&self, dim: usize) -> (Tensor, Tensor, Tensor) {
        assert!(dim < self.shape.len());
        let (bsz, seq, hidden) = (self.shape[0], self.shape[1], self.shape[2]);
        let mut output1 = Tensor::new(vec![bsz, seq, hidden / 3], vec![0.; bsz * seq * hidden / 3]);
        let mut output2 = Tensor::new(vec![bsz, seq, hidden / 3], vec![0.; bsz * seq * hidden / 3]);
        let mut output3 = Tensor::new(vec![bsz, seq, hidden / 3], vec![0.; bsz * seq * hidden / 3]);
        for i in 0..bsz {
            for j in 0..seq {
                for k in 0..hidden {
                    if k < hidden / 3 {
                        output1[vec![i, j, k]] = self[vec![i, j, k]];
                    } else if k < 2 * hidden / 3 {
                        output2[vec![i, j, k - hidden / 3]] = self[vec![i, j, k]];
                    } else {
                        output3[vec![i, j, k - 2 * hidden / 3]] = self[vec![i, j, k]];
                    }
                }
            }
        }
        (output1, output2, output3)
    }

    // reduction functions
    pub fn max4d(&self, dim: usize, keepdim: bool) -> Tensor {
        assert!(dim < self.shape.len() && self.shape.len() == 4);
        let (B, H, T, D) = (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);

        let mut output_shape = self.shape.clone();
        output_shape[dim] = 1;

        let mut output = Tensor::new(
            output_shape.clone(),
            vec![0.; output_shape.iter().product()],
        );

        for i in 0..B {
            for j in 0..H {
                for k in 0..T {
                    for l in 0..D {
                        let index = i * self.strides[0]
                            + j * self.strides[1]
                            + k * self.strides[2]
                            + l * self.strides[3];
                        match dim {
                            0 => {
                                output[vec![0, j, k, l]] =
                                    output[vec![0, j, k, l]].max(self.data[index])
                            }
                            1 => {
                                output[vec![i, 0, k, l]] =
                                    output[vec![i, 0, k, l]].max(self.data[index])
                            }
                            2 => {
                                output[vec![i, j, 0, l]] =
                                    output[vec![i, j, 0, l]].max(self.data[index])
                            }
                            3 => {
                                output[vec![i, j, k, 0]] =
                                    output[vec![i, j, k, 0]].max(self.data[index])
                            }
                            _ => panic!("Invalid dimension"),
                        }
                    }
                }
            }
        }

        if !keepdim {
            output_shape.remove(dim);
            output.reshape(output_shape)
        } else {
            output
        }
    }

    pub fn sum4d(&self, dim: usize, keepdim: bool) -> Tensor {
        assert!(dim < self.shape.len() && self.shape.len() == 4);
        let (B, H, T, D) = (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);

        let mut output_shape = self.shape.clone();
        output_shape[dim] = 1;

        let mut output = Tensor::new(
            output_shape.clone(),
            vec![0.; output_shape.iter().product()],
        );

        for i in 0..B {
            for j in 0..H {
                for k in 0..T {
                    for l in 0..D {
                        let index = i * self.strides[0]
                            + j * self.strides[1]
                            + k * self.strides[2]
                            + l * self.strides[3];
                        match dim {
                            0 => output[vec![0, j, k, l]] += self.data[index],
                            1 => output[vec![i, 0, k, l]] += self.data[index],
                            2 => output[vec![i, j, 0, l]] += self.data[index],
                            3 => output[vec![i, j, k, 0]] += self.data[index],
                            _ => panic!("Invalid dimension"),
                        }
                    }
                }
            }
        }

        if !keepdim {
            output_shape.remove(dim);
            output.reshape(output_shape)
        } else {
            output
        }
    }

    pub fn mean(&self, dim: usize, keepdim: bool) -> Tensor {
        assert!(dim < self.shape.len() && self.shape.len() == 3);
        let (B, T, D) = (self.shape[0], self.shape[1], self.shape[2]);

        let mut output_shape = self.shape.clone();
        output_shape[dim] = 1;

        let mut output = Tensor::new(
            output_shape.clone(),
            vec![0.; output_shape.iter().product()],
        );

        for i in 0..B {
            for j in 0..T {
                for k in 0..D {
                    let index = i * self.strides[0] + j * self.strides[1] + k * self.strides[2];
                    match dim {
                        0 => output[vec![0, j, k]] += self.data[index],
                        1 => output[vec![i, 0, k]] += self.data[index],
                        2 => output[vec![i, j, 0]] += self.data[index],
                        _ => panic!("Invalid dimension"),
                    }
                }
            }
        }

        let output = output.div_scalar(self.shape[dim] as f32);

        if !keepdim {
            output_shape.remove(dim);
            output.reshape(output_shape)
        } else {
            output
        }
    }

    pub fn variance(&self, dim: usize, keepdim: bool) -> Tensor {
        let mean = self.mean(dim, keepdim);
        let x_minus_mean = self.sub(&mean);
        let x_minus_mean_squared = x_minus_mean.mul(&x_minus_mean);
        let var = x_minus_mean_squared.mean(dim, keepdim);
        var
    }

    // unary ops
    pub fn exp(&self) -> Self {
        let data = self.data.iter().map(|&x| x.exp()).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }

    pub fn sqrt(&self) -> Self {
        let data = self.data.iter().map(|&x| x.sqrt()).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }

    pub fn add_scalar(&self, scalar: f32) -> Tensor {
        let data = self.data.iter().map(|&x| x + scalar).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }

    pub fn div_scalar(&self, scalar: f32) -> Tensor {
        let data = self.data.iter().map(|&x| x / scalar).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }

    // element wise operations
    pub fn add(&self, other: &Tensor) -> Tensor {
        // case match for direct adding or broadcasting (broadcasting requires same number of dimensions, need to add logic for more complex broadcasting)
        let mut output = Tensor::new(self.shape.clone(), vec![0.; self.data.len()]);
        if self.shape == other.shape {
            for i in 0..self.data.len() {
                output.data[i] = self.data[i] + other.data[i];
            }
            output
        } else if self.shape.len() == 2 && other.shape.len() == 2 {
            // this is for 2d broadcasting
            for i in 0..self.shape[0] {
                for j in 0..self.shape[1] {
                    let index_self = i * self.strides[0] + j * self.strides[1];
                    let index_other = (i % other.shape[0]) * other.strides[0]
                        + (j % other.shape[1]) * other.strides[1];
                    output.data[index_self] = self.data[index_self] + other.data[index_other];
                }
            }
            output
        } else if self.shape.len() == 2 && other.shape.len() == 1 {
            // this is for 2d broadcasting
            for i in 0..self.shape[0] {
                for j in 0..self.shape[1] {
                    let index_self = i * self.strides[0] + j * self.strides[1];
                    let index_other = (j % other.shape[0]) * other.strides[0];
                    output.data[index_self] = self.data[index_self] + other.data[index_other];
                }
            }
            output
        } else if self.shape.len() == 3 && other.shape.len() == 3 {
            // this is for 3d broadcasting
            for i in 0..self.shape[0] {
                for j in 0..self.shape[1] {
                    for k in 0..self.shape[2] {
                        let index_self =
                            i * self.strides[0] + j * self.strides[1] + k * self.strides[2];
                        let index_other = (i % other.shape[0]) * other.strides[0]
                            + (j % other.shape[1]) * other.strides[1]
                            + (k % other.shape[2]) * other.strides[2];
                        output.data[index_self] = self.data[index_self] + other.data[index_other];
                    }
                }
            }
            output
        }
        else {
            panic!("Invalid shapes for addition")
        }
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        // case match for direct adding or broadcasting (broadcasting requires same number of dimensions, need to add logic for more complex broadcasting)
        let mut output = Tensor::new(self.shape.clone(), vec![0.; self.data.len()]);
        if self.shape == other.shape {
            for i in 0..self.data.len() {
                output.data[i] = self.data[i] - other.data[i];
            }
            output
        } else if self.shape.len() == 4 && other.shape.len() == 4 {
            // this is for 4d broadcasting
            for i in 0..self.shape[0] {
                for j in 0..self.shape[1] {
                    for k in 0..self.shape[2] {
                        for l in 0..self.shape[3] {
                            let index_self = i * self.strides[0]
                                + j * self.strides[1]
                                + k * self.strides[2]
                                + l * self.strides[3];
                            let index_other = (i % other.shape[0]) * other.strides[0]
                                + (j % other.shape[1]) * other.strides[1]
                                + (k % other.shape[2]) * other.strides[2]
                                + (l % other.shape[3]) * other.strides[3];
                            output.data[index_self] =
                                self.data[index_self] - other.data[index_other];
                        }
                    }
                }
            }
            output
        } else if self.shape.len() == 2 && other.shape.len() == 1 {
            // this is for 2d broadcasting
            for i in 0..self.shape[0] {
                for j in 0..self.shape[1] {
                    let index_self = i * self.strides[0] + j * self.strides[1];
                    let index_other = (j % other.shape[0]) * other.strides[0];
                    output.data[index_self] = self.data[index_self] - other.data[index_other];
                }
            }
            output
        } else if self.shape.len() == 3 && other.shape.len() == 3 {
            // this is for 3d broadcasting
            for i in 0..self.shape[0] {
                for j in 0..self.shape[1] {
                    for k in 0..self.shape[2] {
                        let index_self =
                            i * self.strides[0] + j * self.strides[1] + k * self.strides[2];
                        let index_other = (i % other.shape[0]) * other.strides[0]
                            + (j % other.shape[1]) * other.strides[1]
                            + (k % other.shape[2]) * other.strides[2];
                        output.data[index_self] = self.data[index_self] - other.data[index_other];
                    }
                }
            }
            output
        }
        else {
            panic!("Invalid shapes for subtraction")
        }
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        // case match for direct adding or broadcasting (broadcasting requires same number of dimensions, need to add logic for more complex broadcasting)
        let mut output = Tensor::new(self.shape.clone(), vec![0.; self.data.len()]);
        if self.shape == other.shape {
            for i in 0..self.data.len() {
                output.data[i] = self.data[i] * other.data[i];
            }
            output
        } else {
            for i in 0..self.shape[0] {
                for j in 0..self.shape[1] {
                    for k in 0..self.shape[2] {
                        let index_self =
                            i * self.strides[0] + j * self.strides[1] + k * self.strides[2];
                        let index_other = (i % other.shape[0]) * other.strides[0]
                            + (j % other.shape[1]) * other.strides[1]
                            + (k % other.shape[2]) * other.strides[2];
                        output.data[index_self] = self.data[index_self] * other.data[index_other];
                    }
                }
            }
            output
        }
    }

    pub fn div(&self, other: &Tensor) -> Tensor {
        // case match for direct adding or broadcasting (broadcasting requires same number of dimensions, need to add logic for more complex broadcasting)
        let mut output = Tensor::new(self.shape.clone(), vec![0.; self.data.len()]);
        if self.shape == other.shape {
            for i in 0..self.data.len() {
                output.data[i] = self.data[i] / other.data[i];
            }
            output
        } else {
            for i in 0..self.shape[0] {
                for j in 0..self.shape[1] {
                    for k in 0..self.shape[2] {
                        let index_self =
                            i * self.strides[0] + j * self.strides[1] + k * self.strides[2];
                        let index_other = (i % other.shape[0]) * other.strides[0]
                            + (j % other.shape[1]) * other.strides[1]
                            + (k % other.shape[2]) * other.strides[2];
                        output.data[index_self] = self.data[index_self] / other.data[index_other];
                    }
                }
            }
            output
        }
    }

    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape.len(), 2);
        assert_eq!(other.shape.len(), 2);
        assert_eq!(other.shape[0], self.shape[1]);

        let mut output = Tensor::new(
            vec![self.shape[0], other.shape[1]],
            vec![0.; self.shape[0] * other.shape[1]],
        );
        for i in 0..self.shape[0] {
            for j in 0..other.shape[1] {
                for k in 0..self.shape[1] {
                    output[vec![i, j]] += self[vec![i, k]] * other[vec![k, j]];
                }
            }
        }
        output
    }

    fn calculate_flat_index(&self, index: &[usize]) -> usize {
        index
            .iter()
            .zip(self.strides.iter())
            .map(|(&i, &stride)| i * stride)
            .sum()
    }
}

impl Index<Vec<usize>> for Tensor {
    type Output = f32;

    fn index(&self, index: Vec<usize>) -> &Self::Output {
        let flat_index = self.calculate_flat_index(&index);
        &self.data[flat_index]
    }
}

impl IndexMut<Vec<usize>> for Tensor {
    fn index_mut(&mut self, index: Vec<usize>) -> &mut Self::Output {
        let flat_index = self.calculate_flat_index(&index);
        &mut self.data[flat_index]
    }
}

// TODO: add exponetial notation and add line breaks for pretty printing
// for strided permutation, the 2d formatting breaks!
impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fn format_1d(data: &Tensor, each_side_count: usize) -> String {
            let size = data.data.len();
            let mut elements = vec![];

            if size < 2 * each_side_count + 1 {
                for &value in &data.data {
                    elements.push(format!("{:.4}", value));
                }
            } else {
                elements.extend(
                    data.data[..each_side_count]
                        .iter()
                        .map(|&x| format!("{:.4}", x)),
                );
                elements.push(String::from("..."));
                elements.extend(
                    data.data[size - each_side_count..]
                        .iter()
                        .map(|&x| format!("{:.4}", x)),
                );
            }

            format!("[{}]", elements.join(", "))
        }

        fn format_2d(data: &Tensor, each_side_count: usize) -> String {
            let rows = data.shape[0];
            let mut rows_str = Vec::new();

            if rows <= 2 * each_side_count {
                for row in 0..rows {
                    let row_start = row * data.strides[0];
                    let row_end = row_start + data.strides[0];
                    let row_slice =
                        Tensor::new(vec![data.shape[1]], data.data[row_start..row_end].to_vec());
                    rows_str.push(format_1d(&row_slice, each_side_count));
                }
            } else {
                for row in 0..each_side_count {
                    let row_start = row * data.strides[0];
                    let row_end = row_start + data.strides[0];
                    let row_slice =
                        Tensor::new(vec![data.shape[1]], data.data[row_start..row_end].to_vec());
                    rows_str.push(format_1d(&row_slice, each_side_count));
                }

                rows_str.push(String::from("..."));

                for row in rows - each_side_count..rows {
                    let row_start = row * data.strides[0];
                    let row_end = row_start + data.strides[0];
                    let row_slice =
                        Tensor::new(vec![data.shape[1]], data.data[row_start..row_end].to_vec());
                    rows_str.push(format_1d(&row_slice, each_side_count));
                }
            }

            format!("[{}]", rows_str.join(", "))
        }

        fn format_3d(data: &Tensor, each_side_count: usize) -> String {
            let batches = data.shape[0];
            let mut batches_str = Vec::new();

            if batches <= 2 * each_side_count {
                for batch in 0..batches {
                    let batch_start = batch * data.strides[0];
                    let batch_end = batch_start + data.strides[0];
                    let batch_slice = Tensor::new(
                        vec![data.shape[1], data.shape[2]],
                        data.data[batch_start..batch_end].to_vec(),
                    );
                    batches_str.push(format_2d(&batch_slice, each_side_count));
                }
            } else {
                for batch in 0..each_side_count {
                    let batch_start = batch * data.strides[0];
                    let batch_end = batch_start + data.strides[0];
                    let batch_slice = Tensor::new(
                        vec![data.shape[1], data.shape[2]],
                        data.data[batch_start..batch_end].to_vec(),
                    );
                    batches_str.push(format_2d(&batch_slice, each_side_count));
                }

                batches_str.push(String::from("..."));

                for batch in batches - each_side_count..batches {
                    let batch_start = batch * data.strides[0];
                    let batch_end = batch_start + data.strides[0];
                    let batch_slice = Tensor::new(
                        vec![data.shape[1], data.shape[2]],
                        data.data[batch_start..batch_end].to_vec(),
                    );
                    batches_str.push(format_2d(&batch_slice, each_side_count));
                }
            }

            format!("[{}]", batches_str.join(", "))
        }

        fn format_4d(data: &Tensor, each_side_count: usize) -> String {
            let batches = data.shape[0];
            let mut batches_str = Vec::new();

            // Adjusting for the 4D tensor
            if batches <= 2 * each_side_count {
                for batch in 0..batches {
                    let batch_start = batch * data.strides[0];
                    let batch_end = batch_start + data.strides[0];
                    let batch_slice = Tensor::new(
                        vec![data.shape[1], data.shape[2], data.shape[3]],
                        data.data[batch_start..batch_end].to_vec(),
                    );
                    batches_str.push(format_3d(&batch_slice, each_side_count)); // Using format_3d
                }
            } else {
                for batch in 0..each_side_count {
                    let batch_start = batch * data.strides[0];
                    let batch_end = batch_start + data.strides[0];
                    let batch_slice = Tensor::new(
                        vec![data.shape[1], data.shape[2], data.shape[3]],
                        data.data[batch_start..batch_end].to_vec(),
                    );
                    batches_str.push(format_3d(&batch_slice, each_side_count)); // Using format_3d
                }

                batches_str.push(String::from("..."));

                for batch in batches - each_side_count..batches {
                    let batch_start = batch * data.strides[0];
                    let batch_end = batch_start + data.strides[0];
                    let batch_slice = Tensor::new(
                        vec![data.shape[1], data.shape[2], data.shape[3]],
                        data.data[batch_start..batch_end].to_vec(),
                    );
                    batches_str.push(format_3d(&batch_slice, each_side_count)); // Using format_3d
                }
            }

            format!("[{}]", batches_str.join(", "))
        }

        let mut tensor_repr = String::new();
        match self.shape.len() {
            1 => {
                tensor_repr = format_1d(self, 2);
            }
            2 => {
                tensor_repr = format_2d(self, 2);
            }
            3 => {
                tensor_repr = format_3d(self, 2);
            }
            4 => {
                tensor_repr = format_4d(self, 2);
            }
            _ => return Err(std::fmt::Error), // Handle other cases or throw an error
        }

        write!(
            f,
            "Tensor {{ data: {}, shape: {:?}, strides: {:?} }}",
            tensor_repr, self.shape, self.strides
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_permute() {
        // let a = Tensor::new(
        //     vec![2, 2, 3],
        //     vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
        // );
        // let b = a.permute(vec![1, 2, 0]);
        // assert_eq!(
        //     b,
        //     Tensor::new(
        //         vec![2, 3, 2],
        //         vec![1., 7., 2., 8., 3., 9., 4., 10., 5., 11., 6., 12.]
        //     )
        // );
    }

    #[test]
    fn test_matmul() {
        let a = Tensor::new(vec![2, 3], vec![1., 2., 3., 4., 5., 6.]);
        let b = Tensor::new(vec![3, 2], vec![7., 8., 9., 10., 11., 12.]);
        let c = a.matmul(&b);
        assert_eq!(c, Tensor::new(vec![2, 2], vec![58., 64., 139., 154.]));
    }

    #[test]
    fn test_mean() {
        let a = Tensor::new(
            vec![2, 2, 3],
            vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
        );
        assert_eq!(
            a.mean(0, false).data,
            Tensor::new(vec![2, 3], vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).data
        );
        assert_eq!(
            a.mean(1, true).data,
            Tensor::new(vec![2, 1, 3], vec![2.5, 3.5, 4.5, 8.5, 9.5, 10.5]).data
        );
        assert_eq!(
            a.mean(2, false).data,
            Tensor::new(vec![2, 2], vec![2.0, 5.0, 8.0, 11.0]).data
        );
    }
}
