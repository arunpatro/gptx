use crate::tensor::Tensor;
use rayon::prelude::*;

pub fn einsum(expr: &str, t1: &Tensor, t2: &Tensor) -> Tensor {
    match expr {
        "b h i d, b h j d -> b h i j" => {
            // this is to calculate attention scores
            assert!(t1.shape.len() == 4 && t2.shape.len() == 4);
            let (B, H, I, D) = (t1.shape[0], t1.shape[1], t1.shape[2], t1.shape[3]);
            let (_B, _H, J, _D) = (t2.shape[0], t2.shape[1], t2.shape[2], t2.shape[3]);
            assert_eq!(B, _B);
            assert_eq!(H, _H);
            assert_eq!(D, _D);
            // println!("Calc attn");
            // println!("B H I D: {:?} {:?} {:?} {:?}", B, H, I, D);
            // println!("B H J D: {:?} {:?} {:?} {:?} \n ", _B, _H, J, _D);
            let mut output = Tensor::new(vec![B, H, I, J], vec![0.; B * H * I * J]);
            for b in 0..B {
                for h in 0..H {
                    for i in 0..I {
                        for j in 0..J {
                            let mut sum = 0.0;
                            for d in 0..D {
                                let index1 = b * t1.strides[0]
                                    + h * t1.strides[1]
                                    + i * t1.strides[2]
                                    + d * t1.strides[3];
                                let index2 = b * t2.strides[0]
                                    + h * t2.strides[1]
                                    + j * t2.strides[2]
                                    + d * t2.strides[3];
                                sum += t1.data[index1] * t2.data[index2];
                            }
                            let output_index = b * output.strides[0]
                                + h * output.strides[1]
                                + i * output.strides[2]
                                + j * output.strides[3];
                            output.data[output_index] = sum;
                        }
                    }
                }
            }
            // output
            //     .data
            //     .par_chunks_mut(J)
            //     .enumerate()
            //     .for_each(|(b_idx, chunk)| {
            //         let b = b_idx / (H * I); // Compute the batch index
            //         let h = (b_idx / I) % H; // Compute the height index
            //         let i = b_idx % I; // Compute the width index

            //         for (j, out_val) in chunk.iter_mut().enumerate() {
            //             let mut sum = 0.0;
            //             for d in 0..D {
            //                 let index1 = b * t1.strides[0]
            //                     + h * t1.strides[1]
            //                     + i * t1.strides[2]
            //                     + d * t1.strides[3];
            //                 let index2 = b * t2.strides[0]
            //                     + h * t2.strides[1]
            //                     + j * t2.strides[2]
            //                     + d * t2.strides[3];
            //                 sum += t1.data[index1] * t2.data[index2];
            //             }
            //             *out_val = sum;
            //         }
            //     });
            output
        }
        "b h i j, b h j d -> b h i d" => {
            // this is attention x value matrix multiplication
            assert!(t1.shape.len() == 4 && t2.shape.len() == 4);
            let (B, H, I, J) = (t1.shape[0], t1.shape[1], t1.shape[2], t1.shape[3]);
            let (_B, _H, _J, D) = (t2.shape[0], t2.shape[1], t2.shape[2], t2.shape[3]);
            // println!("Calc attn x value");
            // println!("B H I J: {:?} {:?} {:?} {:?}", B, H, I, J);
            // println!("B H J D: {:?} {:?} {:?} {:?} \n ", _B, _H, _J, D);
            assert_eq!(B, _B);
            assert_eq!(H, _H);
            assert_eq!(J, _J);
            let mut output = Tensor::new(vec![B, H, I, D], vec![0.; B * H * I * D]);
            for b in 0..B {
                for h in 0..H {
                    for i in 0..I {
                        for d in 0..D {
                            let mut sum = 0.0;
                            for j in 0..J {
                                let index1 = b * t1.strides[0]
                                    + h * t1.strides[1]
                                    + i * t1.strides[2]
                                    + j * t1.strides[3];
                                let index2 = b * t2.strides[0]
                                    + h * t2.strides[1]
                                    + j * t2.strides[2]
                                    + d * t2.strides[3];
                                sum += t1.data[index1] * t2.data[index2];
                            }
                            let output_index = b * output.strides[0]
                                + h * output.strides[1]
                                + i * output.strides[2]
                                + d * output.strides[3];
                            output.data[output_index] = sum;
                        }
                    }
                }
            }
            // output
            //     .data
            //     .par_chunks_mut(D)
            //     .enumerate()
            //     .for_each(|(b_idx, chunk)| {
            //         let b = b_idx / (H * I); // Compute the batch index
            //         let h = (b_idx / I) % H; // Compute the height index
            //         let i = b_idx % I; // Compute the width index

            //         for (d, out_val) in chunk.iter_mut().enumerate() {
            //             let mut sum = 0.0;
            //             for j in 0..J {
            //                 let index1 = b * t1.strides[0]
            //                     + h * t1.strides[1]
            //                     + i * t1.strides[2]
            //                     + j * t1.strides[3];
            //                 let index2 = b * t2.strides[0]
            //                     + h * t2.strides[1]
            //                     + j * t2.strides[2]
            //                     + d * t2.strides[3];
            //                 sum += t1.data[index1] * t2.data[index2];
            //             }
            //             *out_val = sum;
            //         }
            //     });
            output
        }
        "b i j, j k -> b i k" => {
            // Linear layer implementation
            assert!(t1.shape.len() == 3 && t2.shape.len() == 2);
            let (B, I, J) = (t1.shape[0], t1.shape[1], t1.shape[2]);
            let (_J, K) = (t2.shape[0], t2.shape[1]);
            assert_eq!(J, _J);
            // println!("Calc linear");
            // println!("B I J: {:?} {:?} {:?}", B, I, J);
            // println!("J K: {:?} {:?} \n ", _J, K);
            let mut output = Tensor::new(vec![B, I, K], vec![0.; B * I * K]);
            for b in 0..B {
                for i in 0..I {
                    for k in 0..K {
                        let mut sum = 0.0;
                        for j in 0..J {
                            let index1 = b * t1.strides[0] + i * t1.strides[1] + j * t1.strides[2];
                            let index2 = j * t2.strides[0] + k * t2.strides[1];
                            sum += t1.data[index1] * t2.data[index2];
                        }
                        let output_index =
                            b * output.strides[0] + i * output.strides[1] + k * output.strides[2];
                        output.data[output_index] = sum;
                    }
                }
            }
            // output
            //     .data
            //     .par_chunks_mut(K)
            //     .enumerate()
            //     .for_each(|(b_idx, chunk)| {
            //         let b = b_idx / I; // Compute the batch index
            //         let i = b_idx % I; // Compute the height index

            //         for (k, out_val) in chunk.iter_mut().enumerate() {
            //             let mut sum = 0.0;
            //             for j in 0..J {
            //                 let index1 = b * t1.strides[0] + i * t1.strides[1] + j * t1.strides[2];
            //                 let index2 = j * t2.strides[0] + k * t2.strides[1];
            //                 sum += t1.data[index1] * t2.data[index2];
            //             }
            //             *out_val = sum;
            //         }
            //     });
            output
        }
        _ => panic!("einsum expression not supported"),
    }
}
