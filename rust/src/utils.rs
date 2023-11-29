use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;
use std::fs;

// CONSTANTS
const N_EMBD: usize = 768;
const N_LAYER: usize = 12;
const N_HEAD: usize = 4;
const N_CTX: usize = 1024;
const N_VOCAB: usize = 50257;

// Define a struct to hold the parameter data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Weight {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

// Define a type alias for the weights map
pub type WeightsMap = HashMap<String, Tensor>;

pub fn load_model_weights(file_path: &str) -> Result<WeightsMap, Box<dyn Error>> {
    let data = fs::read_to_string(file_path)?;
    let param_map: HashMap<String, Weight> = serde_json::from_str(&data)?;
    let mut weights_map = WeightsMap::new();
    for (name, param) in param_map {
        let array = Tensor::new(param.shape, param.data);
        weights_map.insert(name, array);
    }

    Ok(weights_map)
}
