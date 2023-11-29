# gptx
A minimal implementation of GPT-2 in Rust and C++.

## About
This is a barebones implemenation of GPT-2 in Rust and C++. We also implement a minimal Tensor Library for the same. Currently, this only supports inference on the GPT-2 model with float32 weights on CPU. 

## Install

0. If you're at NYU CS, you can first setup the enviroment on Crunchy1.
```bash
module load cmake-3
module load gcc-9.2
module load python-3.8
```

1. First download the weights using the code in python folder.
```bash
pip install -r requirements.txt
python get_gpt_weights.py
```

2. For the rust implementation, use the -t for no. of threads, -n for no. of tokens to generate, and -m for the model path in json format.
```bash
cd rust;
cargo build --release;
./target/release/gptx -t 8 -n 20 -m ../python/model_weights.json
```

3. For the CPP implementation, the same commands apply.
```bash
cd cpp
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
./gptxx -t 4 -n 20 -m ../../python/model_weights.json
```

## Benchmarking
After building the binaries for both the projects, run the scripts for the threads like in the `scripts` folder. 
```bash
cd rust
sh ../scripts/rust.sh
```
```bash
cd cpp
sh ../scripts/cpp.sh
```

Plot the results using the script in `python` folder. 
```bash
cd python
python plot_benchmarks.py
```

## Results
We experiment with parallelizing the tensor multiplications using multi-threading, using the rayon crate for rust and OpenMP for C++. 

![tokens-per-second](./python/tokens-per-second_rust_cpp.png)
![seconds-per-token](./python/seconds-per-token_rust_cpp.png)
![speedup](./python/speedup_rust_cpp.png)