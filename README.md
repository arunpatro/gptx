## Install

1. First download the weights using the code in python folder.
```bash
pip install -r requirements.txt
python get_gpt_weights.py
```

2. For the rust implementation, fix the path of the weights in `src/main.rs` and run the following command. `t` is the number of threads and `n` is the number of tokens to generate.
```bash
cd rust;
cargo build --release;
./target/release/gptx -t 4 -n 20
```

3. For the CPP implementation, fix the path of the weights in `src/main.cpp` and run the following command. `t` is the number of threads and `n` is the number of tokens to generate.
```bash
cd cpp
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
./gptxx -t 4 -n 20
```