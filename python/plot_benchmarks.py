import matplotlib.pyplot as plt
import pandas as pd


threads = [1, 2, 4, 8, 16, 32, 64, 128, 256]

rust_data = []
for t in threads:
    with open(f"../rust/rust_{t}.txt") as f:
        data = f.readlines()
    vals = [t] + [float(line.strip().split(" ")[-1]) for line in data[-4:]]
    rust_data.append(vals)
    
cpp_data = []
for t in threads:
    with open(f"../cpp/cpp_{t}.txt") as f:
        data = f.readlines()
    vals = [t] + [float(line.strip().split(" ")[-1]) for line in data[-4:]]
    cpp_data.append(vals)
    
    
df_rust = pd.DataFrame(rust_data, columns=["threads", "model load time", "inference time", "seconds-per-token", "tokens-per-second"])
df_cpp = pd.DataFrame(cpp_data, columns=["threads", "model load time", "inference time", "seconds-per-token", "tokens-per-second"])

# speedup
df_rust["speedup"] = df_rust["tokens-per-second"] / df_rust["tokens-per-second"][0]
df_cpp["speedup"] = df_cpp["tokens-per-second"] / df_cpp["tokens-per-second"][0]

## plot the columns
columns = ["tokens-per-second", "seconds-per-token", "speedup"]
for column in columns:
    fig, ax = plt.subplots()
    ax.plot(df_rust.threads, df_rust[column], 'o-', label="Rust")
    ax.plot(df_cpp.threads, df_cpp[column], 'o-', label="C++")

    ax.set_title(f"{column} vs. Threads in Rust and C++")
    ax.set_xlabel("Threads")
    ax.set_ylabel(column)

    ax.legend()

    plt.savefig(f"{column}_rust_cpp.png")