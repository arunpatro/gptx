#!/bin/bash

# Define the values for 't' to iterate over
values=(1 2 4 8 16 32)

# Path to the model weights (adjust as needed)
MODEL_PATH="../python/model_weights.json"

# Path to the executable (adjust as needed)
EXECUTABLE="./build/gptx"

# Check if the executable exists
if [ ! -f "$EXECUTABLE" ]; then
    echo "Executable not found: $EXECUTABLE"
    exit 1
fi

# Iterate over the values and run the command
for t in "${values[@]}"; do
    echo "Running with t=$t"
    $EXECUTABLE -m $MODEL_PATH -t $t > "cpp_$t.txt"
    echo "Output saved to cpp_$t.txt"
done

echo "All runs completed."
