#include <iostream>

#include "nn.h"
#include "ops.h"
#include "tensor.h"
#include "utils.h"
// #include <omp.h>
#include <chrono>

int main(int argc, char* argv[]) {
    // configs
    int max_threads = 4;
    int max_tokens = 20;
    printf("num threads: %d\n", max_threads);
    printf("num tokens: %d\n", max_tokens);
    printf("==========================");

    // time this part 
    auto start = std::chrono::high_resolution_clock::now();
    auto gpt2 = GPT2::from_json("/Users/arunpatro/multicore-project/gptx/python/model_weights.json");
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;


    std::vector<size_t> input_tokens = {464, 3280, 284, 1204, 11, 262, 6881, 11, 290, 2279, 318};

    auto start2 = std::chrono::high_resolution_clock::now();
    std::vector<size_t> output_tokens = gpt2.generate(input_tokens, max_tokens);
    auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration2 = end2 - start2;

    // print stats
    printf("===============================\n");
    printf("load time: %.2f\n", duration.count());
    printf("inference time: %.2f\n", duration2.count());
    printf("s / token: %f\n", duration2.count() / max_tokens);
    printf("token / s: %f\n", max_tokens / duration2.count());

    return 0;
}