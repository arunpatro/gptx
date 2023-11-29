#include <iostream>

#include "nn.h"
#include "ops.h"
#include "tensor.h"
#include "utils.h"
#include <getopt.h>
#include <chrono>
#include <omp.h>

int main(int argc, char* argv[]) {
    // configs
    int max_threads = 1;
    int max_tokens = 20;
    char* model_path = (char*)"model_weights.json";

    const char* const short_opts = "n:t:m:";
    const option long_opts[] = {
            {"tokens", required_argument, nullptr, 'n'},
            {"threads", required_argument, nullptr, 't'},
            {"model_path", required_argument, nullptr, 'm'},
            {nullptr, no_argument, nullptr, 0}
    };

    while (true) {
        const auto opt = getopt_long(argc, argv, short_opts, long_opts, nullptr);

        if (-1 == opt)
            break;

        switch (opt) {
            case 'n':
                max_tokens = std::stoi(optarg);
                break;
            case 't':
                max_threads = std::stoi(optarg);
                break;
            case 'm':
                model_path = optarg;
                break;
            case '?': // Unknown option
            default:
                std::cout << "Usage: " << argv[0] << " [-n max_tokens] [-t max_threads] [-m model_path]" << std::endl;
                return 1;
        }
    }

    omp_set_num_threads(num_threads);
    printf("num threads: %d\n", max_threads);
    printf("num tokens: %d\n", max_tokens);
    printf("==========================\n");

    // time this part 
    std::cout << "Loading model from " << model_path << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    auto gpt2 = GPT2::from_json(model_path);
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