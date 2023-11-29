cmake_minimum_required(VERSION 3.10)
project(gptx)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(gptx main.cpp tensor.cpp ops.cpp utils.cpp einops.cpp nn.cpp)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/ext)
