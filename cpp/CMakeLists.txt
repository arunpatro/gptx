cmake_minimum_required(VERSION 3.10)
project(gptx)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/ext)

# Define source files
set(SOURCE_FILES 
    src/main.cpp 
    src/tensor.cpp 
    src/ops.cpp 
    src/utils.cpp 
    src/einops.cpp 
    src/nn.cpp)

# Add executable
add_executable(gptx ${SOURCE_FILES})

# Find and link OpenMP if available
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(gptx PUBLIC OpenMP::OpenMP_CXX)
endif()
