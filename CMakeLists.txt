cmake_minimum_required(VERSION 3.24)
project(autograd_engine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # CUDA 12.0 cannot compile architecture compute_120. Volta through
    # Turing/Tesla-style binaries are supported by this toolkit.
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -lineinfo")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -lineinfo")

add_executable(autograd_engine
    src/main.cpp
    src/tensor.cpp
    src/kernels.cu
    src/unary_kernel.cu
)
target_include_directories(autograd_engine PRIVATE include)
set_target_properties(autograd_engine PROPERTIES CUDA_SEPARABLE_COMPILATION ON)