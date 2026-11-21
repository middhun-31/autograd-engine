#pragma once
#include "cuda_check.cuh"
#include "kernels.cuh"

struct UnaryMeta {
    int ndim;
    int no_of_elements;
    int shape[max_dim];
    int stride[max_dim];
};

__host__ __device__ inline int strided_offset(const UnaryMeta& m, int flat) {
    int rem = flat, off = 0;
    for (int d = m.ndim - 1; d >= 0; --d) {
        off += (rem % m.shape[d]) * m.stride[d];
        rem /= m.shape[d];
    }
    return off;
}

__global__ void neg_kernel(const float* a, float* out, UnaryMeta m) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m.no_of_elements) return;
    out[i] = -a[strided_offset(m, i)];
}

void launch_neg(const float* a, float* out, const UnaryMeta& m) {
    if (m.no_of_elements == 0) return;                      // a 0-block launch is a launch error
    const int threads = 256;
    const int blocks = (m.no_of_elements + threads - 1) / threads;
    neg_kernel<<<blocks, threads>>>(a, out, m);
    CUDA_CHECK_KERNEL();
}

__global__ void exp_kernel(const float* a, float* out, UnaryMeta m) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m.no_of_elements) return;
    out[i] = exp(a[strided_offset(m, i)]);
}

void launch_exp(const float* a, float* out, const UnaryMeta& m) {
    if (m.no_of_elements == 0) return;                      
    const int threads = 256;
    const int blocks = (m.no_of_elements + threads - 1) / threads;
    exp_kernel<<<blocks, threads>>>(a, out, m);
    CUDA_CHECK_KERNEL();
}

__global__ void relu_kernel(const float* a, float* out, UnaryMeta m) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m.no_of_elements) return;
    out[i] = fmaxf(a[strided_offset(m, i)], 0.0f);
}

void launch_relu(const float* a, float* out, const UnaryMeta& m) {
    if (m.no_of_elements == 0) return;                      
    const int threads = 256;
    const int blocks = (m.no_of_elements + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(a, out, m);
    CUDA_CHECK_KERNEL();
}