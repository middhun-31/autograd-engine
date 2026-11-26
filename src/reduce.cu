#pragma once
#include "cuda_check.cuh"
#include "kernels.cuh"

// Maps a flat output index to the base offset in the strided input (with coord[reduce_dim] = 0)
__host__ __device__ inline int reduce_base_offset(const ReduceMeta& m, int out_flat) {
    int rem = out_flat;
    int off = 0;
    for (int d = m.ndim - 1; d >= 0; --d) {
        if (d == m.reduce_dim) continue;
        off += (rem % m.shape[d]) * m.stride[d];
        rem /= m.shape[d];
    }
    return off;
}

__global__ void sum_dim_kernel(const float* a, float* out, ReduceMeta m) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m.out_elements) return;

    const int base = reduce_base_offset(m, i);
    const int reduce_size = m.shape[m.reduce_dim];
    const int reduce_stride = m.stride[m.reduce_dim];

    float acc = 0.0f;
    for (int r = 0; r < reduce_size; ++r) {
        acc += a[base + r * reduce_stride];
    }
    out[i] = acc;
}

void launch_sum_dim(const float* a, float* out, const ReduceMeta& m) {
    if (m.out_elements == 0) return;
    const int threads = 256;
    const int blocks = (m.out_elements + threads - 1) / threads;
    sum_dim_kernel<<<blocks, threads>>>(a, out, m);
    CUDA_CHECK_KERNEL();
}

// Backward: accumulates contiguous grad_out into contiguous grad_in
__global__ void sum_dim_backward_kernel(const float* grad_out, float* grad_in, ReduceMeta m) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m.in_elements) return;

    // Map flat input index i -> flat output index (skipping reduce_dim)
    int rem = i;
    int out_idx = 0;
    int out_stride = 1;
    for (int d = m.ndim - 1; d >= 0; --d) {
        int coord = rem % m.shape[d];
        rem /= m.shape[d];
        if (d != m.reduce_dim) {
            out_idx += coord * out_stride;
            out_stride *= m.shape[d];
        }
    }
    grad_in[i] += grad_out[out_idx];
}

void launch_sum_dim_backward(const float* grad_out, float* grad_in, const ReduceMeta& m) {
    if (m.in_elements == 0) return;
    const int threads = 256;
    const int blocks = (m.in_elements + threads - 1) / threads;
    sum_dim_backward_kernel<<<blocks, threads>>>(grad_out, grad_in, m);
    CUDA_CHECK_KERNEL();
}