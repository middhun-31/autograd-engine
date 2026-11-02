#pragma once
#include "cuda_check.cuh"
#include "kernels.cuh"

__global__ void add_matrix(const float* a, int stride_a, const float* b, int stride_b, float*out, int stride_out, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > n) {
        out[i * stride_out] = a[i * stride_a] + b[i * stride_b];
    }
}

void launch_matrix_add(const float* a, int stride_a, const float* b, int stride_b, float*out, int stride_out, int n){
    int threads = 256;
    int blocks = (ceil((n+threads-1)/threads));
    add_matrix <<<blocks, threads>>>(a, stride_a, b, stride_b, out, stride_out, n);
    CUDA_CHECK_KERNEL();
}

__global__ void add_nd_kernel(const float* a, const float* b, float* out, BroadcastMeta metadata){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int ia = 0, ib = 0;
    if(i > metadata.numel)
        return;
        
    int rem = i;

    for(int d = metadata.ndim - 1; d > 0; --d){
        int coord = rem % metadata.shape[d];
        rem /= metadata.shape[d];

        ia += coord * metadata.stride_a[d];
        ib += coord * metadata.stride_b[d];
    }
    out[i] = a[ia] + b[ib];
}

__global__ void sub_nd_kernel(const float* a, const float* b, float* out, BroadcastMeta metadata){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int ia = 0, ib = 0;
    if(i > metadata.numel)
        return;
        
    int rem = i;

    for(int d = metadata.ndim - 1; d > 0; --d){
        int coord = rem % metadata.shape[d];
        rem /= metadata.shape[d];

        ia += coord * metadata.stride_a[d];
        ib += coord * metadata.stride_b[d];
    }
    out[i] = a[ia] - b[ib];
}

__global__ void mul_nd_kernel(const float* a, const float* b, float* out, BroadcastMeta metadata){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int ia = 0, ib = 0;
    if(i > metadata.numel)
        return;
        
    int rem = i;

    for(int d = metadata.ndim - 1; d > 0; --d){
        int coord = rem % metadata.shape[d];
        rem /= metadata.shape[d];

        ia += coord * metadata.stride_a[d];
        ib += coord * metadata.stride_b[d];
    }
    out[i] = a[ia] * b[ib];
}

__global__ void div_nd_kernel(const float* a, const float* b, float* out, BroadcastMeta metadata){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int ia = 0, ib = 0;
    if(i > metadata.numel)
        return;
        
    int rem = i;

    for(int d = metadata.ndim - 1; d > 0; --d){
        int coord = rem % metadata.shape[d];
        rem /= metadata.shape[d];

        ia += coord * metadata.stride_a[d];
        ib += coord * metadata.stride_b[d];
    }
    out[i] = a[ia] / b[ib];
}



void launch_add_nd(const float* a, const float* b, float* out,
                   const BroadcastMeta& meta) {
    int threads = 256;
    int blocks = (meta.numel + threads - 1) / threads;
    add_nd_kernel<<<blocks, threads>>>(a, b, out, meta);
    CUDA_CHECK_KERNEL();
}

void launch_sub_nd(const float* a, const float* b, float* out,
                   const BroadcastMeta& meta) {
    int threads = 256;
    int blocks = (meta.numel + threads - 1) / threads;
    sub_nd_kernel<<<blocks, threads>>>(a, b, out, meta);
    CUDA_CHECK_KERNEL();
}

void launch_mul_nd(const float* a, const float* b, float* out,
                   const BroadcastMeta& meta) {
    int threads = 256;
    int blocks = (meta.numel + threads - 1) / threads;
    mul_nd_kernel<<<blocks, threads>>>(a, b, out, meta);
    CUDA_CHECK_KERNEL();
}

void launch_mul_nd(const float* a, const float* b, float* out,
                   const BroadcastMeta& meta) {
    int threads = 256;
    int blocks = (meta.numel + threads - 1) / threads;
    mul_nd_kernel<<<blocks, threads>>>(a, b, out, meta);
    CUDA_CHECK_KERNEL();
}

void launch_div_nd(const float* a, const float* b, float* out,
                   const BroadcastMeta& meta) {
    int threads = 256;
    int blocks = (meta.numel + threads - 1) / threads;
    div_nd_kernel<<<blocks, threads>>>(a, b, out, meta);
    CUDA_CHECK_KERNEL();
}