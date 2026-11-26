#pragma once

const int max_dim = 8;
constexpr int MATMUL_TILE = 16;

struct MatmulMeta {
    int M;
    int N;
    int K;

    int stride_a_m;
    int stride_a_k;
    int stride_b_k;
    int stride_b_n;

    int batch_ndim;
    int batch_size;
    int batch_shape[max_dim];
    int batch_stride_a[max_dim];
    int batch_stride_b[max_dim];
};

__host__ __device__ inline int batch_offset(
    int batch_ndim,
    const int* batch_shape,
    const int* batch_stride,
    int batch_idx
);

__global__ void matmul_kernel(const float* a, const float* b, float* out, MatmulMeta m);

void launch_matmul(const float* a, const float* b, float* out, const MatmulMeta& m);