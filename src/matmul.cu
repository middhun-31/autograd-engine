#pragma once
#include "cuda_check.cuh"
#include "kernels.cuh"
#include "matmul.cuh"
constexpr int MATMUL_TILE = 16;


__host__ __device__ inline int batch_offset(
    int batch_ndim,
    const int* batch_shape,
    const int* batch_stride,
    int batch_idx
) {
    int rem = batch_idx, off = 0;
    for (int d = batch_ndim - 1; d >= 0; --d) {
        off += (rem % batch_shape[d]) * batch_stride[d];
        rem /= batch_shape[d];
    }
    return off;
}

__global__ void matmul_kernel(const float* a, const float* b, float* out, MatmulMeta m) {
    __shared__ float As[MATMUL_TILE][MATMUL_TILE];
    __shared__ float Bs[MATMUL_TILE][MATMUL_TILE];

    const int batch = blockIdx.z;
    const int row   = blockIdx.y * MATMUL_TILE + threadIdx.y;
    const int col   = blockIdx.x * MATMUL_TILE + threadIdx.x;

    const float* A = a + batch_offset(m.batch_ndim, m.batch_shape, m.batch_stride_a, batch);
    const float* B = b + batch_offset(m.batch_ndim, m.batch_shape, m.batch_stride_b, batch);
    float*       C = out + batch * (m.M * m.N);

    float acc = 0.0f;
    const int num_tiles = (m.K + MATMUL_TILE - 1) / MATMUL_TILE;

    for (int t = 0; t < num_tiles; ++t) {
        const int k_a = t * MATMUL_TILE + threadIdx.x;
        const int k_b = t * MATMUL_TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < m.M && k_a < m.K) ? A[row * m.stride_a_m + k_a * m.stride_a_k] : 0.0f;

        Bs[threadIdx.y][threadIdx.x] =
            (k_b < m.K && col < m.N) ? B[k_b * m.stride_b_k + col * m.stride_b_n] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < MATMUL_TILE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < m.M && col < m.N) {
        C[row * m.N + col] = acc;
    }
}

void launch_matmul(const float* a, const float* b, float* out, const MatmulMeta& m) {
    if (m.M == 0 || m.N == 0 || m.batch_size == 0) return;  // a 0-block launch is a launch error

    const dim3 threads(MATMUL_TILE, MATMUL_TILE);
    const dim3 blocks(
        (m.N + MATMUL_TILE - 1) / MATMUL_TILE,
        (m.M + MATMUL_TILE - 1) / MATMUL_TILE,
        m.batch_size
    );

    matmul_kernel<<<blocks, threads>>>(a, b, out, m);
    CUDA_CHECK_KERNEL();
}