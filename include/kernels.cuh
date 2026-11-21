#pragma once

// Declared here (plain C++-visible header) so main.cpp can call into CUDA
// code without itself needing to be compiled by nvcc.
constexpr int max_dim = 8;
// describes the metadata to beb followed for broadcast operations.
struct BroadcastMeta {
    int ndim;
    int numel;                 // elements in the output
    int shape[max_dim];       // output shape
    int stride_a[max_dim];    // a's stride per output dim (0 = broadcast)
    int stride_b[max_dim];    // b's stride per output dim (0 = broadcast)
};
void launch_matrix_add(const float* a, int stride_a,
                         const float* b, int stride_b,
                         float* out, int stride_out,
                         int n);

void launch_add_nd(const float* a, const float* b, float* out,
                   const BroadcastMeta& metadata);
void launch_sub_nd(const float* a, const float* b, float* out,
                   const BroadcastMeta& meta);

void launch_mul_nd(const float* a, const float* b, float* out,
                   const BroadcastMeta& meta);


void launch_div_nd(const float* a, const float* b, float* out,
                   const BroadcastMeta& meta);