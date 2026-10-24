#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "cuda_check.cuh"
#include "kernels.cuh"
#include "tensor.hpp"

int main(){
    const int n = 1 << 20;
    std::vector<float> h_a(n), h_b(n), h_out(n);
    for(int i =0; i<n; ++i){
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2*i);
    }
    float *d_a, *d_b, *d_out;

    CUDA_CHECK(cudaMalloc(&d_a, n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, n*sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), n*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), n*sizeof(float), cudaMemcpyHostToDevice));

    launch_matrix_add(d_a, 1, d_b, 1, d_out, 1, n);

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, n*sizeof(float), cudaMemcpyDeviceToHost));


    bool ok = true;
    for (int i = 0; i < n; ++i) {
        float expected = h_a[i] + h_b[i];
        if (h_out[i] != expected) {
            std::cerr << "Mismatch at " << i << ": got " << h_out[i]
                      << " expected " << expected << "\n";
            ok = false;
            break;
        }
    }
    std::cout << (ok ? "PASS" : "FAIL") << ": strided_add on " << n << " elements\n";

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
    return ok ? 0 : 1;
}