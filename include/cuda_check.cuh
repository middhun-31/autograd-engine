#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Wrap every CUDA runtime call in this. Silent CUDA errors are the single
// biggest time-sink in early CUDA projects -- an out-of-bounds write in one
// kernel often doesn't crash until three kernels later.
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                         \
        if (err__ != cudaSuccess) {                                          \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,      \
                         __LINE__, cudaGetErrorString(err__));               \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                    \
    } while (0)


// Call after every kernel launch in debug builds. Kernel launches are async,
// so errors (bad launch config, illegal memory access) don't surface until
// the next sync point unless you check explicitly.
#define CUDA_CHECK_KERNEL()                                                  \
    do {                                                                     \
        CUDA_CHECK(cudaGetLastError());                                      \
        CUDA_CHECK(cudaDeviceSynchronize());                                 \
    } while (0)
// End of CUDA_CHECK_KERNEL macro.