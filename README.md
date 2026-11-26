# autograd-engine

A small experimental CUDA tensor/autograd engine written in C++17 and CUDA C++.
The project currently provides GPU-backed tensor storage, reshape and transpose
views, broadcasting metadata, element-wise CUDA kernels, and a command-line
demo that verifies a large vector addition.

> Status: work in progress. The public CMake target currently builds the tensor
> implementation and the element-wise/addition demo. Matmul and reduction sources
> exist in the repository but are not yet included in `CMakeLists.txt`.

## Features

- GPU-owned tensors using `cudaMalloc` and shared ownership
- Host-to-device tensor construction with `Tensor::from_host`
- Device-to-host copying with `Tensor::to_cpu`
- Metadata-only `reshape` and `transpose` tensor views
- Strided and broadcast element-wise addition
- CUDA kernels for matrix/ND addition, subtraction, multiplication, division,
  negation, exponentiation, ReLU, matmul, and reductions
- CUDA error reporting and post-launch synchronization
- Support for up to 8 tensor dimensions

## Requirements

- NVIDIA CUDA Toolkit
- An NVIDIA GPU with a CUDA-capable driver
- CMake 3.24 or newer
- A C++17-compatible host compiler supported by CUDA

The default CMake CUDA architecture is `75` (Turing). Override it when
configuring if you need another architecture:

```powershell
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86
```

## Build

From the repository root:

```powershell
cmake -S . -B build
cmake --build build --config Release
```

Run the executable:

```powershell
.\build\Release\autograd_engine.exe
```

For generators that place executables directly under the build directory:

```powershell
.\build\autograd_engine
```

On Linux or other Unix-like systems:

```bash
cmake -S . -B build
cmake --build build --config Release
./build/autograd_engine
```

## Demo

`src/main.cpp` creates two host vectors containing 1,048,576 `float` values,
copies them to GPU memory, performs strided element-wise addition, copies the
result back, and validates every result.

A successful run prints:

```text
PASS: strided_add on 1048576 elements
```

## C++ API

`Tensor` represents float device data together with shape and stride metadata.

```cpp
#include "tensor.hpp"

Tensor a = Tensor::from_host({2, 3}, {
    1.0f, 2.0f, 3.0f,
    4.0f, 5.0f, 6.0f
});

Tensor b = Tensor::from_host({2, 3}, {
    6.0f, 5.0f, 4.0f,
    3.0f, 2.0f, 1.0f
});

Tensor c = a.add(b);
std::vector<float> values = c.to_cpu();
```

Useful operations:

- `Tensor({rows, columns})` allocates an uninitialized GPU tensor
- `Tensor::from_host(shape, values)` uploads host values
- `Tensor::to_cpu()` copies tensor values, including strided views, back to CPU
- `reshape(new_shape)` creates a metadata view when the tensor is contiguous
- `transpose(dim0, dim1)` swaps dimensions and strides
- `add(other)` launches an ND broadcast-aware CUDA addition

## Project Layout

```text
include/
  tensor.hpp        Tensor API and metadata
  kernels.cuh       Element-wise CUDA launch declarations
  matmul.cuh        Tiled matmul metadata and declarations
  reduce.hpp        Reduction metadata helpers
  unary.cuh         Unary operation declarations
  cuda_check.cuh    CUDA error-checking macros

src/
  main.cpp          Demo and executable entry point
  tensor.cpp        Tensor implementation and broadcasting logic
  kernels.cu        Element-wise CUDA kernels
  unary_kernel.cu   Negate, exp, and ReLU kernels
  matmul.cu         Tiled shared-memory matmul kernel
  reduce.cu         Sum reduction and backward kernels
```

## Development Notes

- Tensor strides are stored in element units, not bytes.
- Broadcast dimensions use a zero stride for dimensions of size 1.
- Transposed and reshaped tensors share the original device allocation.
- Every CUDA launch in this project synchronizes and checks for launch/runtime
  errors through `CUDA_CHECK_KERNEL()`.
- The Tensor constructor and host upload helpers may terminate the process on
  CUDA runtime errors through `CUDA_CHECK`.

## Limitations

- Tensor values are `float` only.
- Tensor storage is allocated on CUDA devices; a CUDA-capable environment is
  required.
- The Tensor API currently implements reshape and broadcast-aware addition.
- No automatic differentiation graph or optimizer is implemented yet.
- Dimensions are currently limited to 8.
- Negative or incompatible shapes throw C++ exceptions where validation exists.