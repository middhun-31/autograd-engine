#include "tensor.hpp"
#include "cuda_check.cuh"
#include "kernels.cuh"

#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

std::vector<int> Tensor::contiguous_strides(const std::vector<int>& shape) {
    std::vector<int> result(shape.size());
    long long running = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        if (shape[i] < 0) {
            throw std::invalid_argument("Tensor dimensions cannot be negative");
        }
        if (running > std::numeric_limits<int>::max()) {
            throw std::overflow_error("Tensor strides exceed the supported range");
        }
        result[i] = static_cast<int>(running);
        running *= shape[i];
    }
    return result;
}

int Tensor::compute_elements(const std::vector<int>& shape) {
    long long result = 1;
    for (int dimension : shape) {
        if (dimension < 0) {
            throw std::invalid_argument("Tensor dimensions cannot be negative");
        }
        result *= dimension;
        if (result > std::numeric_limits<int>::max()) {
            throw std::overflow_error("Tensor contains too many elements");
        }
    }
    return static_cast<int>(result);
}

Tensor::Tensor(const std::vector<int>& shape)
    : shape_(shape),
      strides_(contiguous_strides(shape)),
      offset_(0),
      no_of_elements_(compute_elements(shape)) {
    float* raw = nullptr;
    if (no_of_elements_ > 0) {
        CUDA_CHECK(cudaMalloc(&raw, static_cast<std::size_t>(no_of_elements_) * sizeof(float)));
    }
    data_ = std::shared_ptr<float>(raw, [](float* pointer) {
        if (pointer) {
            cudaFree(pointer);
        }
    });
}

Tensor::Tensor(std::shared_ptr<float> data, std::vector<int> shape,
               std::vector<int> strides, int offset)
    : data_(std::move(data)),
      shape_(std::move(shape)),
      strides_(std::move(strides)),
      offset_(offset),
      no_of_elements_(compute_elements(shape_)) {
    if (strides_.size() != shape_.size()) {
        throw std::invalid_argument("Tensor shape and strides must have the same rank");
    }
    if (offset_ < 0) {
        throw std::invalid_argument("Tensor offset cannot be negative");
    }
    if (no_of_elements_ > 0 && !data_) {
        throw std::invalid_argument("A non-empty tensor view requires device storage");
    }
}

Tensor::Tensor(Tensor&& other) noexcept = default;

Tensor& Tensor::operator=(Tensor&& other) noexcept = default;

Tensor Tensor::reshape(const std::vector<int>& new_shape) const {
    if (compute_elements(new_shape) != no_of_elements_) {
        throw std::invalid_argument("Reshape must preserve the number of elements");
    }
    if (contiguous_strides(shape_) != strides_) {
        throw std::invalid_argument("Reshape currently requires a contiguous tensor");
    }
    return Tensor(data_, new_shape, contiguous_strides(new_shape), offset_);
}

Tensor Tensor::transpose(int dim0, int dim1) const {
    const int rank = ndim();
    if (dim0 < 0) dim0 += rank;
    if (dim1 < 0) dim1 += rank;
    if (dim0 < 0 || dim0 >= rank || dim1 < 0 || dim1 >= rank) {
        throw std::out_of_range("Transpose dimension is out of range");
    }

    std::vector<int> transposed_shape = shape_;
    std::vector<int> transposed_strides = strides_;
    std::swap(transposed_shape[dim0], transposed_shape[dim1]);
    std::swap(transposed_strides[dim0], transposed_strides[dim1]);
    return Tensor(data_, std::move(transposed_shape), std::move(transposed_strides), offset_);
}

Tensor Tensor::add(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensor addition requires matching shapes");
    }
    if (ndim() > 1) {
        throw std::invalid_argument("Tensor addition currently supports vectors and scalars only");
    }

    Tensor result(shape_);
    if (no_of_elements_ == 0) {
        return result;
    }

    const int stride_a = ndim() == 0 ? 1 : strides_[0];
    const int stride_b = other.ndim() == 0 ? 1 : other.strides_[0];
    launch_matrix_add(data_.get() + offset_, stride_a,
                      other.data_.get() + other.offset_, stride_b,
                      result.data_.get(), 1, no_of_elements_);
    return result;
}

std::vector<float> Tensor::to_cpu() const {
    std::vector<float> result(static_cast<std::size_t>(no_of_elements_));
    if (no_of_elements_ == 0) {
        return result;
    }

    if (contiguous_strides(shape_) == strides_) {
        CUDA_CHECK(cudaMemcpy(result.data(), data_.get() + offset_,
                              result.size() * sizeof(float), cudaMemcpyDeviceToHost));
        return result;
    }

    for (int flat_index = 0; flat_index < no_of_elements_; ++flat_index) {
        int remainder = flat_index;
        int device_index = offset_;
        for (int dim = ndim() - 1; dim >= 0; --dim) {
            const int coordinate = remainder % shape_[dim];
            remainder /= shape_[dim];
            device_index += coordinate * strides_[dim];
        }
        CUDA_CHECK(cudaMemcpy(&result[flat_index], data_.get() + device_index,
                              sizeof(float), cudaMemcpyDeviceToHost));
    }
    return result;
}