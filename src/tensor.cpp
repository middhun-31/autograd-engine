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
    BroadcastMeta m = make_broadcast_metadata(*this, other);
    Tensor result(std::vector<int>(m.shape, m.shape + m.ndim));
    if (no_of_elements_ == 0) {
        return result;
    }

    launch_add_nd(data_.get() + offset_, other.data_.get() + other.offset_,
                  result.data_.get(), m);
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

static BroadcastMeta make_broadcast_metadata(const Tensor& a, const Tensor& b){
    const int nd = std::max(a.ndim(), b.ndim());          // the dimension of the tensor with max dimension is chosen
    if(nd > max_dim){
        throw std::invalid_argument("Number of dimensions greater than maximum allowed dimensions");
    }

    BroadcastMeta m{};               // initialising new datastructure to return metadata
    m.numel = 1;
    m.ndim = nd;

    for(int i = 0; i < nd; ++i){
        int ai = a.ndim() - 1 - i;
        int bi = b.ndim() - 1 - i;

        int da = (ai < 0)? 1:a.shape()[ai];
        int db = (bi < 0)? 1:b.shape()[bi];

        int out_dim;
        if (da == db)      
            out_dim = da;
        else if (da == 1)  
            out_dim = db;
        else if (db == 1)  
            out_dim = da;
        else 
            throw std::invalid_argument("Shapes are not broadcastable");

        int pos = nd - 1 - i;                 // position in output shape
        m.shape[pos]    = out_dim;
        m.stride_a[pos] = (ai >= 0 && da != 1) ? a.strides()[ai] : 0;
        m.stride_b[pos] = (bi >= 0 && db != 1) ? b.strides()[bi] : 0;
        m.numel *= out_dim;
    }
    return m;
}

Tensor Tensor::from_host(const std::vector<int>& shape,
                         const std::vector<float>& values) {
    Tensor t(shape);
    if (static_cast<int>(values.size()) != t.no_of_elements_)
        throw std::invalid_argument("from_host: size mismatch");
    if (t.no_of_elements_ > 0)
        CUDA_CHECK(cudaMemcpy(t.data_.get(), values.data(),
                              values.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
    return t;
}