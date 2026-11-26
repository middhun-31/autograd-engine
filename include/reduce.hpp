#pragma once
#include <vector>
#include <stdexcept>
const int max_dim = 8;
struct ReduceMeta {
    int ndim;
    int reduce_dim;
    int in_elements;
    int out_elements;
    int shape[max_dim];
    int stride[max_dim];
};

// Host helper: validates dim, builds ReduceMeta, and computes out_shape (respecting keepdim)
inline ReduceMeta make_reduce_meta(
    const std::vector<int>& shape,
    const std::vector<int>& stride,
    int dim,
    bool keepdim,
    std::vector<int>& out_shape)
{
    int ndim = static_cast<int>(shape.size());
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::out_of_range("sum: dim out of range");
    }

    ReduceMeta m{};
    m.ndim = ndim;
    m.reduce_dim = dim;
    m.in_elements = 1;
    m.out_elements = 1;
    out_shape.clear();

    for (int d = 0; d < ndim; ++d) {
        m.shape[d] = shape[d];
        m.stride[d] = stride[d];
        m.in_elements *= shape[d];

        if (d == dim) {
            if (keepdim) out_shape.push_back(1);
        } else {
            m.out_elements *= shape[d];
            out_shape.push_back(shape[d]);
        }
    }
    if (out_shape.empty()) {
        out_shape.push_back(1); // 1-D reduction with keepdim=false -> scalar shape {1}
    }
    return m;
}

void launch_sum_dim(const float* a, float* out, const ReduceMeta& m);
void launch_sum_dim_backward(const float* grad_out, float* grad_in, const ReduceMeta& m);