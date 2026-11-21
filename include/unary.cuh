const int max_dim = 8;
struct UnaryMeta {
    int ndim;
    int no_of_elements;
    int shape[max_dim];
    int stride[max_dim];
};

__host__ __device__ inline int strided_offset(const UnaryMeta& m, int flat);

__global__ void neg_kernel(const float* a, float* out, UnaryMeta m);

void launch_neg(const float* a, float* out, const UnaryMeta& m);

__global__ void exp_kernel(const float* a, float* out, UnaryMeta m);

void launch_exp(const float* a, float* out, const UnaryMeta& m);

__global__ void relu_kernel(const float* a, float* out, UnaryMeta m);

void launch_relu(const float* a, float* out, const UnaryMeta& m);