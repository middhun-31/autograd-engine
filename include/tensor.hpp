#include <memory>
#include <vector>

class Tensor{
    std::shared_ptr<float> data_;
    std::vector<int> shape_;
    std::vector<int> strides_;
    int offset_;
    int no_of_elements_;
    

    static std::vector<int> contiguous_strides(const std::vector<int>& shape);
    static int compute_elements(const std::vector<int>& shape);


public:
    explicit Tensor(const std::vector<int>& shape);      // constructor that allocates floats on device and owns memory

      // constructor for non-owning operations like transpose /reshape
    Tensor(std::shared_ptr<float> data, std::vector<int> shape,
           std::vector<int> strides, int offset);

    // disabling copy and assignment as memory is owned
    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = delete;
    //move operation is permitted
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
     
    //functions to access private elements
    const std::vector<int>& shape() const { return shape_; }
    const std::vector<int>& strides() const { return strides_; }
    int offset() const { return offset_; }
    int no_of_elements() const { return no_of_elements_; }
    int ndim() const { return static_cast<int>(shape_.size()); }
    float* data() const { return data_.get(); }

    // Metadata-only: returns a view, no data movement.
    Tensor reshape(const std::vector<int>& new_shape) const;
    Tensor transpose(int dim0, int dim1) const;


    Tensor add(const Tensor& other) const;
    
    std::vector<float> to_cpu() const;

    
};