/// @file lowtrimatrix.hpp
/// @brief GPU-based lower triangular matrix tensor implementation.
///
/// Provides the LowTriMatrix_ class for lower triangular matrix storage and operations on GPU.
/// Optimized storage for lower triangular structure with cuBLAS compatibility.
/// Inherits from both GPU and CPU lower triangular interfaces for mixed-precision computations.

#pragma once
#include "runtime.hpp"
#include "impl/tensor/gpu/gputensor.hpp"
#include "impl/tensor/cpu/lowtrimatrix.hpp"
#include <omp.h>

namespace lahva
{
    namespace gpu
    {
        

    /// @brief Abstract GPU lower triangular matrix base class
    /// @tparam T data type for matrix elements
    template<typename T>
    class LowTriMatrix_ : virtual public GPUTensor_<T>, virtual public cpu::LowTriMatrix_<T>
    {
        public:
            /// @brief Get matrix shape
            /// @return Shape object with dimensions (rows, cols)
            virtual Shape shape() const  = 0;

            /// @brief Element access operator
            /// @param[in] i row index (must be >= j)
            /// @param[in] j column index
            /// @return reference to lower triangular matrix element at (i,j)
            virtual T &operator()(size_t i, size_t j) = 0;

            /// @brief Const element access operator
            /// @param[in] i row index (must be >= j)
            /// @param[in] j column index
            /// @return const reference to lower triangular matrix element at (i,j)
            virtual const T &operator()(size_t i, size_t j) const = 0;
    };

    /// @brief GPU-based lower triangular matrix tensor with optimized storage
    ///
    /// Specialization for lower triangular matrices with optimized memory layout.
    /// Stores only the lower triangular part (approximately half the memory of full matrix).
    /// Supports both host (CPU) and device (GPU) memory. Enforces lower triangular
    /// structure with asserts in debug mode.
    ///
    /// @tparam T data type for matrix elements
    /// @tparam Allocator host (CPU) memory allocator type (default: CudaHostAllocator)
    /// @tparam GPUAllocator device (GPU) memory allocator type (default: CudaDeviceAllocator)
    template <class T, class Allocator = CudaHostAllocator<T>, class GPUAllocator = CudaDeviceAllocator<T>>
    class LowTriMatrix : virtual public GPUTensor<T, Allocator, GPUAllocator>, virtual public LowTriMatrix_<T>
    {
        using alloc_ptr = CPUAllocator<T>;
        using gpualloc_ptr = GPUAllocator_<T>;   
    protected:
        /// @brief Matrix dimension (n x n)
        size_t n_;

        /// @brief Calculate linear index for lower triangular element
        /// @param[in] i row index
        /// @param[in] j column index
        /// @return linear index in 1D storage array
        /// @note Range checks deactivated if NDEBUG is defined
        inline size_t data_id_(size_t i, size_t j) const
        {
            assert(i <= n_ && j <= n_);
            assert(i >= j);
            return (n_ * (j) - (j - 1) * (j) / 2) + (i - j);
        }

        /// @brief Calculate total size needed for lower triangular storage
        /// @param[in] n matrix dimension (n x n)
        /// @return number of elements needed: n*(n+n)/2
        static size_t data_size_(size_t n)
        {
            return (n * n + n) / 2;
        }

        /// @brief Validate matrix dimensions
        /// @param[in] shape dimension to validate
        /// @throws std::out_of_range if dimension exceeds maximum size
        static void check_size_(long unsigned int shape);

    public:
        /// @brief Construct an n x n lower triangular GPU matrix
        /// @param[in] n dimension of square matrix (n x n)
        /// @param[in] cpualloc host (CPU) memory allocator
        /// @param[in] alloc device (GPU) memory allocator
        /// @note values are not guaranteed to be initialized
        LowTriMatrix(size_t n,const alloc_ptr &cpualloc = Allocator(), const gpualloc_ptr &alloc = GPUAllocator());

        /// @brief Construct an n x n lower triangular GPU matrix initialized with a value
        /// @param[in] n dimension of square matrix (n x n)
        /// @param[in] val initial value for all elements
        /// @param[in] cpualloc host (CPU) memory allocator
        /// @param[in] alloc device (GPU) memory allocator
        LowTriMatrix(size_t n, T val,const alloc_ptr &cpualloc = Allocator(), const gpualloc_ptr &alloc = GPUAllocator());

        /// @brief Construct a lower triangular matrix from a vector
        /// @param[in] n dimension of square matrix (n x n)
        /// @param[in] data vector containing values of the lower triangular matrix
        /// @param[in] cpualloc host (CPU) memory allocator
        /// @param[in] alloc device (GPU) memory allocator
        /// @note data must contain n*(n+n)/2 elements in lower triangular format
        LowTriMatrix(size_t n, const Vector<T, Allocator, GPUAllocator> &data, const alloc_ptr &cpualloc = Allocator(), const gpualloc_ptr &alloc = GPUAllocator());

        /// @brief Construct a lower triangular GPU matrix with specified shape
        /// @param[in] shape matrix dimensions (must be square)
        /// @param[in] cpualloc host (CPU) memory allocator
        /// @param[in] alloc device (GPU) memory allocator
        /// @note values are not guaranteed to be initialized
        LowTriMatrix(const Shape &shape, const alloc_ptr &cpualloc = Allocator(), const gpualloc_ptr &alloc = GPUAllocator());

        /// @brief Construct a lower triangular GPU matrix with specified shape and initial value
        /// @param[in] shape matrix dimensions (must be square)
        /// @param[in] val initial value for all elements
        /// @param[in] cpualloc host (CPU) memory allocator
        /// @param[in] alloc device (GPU) memory allocator
        LowTriMatrix(const Shape &shape, T val, const alloc_ptr &cpualloc = Allocator(), const gpualloc_ptr &alloc = GPUAllocator());

        /// @brief Construct a lower triangular matrix from raw data pointer
        /// @param[in] n dimension of square matrix (n x n)
        /// @param[in] data pointer to matrix data in lower triangular format
        /// @param[in] take_ownership if false, data is not freed when matrix is destructed
        /// @param[in] cpualloc host (CPU) memory allocator
        /// @param[in] alloc device (GPU) memory allocator
        /// @note data must be in format defined by data_id_() function
        LowTriMatrix(size_t n, T *data, bool take_ownership = true,const alloc_ptr &cpualloc = Allocator(), const gpualloc_ptr &alloc = GPUAllocator());

        /// @brief Copy constructor for GPU lower triangular matrix
        /// @param[in] other source matrix to copy
        LowTriMatrix(const LowTriMatrix &);

        /// @brief Move constructor for GPU lower triangular matrix
        /// @param[in] other source matrix to move from
        LowTriMatrix(LowTriMatrix &&);

        /// @brief Copy assignment operator
        /// @param[in] other source matrix
        /// @return reference to this matrix
        LowTriMatrix &operator=(const LowTriMatrix &);

        /// @brief Move assignment operator
        /// @param[in] other source matrix to move from
        /// @return reference to this matrix
        LowTriMatrix &operator=(LowTriMatrix &&);

        /// @brief Destructor for GPU lower triangular matrix
        virtual ~LowTriMatrix();

        /// @brief Element access operator for lower triangular matrix
        /// @param[in] i row index (must be >= j)
        /// @param[in] j column index
        /// @return reference to matrix element at (i,j)
        /// @note if NDEBUG is not defined, range checks and lower triangular check are performed
        T &operator()(size_t i, size_t j);

        /// @brief Const element access operator for lower triangular matrix
        /// @param[in] i row index (must be >= j)
        /// @param[in] j column index
        /// @return const reference to matrix element at (i,j)
        /// @note if NDEBUG is not defined, range checks and lower triangular check are performed
        const T &operator()(size_t i, size_t j) const;

        /// @brief Get matrix shape
        /// @return Shape object with (rows, columns)
        Shape shape() const { return Shape(n_, n_); }

        /// @brief Print lower triangular matrix to standard output
        void print() const;

        /// @brief Extract diagonal from lower triangular matrix
        /// @return vector containing diagonal elements
        Vector<T, Allocator, GPUAllocator> get_diagonal() const;

        /// @brief Set diagonal of lower triangular matrix
        /// @param[in] diag vector containing values to set on diagonal
        void set_diagonal(Vector<T, Allocator, GPUAllocator> &diag);
    };

    /// @brief Implementation: Validate matrix dimensions against SIZE_MAX
    template <typename T, class Allocator, class GPUAllocator>
    void LowTriMatrix<T, Allocator, GPUAllocator>::check_size_(long unsigned int n)
    {
        if (n > SIZE_MAX || n > std::sqrt(SIZE_MAX))
        {
            throw std::out_of_range("Vector exceeds maximum LowTriMatrix size.");
        }
    }

    /// @brief Implementation: Allocate lower triangular storage without initialization
    template <typename T, class Allocator, class GPUAllocator>
    LowTriMatrix<T, Allocator, GPUAllocator>::LowTriMatrix(size_t size, const alloc_ptr& alloc , const gpualloc_ptr& gpualloc) :
    GPUTensor<T, Allocator, GPUAllocator>{data_size_(size), alloc, gpualloc}, n_{size}
    {
        check_size_(size);
    }

    /// @brief Implementation: Allocate and fill lower triangular storage with value
    template <typename T, class Allocator, class GPUAllocator>
    LowTriMatrix<T, Allocator, GPUAllocator>::LowTriMatrix(size_t size, T val, const alloc_ptr& alloc , const gpualloc_ptr& gpualloc) :
    LowTriMatrix(size, alloc, gpualloc)
    {
        check_size_(size);
        std::fill(this->data_, this->data_ + data_size_(size), val);
    }

    /// @brief Implementation: Construct from Shape
    template <typename T, class Allocator, class GPUAllocator>
    LowTriMatrix<T, Allocator, GPUAllocator>::LowTriMatrix(const Shape &shape, const alloc_ptr& alloc , const gpualloc_ptr& gpualloc) :
    LowTriMatrix(shape.first, alloc, gpualloc)
    {
        if (shape.first != shape.second)
        {
            throw std::logic_error("shape.first must be equal to shape.second.");
        }
    }

    /// @brief Implementation: Construct from Shape with value
    template <typename T, class Allocator, class GPUAllocator>
    LowTriMatrix<T, Allocator, GPUAllocator>::LowTriMatrix(const Shape &shape, T val, const alloc_ptr& alloc , const gpualloc_ptr& gpualloc) :
    LowTriMatrix{shape.first, val, alloc, gpualloc}
    {
        if (shape.first != shape.second)
        {
            throw std::logic_error("shape.first must be equal to shape.second.");
        }
    }

    /// @brief Implementation: Copy elements from vector into lower triangular storage
    template <typename T, class Allocator, class GPUAllocator>
    LowTriMatrix<T, Allocator, GPUAllocator>::LowTriMatrix(size_t n, const Vector<T, Allocator, GPUAllocator> &data, const alloc_ptr& alloc , const gpualloc_ptr& gpualloc) :
    LowTriMatrix{n, alloc, gpualloc}
    {
        check_size_(data.size());
        #pragma omp for
        for (size_t i = 0; i < data_size_(n_); i++)
        {
            this->data_[i] = data[i];
        }
    }

    /// @brief Implementation: Wrap existing data pointer, set ownership based on parameter
    template <typename T, class Allocator, class GPUAllocator>
    LowTriMatrix<T, Allocator, GPUAllocator>::LowTriMatrix(size_t n, T *data, bool take_ownership, const alloc_ptr &cpualloc, const gpualloc_ptr &gpualloc)
    :  GPUTensor<T, Allocator, GPUAllocator>{cpualloc, gpualloc}, n_{n}
    {
        this->data_ = data;
        this->count_ = data_size_(n);
        this->is_owner_ = take_ownership;

    }

    /// @brief Destructor implementation - base class handles memory cleanup
    template <typename T, class Allocator, class GPUAllocator>
    LowTriMatrix<T, Allocator, GPUAllocator>::~LowTriMatrix()
    {

    }

    /// @brief Copy constructor implementation
    template <typename T, class Allocator, class GPUAllocator>
    LowTriMatrix<T, Allocator, GPUAllocator>::LowTriMatrix(const LowTriMatrix<T, Allocator, GPUAllocator> &other) :
    n_{other.n_},
    GPUTensor<T, Allocator, GPUAllocator>(static_cast<const GPUTensor<T, Allocator, GPUAllocator>&>(other))
    {

    }

    /// @brief Copy assignment operator implementation
    template <typename T, class Allocator, class GPUAllocator>
    LowTriMatrix<T, Allocator, GPUAllocator> &LowTriMatrix<T, Allocator, GPUAllocator>::operator=(const LowTriMatrix<T, Allocator, GPUAllocator> &other)
    {
        if (this != &other)
        {
            GPUTensor<T, Allocator, GPUAllocator>::operator=(static_cast<const GPUTensor<T, Allocator, GPUAllocator>&>(other));
            n_ = other.size();
        }
        return *this;
    }

    /// @brief Move constructor implementation
    template <typename T, class Allocator, class GPUAllocator>
    LowTriMatrix<T, Allocator, GPUAllocator>::LowTriMatrix(LowTriMatrix<T, Allocator, GPUAllocator> &&other) :
    n_{other.n_}, GPUTensor<T, Allocator, GPUAllocator>{static_cast<GPUTensor<T, Allocator, GPUAllocator>&&>(other)}
    {

    }

    /// @brief Move assignment operator implementation
    template <typename T, class Allocator, class GPUAllocator>
    LowTriMatrix<T, Allocator, GPUAllocator> &LowTriMatrix<T, Allocator, GPUAllocator>::operator=(LowTriMatrix<T, Allocator, GPUAllocator> &&other)
    {
       if (this != &other)
        {
            GPUTensor<T, Allocator, GPUAllocator>::operator=(static_cast<GPUTensor<T, Allocator, GPUAllocator>&&>(other));
            n_ = other.size();
            other.n_ = 0;
        }
        return *this;
    }

    /// @brief Implementation: Element access with automatic symmetrization for upper triangle
    /// If i < j, swaps indices to maintain lower triangular access pattern
    template <typename T, class Allocator, class GPUAllocator>
    T &LowTriMatrix<T, Allocator, GPUAllocator>::operator()(size_t i, size_t j)
    {
        if (i < j)
        {
            size_t tmp = i;
            i = j;
            j = tmp;
        }
        return this->data_[data_id_(i, j)];
    }

    /// @brief Implementation: Const element access with automatic symmetrization for upper triangle
    /// If i < j, swaps indices to maintain lower triangular access pattern
    template <typename T, class Allocator, class GPUAllocator>
    const T &LowTriMatrix<T, Allocator, GPUAllocator>::operator()(size_t i, size_t j) const
    {
        if (i < j)
        {
            size_t tmp = i;
            i = j;
            j = tmp;
        }
        return this->data_[data_id_(i, j)];
    }

    /// @brief Implementation: Print lower triangular portion to stdout
    template <typename T, class Allocator, class GPUAllocator>
    void LowTriMatrix<T, Allocator, GPUAllocator>::print() const
    {
        for (size_t i = 0; i < n_; i++)
        {
            for (size_t j = 0; j <= i; j++)
            {
                std::cout << this->data_[data_id_(i, j)] << ", ";
            }
            std::cout << std::endl;
        }
    }

    /// @brief Implementation: Extract diagonal elements into new vector
    template <typename T, class Allocator, class GPUAllocator>
    Vector<T, Allocator, GPUAllocator> LowTriMatrix<T, Allocator, GPUAllocator>::get_diagonal() const
    {
        size_t min_dim = n_;
        Vector<T, Allocator, GPUAllocator> diag(min_dim);

        for (size_t i = 0; i < min_dim; i++)
        {
            diag[i] = (this->data_[data_id_(i, i)]);
        }

        return diag;
    }

    /// @brief Implementation: Set diagonal elements from vector
    template <typename T, class Allocator, class GPUAllocator>
    void LowTriMatrix<T, Allocator, GPUAllocator>::set_diagonal(Vector<T, Allocator, GPUAllocator> &diag)
    {
        size_t min_dim = n_;
#pragma omp for
        for (size_t i = 0; i < min_dim; i++)
        {
            this->data_[data_id_(i, i)] = diag[i];
        }
    }

} // namespace gpu
} // namespace lahva