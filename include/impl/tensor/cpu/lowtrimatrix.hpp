/// @file lowtrimatrix.hpp
/// @brief CPU-based lower triangular matrix tensor implementation.
///
/// Provides the LowTriMatrix_ class for lower triangular matrix storage and operations on CPU.
/// Optimized storage for lower triangular structure with BLAS compatibility.
/// Includes specialized operations leveraging triangular matrix properties.

#pragma once
#include <cassert>
#include <omp.h>

#include "impl/tensor/allocators.hpp"
#include "impl/tensor/cpu/vector.hpp"

namespace lahva
{
    namespace cpu
    {
        

    template<typename T>
    class LowTriMatrix_ : virtual public Tensor<T>
    {
        public:
            virtual Shape shape() const  = 0;

            /// @param[in] i row index
            /// @param[in] j column index
            /// @return reference to Matrix element i,j
            virtual T &operator()(size_t i, size_t j) = 0;

            /// @param[in] i row index
            /// @param[in] j column index
            /// @return reference to Matrix element i,j
            virtual const T &operator()(size_t i, size_t j) const = 0;
    };

    /// @brief Lower triangular matrix with optimized storage.
    ///
    /// Stores only lower triangular elements in row-major order of the lower triangle.
    /// Provides compact storage and efficient operations for lower triangular matrices.
    /// If NDEBUG is **not** defined, range checks are performed.
    ///
    template <class T, class Allocator = StdAllocator<T>>
    class LowTriMatrix : virtual public CPUTensor<T, Allocator>, virtual public LowTriMatrix_<T>
    {
        using alloc_ptr = CPUAllocator<T>;   
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
        /// @brief Allocate lower triangular matrix without initialization
        /// @param[in] n matrix dimension (n x n)
        /// @param[in] cpualloc allocator instance for memory management
        LowTriMatrix(size_t n,const alloc_ptr &cpualloc = Allocator());
        template <typename U>
        LowTriMatrix(size_t n, const std::shared_ptr<CPUAllocator<U>> &alloc)
        : LowTriMatrix<T,Allocator>(n, Allocator(*alloc))
        { };

        /// @brief Allocate lower triangular matrix initialized with uniform value
        /// @param[in] n matrix dimension (n x n)
        /// @param[in] val initialization value for all elements
        /// @param[in] cpualloc allocator instance for memory management
        LowTriMatrix(size_t n, T val,const alloc_ptr &cpualloc = Allocator());
        template <typename U>
        LowTriMatrix(size_t n, T val, const std::shared_ptr<CPUAllocator<U>> &alloc)
        : LowTriMatrix<T,Allocator>(n, val, Allocator(*alloc))
        { };

        /// @brief Copy data from vector into lower triangular storage
        /// @param[in] n matrix dimension (n x n)
        /// @param[in] data vector containing lower triangular elements
        /// @param[in] cpualloc allocator instance for memory management
        LowTriMatrix(size_t n, const Vector_<T> &data, const alloc_ptr &cpualloc = Allocator());

        /// @brief Allocate lower triangular matrix with specified shape without initialization
        /// @param[in] shape dimensions (must be square)
        /// @param[in] cpualloc allocator instance for memory management
        /// @throws std::logic_error if shape is not square
        LowTriMatrix(const Shape &shape, const alloc_ptr &cpualloc = Allocator());

        /// @brief Allocate lower triangular matrix with specified shape initialized with uniform value
        /// @param[in] shape dimensions (must be square)
        /// @param[in] val initialization value for all elements
        /// @param[in] cpualloc allocator instance for memory management
        /// @throws std::logic_error if shape is not square
        LowTriMatrix(const Shape &shape, T val, const alloc_ptr &cpualloc = Allocator());

        /// @brief Wrap existing data pointer with optional ownership
        /// @param[in] n matrix dimension (n x n)
        /// @param[in] data pointer to lower triangular matrix data in storage format
        /// @param[in] take_ownership if true, matrix will free data on destruction; if false, external code is responsible
        /// @param[in] cpualloc allocator instance for memory management
        /// @note data must be in lower triangular storage format as defined by data_id_()
        LowTriMatrix(size_t n, T *data, bool take_ownership = true,const alloc_ptr &cpualloc = Allocator());

        /// @brief Copy constructor
        LowTriMatrix(const LowTriMatrix &);

        /// @brief Move constructor
        LowTriMatrix(LowTriMatrix &&);

        /// @brief Copy assignment operator
        LowTriMatrix &operator=(const LowTriMatrix &);

        /// @brief Move assignment operator
        LowTriMatrix &operator=(LowTriMatrix &&);

        /// @brief Destructor - deallocates matrix data if owned
        virtual ~LowTriMatrix();

        /// @brief Non-const element access with automatic index swapping for upper triangle
        /// @param[in] i row index
        /// @param[in] j column index
        /// @return reference to matrix element i,j
        /// @note If i < j, indices are swapped to maintain lower triangular access pattern
        /// @note Range checks deactivated if NDEBUG is defined
        T &operator()(size_t i, size_t j);

        /// @brief Const element access with automatic index swapping for upper triangle
        /// @param[in] i row index
        /// @param[in] j column index
        /// @return reference to matrix element i,j
        /// @note If i < j, indices are swapped to maintain lower triangular access pattern
        /// @note Range checks deactivated if NDEBUG is defined
        const T &operator()(size_t i, size_t j) const;

        /// @brief Get matrix dimensions
        /// @return shape with equal rows and columns
        Shape shape() const { return Shape(n_, n_); }

        /// @brief Print lower triangular portion to stdout
        void print() const;

        /// @brief Extract diagonal elements into new vector
        /// @return vector containing diagonal elements
        Vector<T, Allocator> get_diagonal() const;

        /// @brief Set diagonal elements from vector
        /// @param[in] diag vector containing diagonal values
        /// @throws std::runtime_error if vector size doesn't match matrix dimension
        void set_diagonal(Vector<T, Allocator> &diag);
    };

    /// @brief Implementation: Validate matrix dimensions against SIZE_MAX
    template <typename T, class Allocator>
    void LowTriMatrix<T, Allocator>::check_size_(long unsigned int n)
    {
        if (n > SIZE_MAX || n > std::sqrt(SIZE_MAX))
        {
            throw std::out_of_range("Vector exceeds maximum LowTriMatrix size.");
        }
    }

    /// @brief Implementation: Allocate lower triangular storage without initialization
    template <typename T, class Allocator>
    LowTriMatrix<T, Allocator>::LowTriMatrix(size_t size, const alloc_ptr& alloc ) :
    CPUTensor<T, Allocator>{data_size_(size), alloc}, n_{size}
    {
        check_size_(size);
    }

    /// @brief Implementation: Allocate and fill lower triangular storage with value
    template <typename T, class Allocator>
    LowTriMatrix<T, Allocator>::LowTriMatrix(size_t size, T val, const alloc_ptr& alloc ) :
    LowTriMatrix(size, alloc)
    {
        check_size_(size);

        std::fill(this->data_, this->data_ + data_size_(size), val);
    }

    /// @brief Implementation: Construct from Shape, validate square dimensions
    template <typename T, class Allocator>
    LowTriMatrix<T, Allocator>::LowTriMatrix(const Shape &shape, const alloc_ptr& alloc ) :
    LowTriMatrix(shape.first, alloc)
    {
        if (shape.first != shape.second)
        {
            throw std::logic_error("shape.first must be equal to shape.second.");
        }
    }

    /// @brief Implementation: Construct from Shape with value, validate square dimensions
    template <typename T, class Allocator>
    LowTriMatrix<T, Allocator>::LowTriMatrix(const Shape &shape, T val, const alloc_ptr& alloc ) :
    LowTriMatrix{shape.first, val, alloc}
    {
        if (shape.first != shape.second)
        {
            throw std::logic_error("shape.first must be equal to shape.second.");
        }
    }

    /// @brief Implementation: Copy elements from vector into lower triangular storage
    template <typename T, class Allocator>
    LowTriMatrix<T, Allocator>::LowTriMatrix(size_t n, const Vector_<T> &data, const alloc_ptr& alloc ) :
    LowTriMatrix{n, alloc}
    {
        check_size_(data.size());
        #pragma omp for
        for (size_t i = 0; i < data_size_(n_); i++)
        {
            this->data_[i] = data[i];
        }
    }


    /// @brief Implementation: Wrap existing data pointer, set ownership based on parameter
    template <typename T, class Allocator>
    LowTriMatrix<T, Allocator>::LowTriMatrix(size_t n, T *data, bool take_ownership,const alloc_ptr &cpualloc)
    :  CPUTensor<T, Allocator>{cpualloc}, n_{n}
    {
        this->data_ = data;
        this->count_ = data_size_(n);
        this->is_owner_ = take_ownership;

    }

    /// @brief Destructor implementation
    template <typename T, class Allocator>
    LowTriMatrix<T, Allocator>::~LowTriMatrix()
    {

    }

    /// @brief Copy constructor implementation
    template <typename T, class Allocator>
    LowTriMatrix<T, Allocator>::LowTriMatrix(const LowTriMatrix<T, Allocator> &other) :
    n_{other.n_},
    CPUTensor<T, Allocator>(other)
    {

    }

    /// @brief Copy assignment operator implementation
    template <typename T, class Allocator>
    LowTriMatrix<T, Allocator> &LowTriMatrix<T, Allocator>::operator=(const LowTriMatrix<T, Allocator> &other)
    {
        if (this != &other)
        {
            CPUTensor<T, Allocator>::operator=(other);
            n_ = other.size();
        }
        return *this;
    }

    /// @brief Move constructor implementation
    template <typename T, class Allocator>
    LowTriMatrix<T, Allocator>::LowTriMatrix(LowTriMatrix<T, Allocator> &&other) :
    n_{other.n_}, CPUTensor<T, Allocator>{other}
    {

    }

    /// @brief Move assignment operator implementation
    template <typename T, class Allocator>
    LowTriMatrix<T, Allocator> &LowTriMatrix<T, Allocator>::operator=(LowTriMatrix<T, Allocator> &&other)
    {
       if (this != &other)
        {
            CPUTensor<T, Allocator>::operator=(std::move(other));
            n_ = other.size();
            other.n_ = 0;
        }
        return *this;
    }

    /// @brief Implementation: Element access with automatic symmetrization for upper triangle
    /// If i < j, swaps indices to maintain lower triangular access pattern
    template <typename T, class Allocator>
    T &LowTriMatrix<T, Allocator>::operator()(size_t i, size_t j)
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
    template <typename T, class Allocator>
    const T &LowTriMatrix<T, Allocator>::operator()(size_t i, size_t j) const
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
    template <typename T, class Allocator>
    void LowTriMatrix<T, Allocator>::print() const
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
    template <typename T, class Allocator>
    Vector<T, Allocator> LowTriMatrix<T, Allocator>::get_diagonal() const
    {
        size_t min_dim = n_;
        Vector<T, Allocator> diag(min_dim);

        for (size_t i = 0; i < min_dim; i++)
        {
            diag[i] = (this->data_[data_id_(i, i)]);
        }

        return diag;
    }

    /// @brief Implementation: Set diagonal elements from vector
    template <typename T, class Allocator>
    void LowTriMatrix<T, Allocator>::set_diagonal(Vector<T, Allocator> &diag)
    {
        size_t min_dim = n_;
#pragma omp for
        for (size_t i = 0; i < min_dim; i++)
        {
            this->data_[data_id_(i, i)] = diag[i];
        }
    }

} // namespace cpu
} // namespace lahva