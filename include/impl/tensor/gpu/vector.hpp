/// @file vector.hpp
/// @brief GPU-based vector tensor implementation.
///
/// Provides the Vector_ class for 1-dimensional tensor storage and operations on the GPU.
/// Supports dense vector operations with cuBLAS compatibility and column-major storage.
/// Inherits from both GPU and CPU vector interfaces for mixed-precision computations.

#pragma once
#include "runtime.hpp"
#include "impl/tensor/gpu/gputensor.hpp"
#include "impl/tensor/cpu/vector.hpp"
#include <initializer_list>
#include <omp.h>

namespace lahva
{
    namespace gpu
    {
        
    /// @brief Abstract GPU vector base class
    /// @tparam T data type for vector elements
    template<typename T>
    class Vector_ : public virtual GPUTensor_<T>, public virtual cpu::Vector_<T>
    {

    };

    /// @brief GPU-based 1-dimensional vector tensor with dual memory management
    /// @tparam T data type for vector elements
    /// @tparam Allocator host (CPU) memory allocator type
    /// @tparam GPUAllocator device (GPU) memory allocator type
    template <class T, class Allocator = CudaHostAllocator<T>, class GPUAllocator = CudaDeviceAsyncAllocator<T>>
    class Vector : public GPUTensor<T, Allocator, GPUAllocator>, public virtual Vector_<T>
    {

    public:
        using size_type = std::size_t;
        using alloc_ptr = CPUAllocator<T>;
        using gpualloc_ptr = GPUAllocator;

        /// @brief Default constructor for GPU vector
        Vector() : GPUTensor<T, Allocator, GPUAllocator>() {};

        /// @brief Construct a GPU vector with specified count
        /// @param[in] count number of elements in the vector
        /// @param[in] alloc host (CPU) memory allocator
        /// @param[in] gpualloc device (GPU) memory allocator
        Vector(size_type count, const alloc_ptr &alloc = Allocator(), const gpualloc_ptr &gpualloc = GPUAllocator());
        template<typename U, typename V>
        Vector(size_type count, const CPUAllocator<U> &alloc = Allocator(), const GPUAllocator_<V> &gpualloc = GPUAllocator())
        : GPUTensor<T, Allocator, GPUAllocator>{count, static_cast<Allocator>(alloc), static_cast<GPUAllocator>(gpualloc)}  {};

        /// @brief Construct a GPU vector using GPU memory with CudaRuntime
        /// @param[in] count number of elements in the vector
        /// @param[in] cudart CUDA runtime instance for GPU operations
        /// @param[in] gpualloc device (GPU) memory allocator
        Vector(size_type count, const CudaRuntime &cudart, const gpualloc_ptr  &gpualloc = GPUAllocator());
        template<typename V>
        Vector(size_type count, const CudaRuntime &cudart, const GPUAllocator_<V>  &gpualloc = GPUAllocator())
        : GPUTensor<T, Allocator, GPUAllocator>{gpualloc} { std::cout << "Created here << std::endl";};

        /// @brief Copy constructor for GPU vector
        /// @param[in] x source vector to copy
        Vector(const Vector &x);

        /// @brief Move constructor for GPU vector
        /// @param[in] x source vector to move from
        Vector(Vector &&x);

        /// @brief Construct a GPU vector initialized with a single value
        /// @param[in] count number of elements in the vector
        /// @param[in] value initial value for all elements
        /// @param[in] alloc host (CPU) memory allocator
        /// @param[in] gpualloc device (GPU) memory allocator
        Vector(size_type count, const T &value, const alloc_ptr &alloc = Allocator(), const gpualloc_ptr  &gpualloc = GPUAllocator());

        /// @brief Construct a GPU vector from existing data pointer
        /// @param[in] count number of elements in the vector
        /// @param[in] ptr pointer to data
        /// @param[in] take_onwership if true, vector takes ownership of memory
        /// @param[in] alloc host (CPU) memory allocator
        /// @param[in] gpualloc device (GPU) memory allocator
        Vector(size_type count, T *ptr, bool take_onwership = true, const alloc_ptr &alloc = Allocator(), const gpualloc_ptr  &gpualloc = GPUAllocator());

        /// @brief Construct a GPU vector from const data pointer
        /// @param[in] count number of elements in the vector
        /// @param[in] ptr pointer to constant data to copy
        /// @param[in] alloc host (CPU) memory allocator
        /// @param[in] gpualloc device (GPU) memory allocator
        Vector(size_type count, const T *ptr, const alloc_ptr &alloc = Allocator(), const gpualloc_ptr  &gpualloc = GPUAllocator());

        /// @brief Construct a GPU vector from initializer list
        /// @param[in] init initializer list with vector values
        /// @param[in] alloc host (CPU) memory allocator
        /// @param[in] gpualloc device (GPU) memory allocator
        Vector(std::initializer_list<T> init, const alloc_ptr &alloc = Allocator(), const gpualloc_ptr &gpualloc = GPUAllocator());

        /// @brief Destructor for GPU vector
        ~Vector();

        /// @brief Copy assignment operator
        /// @param[in] other source vector
        /// @return reference to this vector
        Vector &operator=(const Vector &other);

        /// @brief Move assignment operator
        /// @param[in] other source vector to move from
        /// @return reference to this vector
        Vector &operator=(Vector &&other);

        /// @brief Returns iterator to beginning of vector data
        /// @return pointer to first element
        T *begin() const { return &(this->data_[0]); };

        /// @brief Returns iterator to end of vector data
        /// @return pointer to one past the last element
        T *end() const { return &(this->data_[this->count_]); };

        /// @brief Print vector contents to standard output
        void print() const;
    };


    /// @brief Implementation: Allocate vector storage without initialization
    template <typename T, class Allocator, class GPUAllocator>
    Vector<T, Allocator, GPUAllocator>::Vector(size_type count, const alloc_ptr& alloc , const gpualloc_ptr& gpualloc ) :
    Vector_<T>(),
    GPUTensor<T, Allocator, GPUAllocator>{count, alloc, gpualloc} {};

    /// @brief Implementation: Create GPU-only vector using CUDA runtime
    /// Data is not allocated on host, only on GPU device
    template <typename T, class Allocator, class GPUAllocator>
    Vector<T, Allocator, GPUAllocator>::Vector(size_type count, const CudaRuntime &cudart, const gpualloc_ptr& gpualloc ) :
    GPUTensor<T, Allocator, GPUAllocator>{gpualloc}
    {
        this->gpu_buffer = true;
        this->count_ = count;
        this->is_owner_ = false;
        this->is_on_device_ = true;
    };

    /// @brief Implementation: Allocate and fill vector with uniform value
    template <typename T, class Allocator, class GPUAllocator>
    Vector<T, Allocator, GPUAllocator>::Vector(size_type count, const T &value, const alloc_ptr& alloc , const gpualloc_ptr& gpualloc ) :
    Vector{count, alloc, gpualloc}
    {
        std::fill(this->data_, this->data_ + count, value);
    };

    /// @brief Implementation: Wrap existing data pointer with optional ownership
    template <typename T, class Allocator, class GPUAllocator>
    Vector<T, Allocator, GPUAllocator>::Vector(size_type count, T *ptr, bool take_ownership, const alloc_ptr& alloc , const gpualloc_ptr& gpualloc ) :
    GPUTensor<T, Allocator, GPUAllocator>{gpualloc}
    {
        this->data_ = ptr;
        this->count_ = count;
        this->is_owner_ = take_ownership;
    };

    /// @brief Implementation: Copy from const data pointer
    template <typename T, class Allocator, class GPUAllocator>
    Vector<T, Allocator, GPUAllocator>::Vector(size_type count, const T *ptr, const alloc_ptr& alloc , const gpualloc_ptr& gpualloc ) :
    Vector{count, alloc, gpualloc}
    {
        std::copy(ptr, ptr + count, this->data_);
    };

    /// @brief Implementation: Initialize from initializer list
    template <class T, class Allocator, class GPUAllocator>
Vector<T, Allocator, GPUAllocator>::Vector(std::initializer_list<T> init, const alloc_ptr &alloc, const gpualloc_ptr &gpualloc)
    : Vector_<T>(), GPUTensor<T, Allocator, GPUAllocator>(init.size(), alloc, gpualloc)
{
    std::copy(init.begin(), init.end(), this->data_);
}

    /// @brief Copy constructor implementation
    template <typename T, class Allocator, class GPUAllocator>
    Vector<T, Allocator, GPUAllocator>::Vector(const Vector &x) :
    GPUTensor<T, Allocator, GPUAllocator>{static_cast<const GPUTensor<T, Allocator, GPUAllocator>&>(x)}
    {
    };

    /// @brief Move constructor implementation
    template <typename T, class Allocator, class GPUAllocator>
    Vector<T, Allocator, GPUAllocator>::Vector(Vector &&x) :
    GPUTensor<T, Allocator, GPUAllocator>{static_cast<GPUTensor<T, Allocator, GPUAllocator>&&>(x)}
    {
    };

    /// @brief Copy assignment operator implementation
    template <typename T, class Allocator, class GPUAllocator>
    Vector<T, Allocator, GPUAllocator> &Vector<T,Allocator, GPUAllocator>::operator=(const Vector &other)
    {
        if (this != &other)
        {
            GPUTensor<T, Allocator, GPUAllocator>::operator=(static_cast<const GPUTensor<T, Allocator, GPUAllocator>&>(other));
        }
        return *this;
    };

    /// @brief Move assignment operator implementation
    template <typename T, class Allocator, class GPUAllocator>
    Vector<T, Allocator, GPUAllocator> &Vector<T, Allocator, GPUAllocator>::operator=(Vector &&other)
    {
        if (this != &other)
        {
            GPUTensor<T, Allocator, GPUAllocator>::operator=(std::move(other));
        }
        return *this;
    };

    /// @brief Destructor implementation - base class handles memory cleanup
    template <typename T, class Allocator, class GPUAllocator>
    Vector<T, Allocator, GPUAllocator>::~Vector()
    {

    };

    /// @brief Implementation: Print vector elements to stdout with line breaks
    template <typename T, class Allocator, class GPUAllocator>
    void Vector<T, Allocator, GPUAllocator>::print() const
    {
        for (size_t i = 0; i < this->size(); i++)
        {
            std::cout << this->data()[i] << ", ";
            std::cout << std::endl;
        }
    }

} // namespace gpu
} // namespace lahva