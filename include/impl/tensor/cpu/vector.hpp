/// @file vector.hpp
/// @brief CPU-based vector tensor implementation.
///
/// Provides the Vector_ class for 1-dimensional tensor storage and operations on CPU.
/// Supports dense vector operations with BLAS compatibility and column-major storage.
/// Vectors can be initialized from various sources and support mathematical operations.

#pragma once

#include "impl/tensor/cpu/tensor.hpp"

namespace lahva
{
    namespace cpu
    {

        template <typename T>
        class Vector_ : public virtual Tensor_<T>
        {
        };

        /// @brief Vector wrapper
        template <class T, class Allocator = StdAllocator<T>>
        class Vector : public Tensor<T, Allocator>, virtual public Vector_<T>
        {

        public:
            using size_type = std::size_t;
            using alloc_ptr = CPUAllocator<T>;

            /// @brief Default constructor - creates empty vector without allocation
            Vector() { this->no_alloc = true; };

            /// @brief Allocate vector storage without initialization
            /// @param[in] count number of elements to allocate
            /// @param[in] alloc allocator instance for memory management
            Vector(size_type count, const alloc_ptr &alloc = Allocator());

            /// @brief Copy constructor
            /// @param[in] x source vector to copy
            Vector(const Vector &x);

            /// @brief Move constructor
            /// @param[in] x source vector to move from
            Vector(Vector &&x) noexcept;

            /// @brief Allocate vector initialized with uniform value
            /// @param[in] count number of elements to allocate
            /// @param[in] value initialization value for all elements
            /// @param[in] alloc allocator instance for memory management
            Vector(size_type count, const T &value, const alloc_ptr &alloc = Allocator());

            /// @brief Wrap existing data pointer with optional ownership
            /// @param[in] count number of elements
            /// @param[in] ptr pointer to vector data
            /// @param[in] take_onwership if true, vector will free data on destruction; if false, external code is responsible
            /// @param[in] alloc allocator instance for memory management
            Vector(size_type count, T *ptr, bool take_onwership = true, const alloc_ptr &alloc = Allocator());

            /// @brief Copy data from const pointer into newly allocated vector
            /// @param[in] count number of elements
            /// @param[in] ptr pointer to read-only vector data
            /// @param[in] alloc allocator instance for memory management
            Vector(size_type count, const T *ptr, const alloc_ptr &alloc = Allocator());

            /// @brief Initialize vector from initializer list
            /// @param[in] init initializer list with element values
            /// @param[in] alloc allocator instance for memory management
            Vector(std::initializer_list<T> init, const alloc_ptr &alloc = Allocator());

            /// @brief Destructor - deallocates vector data if owned
            ~Vector();

            /// @brief Copy assignment operator
            Vector &operator=(const Vector &other);

            /// @brief Move assignment operator
            Vector &operator=(Vector &&other);

            /// @brief Non-const element access operator
            /// @tparam D index type (implicitly converted to size_t)
            /// @param[in] index element index
            /// @return reference to element at index
            template <typename D>
            T &operator[](D index) const
            {
                return this->data_[static_cast<size_t>(index)];
            };

            /// @brief Const element access operator
            /// @tparam D index type (implicitly converted to size_t)
            /// @param[in] index element index
            /// @return reference to element at index
            template <typename D>
            T &operator[](D index)
            {
                return this->data_[static_cast<size_t>(index)];
            };

            /// @brief Get pointer to first element
            /// @return pointer to beginning of vector data
            T *begin() const { return &(this->data_[0]); };

            /// @brief Get pointer past the last element
            /// @return pointer to one past the last element
            T *end() const { return &(this->data_[this->count_]); };

            /// @brief Print vector elements to stdout with line breaks
            void print() const;
        };

        /// @brief Implementation: Allocate vector storage without initialization
        template <typename T, class Allocator>
        Vector<T, Allocator>::Vector(size_type count, const alloc_ptr &alloc) : Tensor<T, Allocator>{count, alloc} {};

        /// @brief Implementation: Allocate and fill vector with uniform value
        template <typename T, class Allocator>
        Vector<T, Allocator>::Vector(size_type count, const T &value, const alloc_ptr &alloc) : Vector{count, alloc}
        {
            std::fill(this->data_, this->data_ + count, value);
        };

        /// @brief Implementation: Wrap existing data pointer with optional ownership
        template <typename T, class Allocator>
        Vector<T, Allocator>::Vector(size_type count, T *ptr, bool take_ownership, const alloc_ptr &alloc) : Tensor<T, Allocator>{}
        {
            this->data_ = ptr;
            this->is_owner_ = take_ownership;
            this->count_ = count;
        };

        /// @brief Implementation: Copy from const data pointer
        template <typename T, class Allocator>
        Vector<T, Allocator>::Vector(size_type count, const T *ptr, const alloc_ptr &alloc) : Vector{count, alloc}
        {
            std::copy(ptr, ptr + count, this->data_);
        };

        /// @brief Copy constructor implementation
        template <typename T, class Allocator>
        Vector<T, Allocator>::Vector(const Vector &x) : Tensor<T, Allocator>{x}
        {
        };

        template <typename T, class Allocator>
        Vector<T, Allocator>::Vector(Vector &&x) noexcept : Tensor<T, Allocator>{std::move(x)}
        {
        };

        template <class T, class Allocator>
        Vector<T, Allocator>::Vector(std::initializer_list<T> init, const alloc_ptr &alloc)
            : Vector_<T>(), Tensor<T, Allocator>(init.size(), alloc)
        {
            std::copy(init.begin(), init.end(), this->data_);
        }

        /// @brief Copy assignment operator implementation
        template <typename T, class Allocator>
        Vector<T, Allocator> &Vector<T, Allocator>::operator=(const Vector &other)
        {
            if (this != &other)
            {
                Tensor<T, Allocator>::operator=(other);
            }
            return *this;
        };

        /// @brief Move assignment operator implementation
        template <typename T, class Allocator>
        Vector<T, Allocator> &Vector<T, Allocator>::operator=(Vector &&other)
        {
            if (this != &other)
            {
                Tensor<T, Allocator>::operator=(std::move(other));
            }
            return *this;
        };

        /// @brief Destructor implementation
        template <typename T, class Allocator>
        Vector<T, Allocator>::~Vector() {

        };

        /// @brief Implementation: Print vector elements to stdout with line breaks
        template <typename T, class Allocator>
        void Vector<T, Allocator>::print() const
        {
            for (size_t i = 0; i < this->size(); i++)
            {
                std::cout << this->data()[i] << ", ";
                std::cout << std::endl;
            }
        }

    } // namespace cpu
} // namespace lahva