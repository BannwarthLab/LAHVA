/// @file tensor.hpp
/// @brief Abstract base tensor class for multi-dimensional array data structures.
///
/// Defines the abstract tensor interface providing common functionality for all tensor types
/// (vectors, matrices, etc.). Includes memory management, shape information, and basic operations.

#pragma once

#include <iostream>
#include <iterator>
#include <memory>
#include <omp.h>

#include "impl/tensor/allocators.hpp"

namespace lahva
{

    /// @brief Abstract base class for tensor
    /// @tparam T type of values to be stored in tensor
    template <typename T>
    class Tensor
    {
    public:
        /// @brief Get number of elements in tensor
        /// @return tensor size
        virtual size_t size() = 0;

        /// @brief Get pointer to tensor data
        /// @return non-const pointer to tensor data
        virtual T *data() = 0;

        /// @brief Get number of elements in tensor
        /// @return tensor size
        virtual size_t size() const = 0;

        /// @brief Get pointer to tensor data
        /// @return const pointer to tensor data
        virtual T *data() const = 0;

        /// @brief Compute sum of all tensor elements using parallel reduction (const version)
        /// @return sum of all elements
        T sum() const
        {
            T res = (T)0;
            #pragma omp parallel shared(res)
            {
                T my_part = (T)0.0;
        
                #pragma omp for
                for (long i = 0; i < this->size(); i++)
                {
                    my_part += this->data()[i];
                }
        
                #pragma omp critical
                {
                    res += my_part;
                }
            }
            
            return res;
        }

        /// @brief Compute sum of all tensor elements using parallel reduction
        /// @return sum of all elements
        T sum()
        {
            T res = (T)0;
            #pragma omp parallel shared(res)
            {
                T my_part = (T)0.0;

                #pragma omp for
                for (long i = 0; i < static_cast<long>(this->size()); i++)
                {
                    my_part += this->data()[i];
                }

                #pragma omp critical
                {
                    res += my_part;
                }
            }
            return res;
        }

        /// @brief Const element access operator
        /// @tparam D index type
        /// @param[in] index element index
        /// @return const reference to element at index
        template <typename D>
        const T &operator[](D index) const
        {
            return this->data()[static_cast<size_t>(index)];
        };

        /// @brief Element access operator
        /// @tparam D index type
        /// @param[in] index element index
        /// @return reference to element at index
        template <typename D>
        T &operator[](D index)
        {
            return this->data()[static_cast<size_t>(index)];
        };

    };

    /// @brief CPU-based tensor with host memory management
    /// Base class for CPU tensors providing automatic memory management.
    ///
    /// @tparam T data type for tensor elements
    /// @tparam Allocator host (CPU) memory allocator type (default: StdAllocator)
    template <typename T, typename Allocator = StdAllocator<T>>
    class CPUTensor : virtual public Tensor<T>
    {
        using alloc_ptr = CPUAllocator<T>;

    protected:
        /// @brief Number of elements in the tensor
        std::size_t count_ = 0;

        /// @brief Memory allocator instance
        Allocator alloc_;

        /// @brief Rebound allocator for memory management
        std::shared_ptr<alloc_ptr> rebind_;

        /// @brief Pointer to tensor data
        T *data_ = nullptr;

        /// @brief Flag indicating if this object owns the data
        bool is_owner_ = true;

        /// @brief Flag indicating if memory should not be allocated
        bool no_alloc = false;

    public:
        /// @brief Construct CPU tensor with specified element count
        /// @param[in] count number of elements in the tensor
        /// @param[in] alloc host (CPU) memory allocator
        CPUTensor(size_t count, const alloc_ptr &alloc = Allocator()) : count_{count}, alloc_{alloc}
        {
            if (count_ > 0)
            {
                is_owner_ = true;
                data_ = alloc_.allocate(count_);
            }
            else
            {
                is_owner_ = false;
                data_ = nullptr;
            }
        };

        /// @brief Construct CPU tensor without allocating initial memory
        /// @param[in] alloc host (CPU) memory allocator
        CPUTensor(const alloc_ptr &alloc) : alloc_{alloc} { is_owner_ = false; };

        /// @brief Default constructor for CPU tensor
        CPUTensor() : count_{0}, data_{nullptr}, no_alloc{true}
        {

        };

        /// @brief Copy constructor for CPU tensor
        /// @param[in] other source tensor to copy
        CPUTensor(const CPUTensor &other) :  no_alloc{other.no_alloc}
        {
            this->count_ = other.count_;
            this->alloc_ = other.get_allocator();
            if (!this->no_alloc)
            {
                this->data_ = this->alloc_.allocate(other.size());
                this->is_owner_ = true;
                std::copy(other.data_, other.data_ + count_, this->data_);
            }
            else
            {
                this->data_ = nullptr;
            }
            
        };

        /// @brief Move constructor for CPU tensor
        /// @param[in] other source tensor to move from
        CPUTensor(CPUTensor &&other) : count_{other.count_}, alloc_{other.alloc_}
        {

            this->data_ = other.data_;
            this->is_owner_ = other.is_owner_;
            other.is_owner_ = false;
            other.data_ = nullptr;
            other.count_ = 0;
        };

        /// @brief Destructor for CPU tensor, releases memory
        virtual ~CPUTensor()
        {
            if (is_owner_)
            {
                alloc_.deallocate(data_, count_);
            }
        }

        /// @brief Get pointer to tensor data
        /// @return non-const pointer to tensor data
        T *data() override { return data_; };

        /// @brief Get pointer to tensor data
        /// @return const pointer to tensor data
        T *data() const override { return data_; };

        /// @brief Get number of elements in tensor
        /// @return tensor size
        size_t size() const override { return count_; };

        /// @brief Get number of elements in tensor
        /// @return tensor size
        size_t size() override { return count_; };

        /// @brief Copy assignment operator
        /// @param[in] other source tensor
        /// @return reference to this tensor
        CPUTensor<T, Allocator> &operator=(const CPUTensor<T, Allocator> &other)
        {
            if (this != &other)
            {
                if (this->is_owner_ && this->data_ != nullptr)
                    this->alloc_.deallocate(this->data_, this->count_);
                this->alloc_ = other.get_allocator();
                count_ = other.count_;
                this->data_ = this->alloc_.allocate(other.size());
                this->is_owner_ = true;
                std::copy(other.data(), other.data() + other.size(), this->data_);
            }
            return *this;
        };

        /// @brief Move assignment operator
        /// @param[in] other source tensor to move from
        /// @return reference to this tensor
        CPUTensor<T, Allocator> &operator=(CPUTensor &&other)
        {
            if (this != &other)
            {
                if (this->is_owner_ && this->data_ != nullptr)
                    this->alloc_.deallocate(this->data_, this->count_);
                this->alloc_ = other.get_allocator();
                this->data_ = other.data_;
                this->is_owner_ = other.is_owner_;
                count_ = other.count_;

                other.count_ = 0;
                other.data_ = nullptr;
                other.is_owner_ = false;
            }
            return *this;
        };

        /// @brief Get memory allocator
        /// @return allocator instance by reference
        const Allocator get_allocator() const { return alloc_; };
    };

} // namespace lahva
