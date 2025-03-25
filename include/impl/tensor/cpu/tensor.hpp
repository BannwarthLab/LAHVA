#ifndef LAHVA_TENSOR_HPP
#define LAHVA_TENSOR_HPP

#include <memory>
#include <iterator>
#include <iostream>
#include "impl/tensor/allocators.hpp"
namespace lahva
{

    /// @brief Abstract base class for tensor
    /// @tparam T type of values to be stored in tensor
    template <typename T>
    class Tensor
    {
    public:
        /// @brief return size of Tensor
        /// @return size of values
        virtual size_t size() = 0;
        /// @brief return pointer to the underlying data
        /// @return ptr to begin of buffer
        virtual T *data() = 0;

        /// @brief return size of Tensor
        /// @return size of values
        virtual size_t size() const = 0;
        /// @brief return pointer to the underlying data
        /// @return ptr to begin of buffer
        virtual T *data() const = 0;

        T sum() const
        {
            T res = (T)0;
#pragma omp parallel for reduction(+ : res)
            for (size_t i = 0; i < this->size(); i++)
            {
                res += this->data()[i];
            }
            return res;
        }

        T sum()
        {
            T res = (T)0;
#pragma omp parallel for reduction(+ : res)
            for (size_t i = 0; i < this->size(); i++)
            {
                res += this->data()[i];
            }
            return res;
        }
        template <typename D>
        const T &operator[](D index) const
        {
            return this->data()[static_cast<size_t>(index)];
        };
        template <typename D>
        T &operator[](D index)
        {
            return this->data()[static_cast<size_t>(index)];
        };

       // virtual const CPUAllocator<T> get_allocator() const = 0;

    };

    template <typename T, typename Allocator = StdAllocator<T>>
    class CPUTensor : virtual public Tensor<T>
    {
        using alloc_ptr = CPUAllocator<T>;

    protected:
        std::size_t count_ = 0;
        Allocator alloc_;
        std::shared_ptr<alloc_ptr> rebind_;
        T *data_ = nullptr;
        bool is_owner_ = true;
        bool no_alloc = false;

    public:
        CPUTensor(size_t count, const alloc_ptr &alloc = Allocator()) : count_{count}, alloc_{alloc}, data_{alloc.allocate(count)}
        {
            
         };
        CPUTensor(const alloc_ptr &alloc = Allocator()) : alloc_{alloc} { is_owner_ = false; };
        CPUTensor(const CPUTensor &other) : CPUTensor{other.count_, other.get_allocator()}
        {
            if (!this->no_alloc) std::copy(other.data_, other.data_ + count_, this->data_);
            
        };

        CPUTensor(CPUTensor &&other) : count_{other.count_}, alloc_{other.alloc_}
        {

            this->data_ = other.data_;
            this->is_owner_ = other.is_owner_;
            other.is_owner_ = false;
            other.data_ = nullptr;
            other.count_ = 0;
        };

        virtual ~CPUTensor()
        {
            if (is_owner_)
            {
                alloc_.deallocate(data_, count_);
            }
        }

        T *data() override { return data_; };
        T *data() const override { return data_; };
        size_t size() const override { return count_; };
        size_t size() override { return count_; };

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

        const Allocator get_allocator() const { return alloc_; };
    };

} // namespace lahva

#endif