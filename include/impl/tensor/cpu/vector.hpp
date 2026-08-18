#pragma once
#include <memory>
#include "impl/tensor/cpu/tensor.hpp"
#include <initializer_list>
#include <omp.h>
namespace lahva
{
    namespace cpu
    {

        template <typename T>
        class Vector_ : public virtual Tensor<T>
        {
        };

        template <class T, class Allocator = StdAllocator<T>>
        class Vector : public CPUTensor<T, Allocator>, virtual public Vector_<T>
        {

        public:
            using size_type = std::size_t;
            using alloc_ptr = CPUAllocator<T>;
            Vector() { this->no_alloc = true; };
            Vector(size_type count, const alloc_ptr &alloc = Allocator());
            Vector(const Vector &x);
            Vector(Vector &&x) noexcept;
            Vector(size_type count, const T &value, const alloc_ptr &alloc = Allocator());

            Vector(size_type count, T *ptr, bool take_onwership = true, const alloc_ptr &alloc = Allocator());
            Vector(size_type count, const T *ptr, const alloc_ptr &alloc = Allocator());
            Vector(std::initializer_list<T> init, const alloc_ptr &alloc = Allocator());
            ~Vector();
            Vector &operator=(const Vector &other);
            Vector &operator=(Vector &&other);
            template <typename D>
            T &operator[](D index) const
            {
                return this->data_[static_cast<size_t>(index)];
            };
            template <typename D>
            T &operator[](D index)
            {
                return this->data_[static_cast<size_t>(index)];
            };

            T *begin() const { return &(this->data_[0]); };
            T *end() const { return &(this->data_[this->count_]); };

            void print() const;
        };

        ///////////////////////////////////////////////////////////////////////////
        // Vector class
        ///////////////////////////////////////////////////////////////////////////
        template <typename T, class Allocator>
        Vector<T, Allocator>::Vector(size_type count, const alloc_ptr &alloc) : CPUTensor<T, Allocator>{count, alloc} {};

        template <typename T, class Allocator>
        Vector<T, Allocator>::Vector(size_type count, const T &value, const alloc_ptr &alloc) : Vector{count, alloc}
        {
            std::fill(this->data_, this->data_ + count, value);
        };

        template <typename T, class Allocator>
        Vector<T, Allocator>::Vector(size_type count, T *ptr, bool take_ownership, const alloc_ptr &alloc) : CPUTensor<T, Allocator>{}
        {
            this->data_ = ptr;
            this->is_owner_ = take_ownership;
            this->count_ = count;
        };

        template <typename T, class Allocator>
        Vector<T, Allocator>::Vector(size_type count, const T *ptr, const alloc_ptr &alloc) : Vector{count, alloc}
        {
            std::copy(ptr, ptr + count, this->data_);
        };

        template <typename T, class Allocator>
        Vector<T, Allocator>::Vector(const Vector &x) : CPUTensor<T, Allocator>{x}
        {
        };

        template <typename T, class Allocator>
        Vector<T, Allocator>::Vector(Vector &&x) noexcept : CPUTensor<T, Allocator>{std::move(x)}
        {
        };

        template <class T, class Allocator>
        Vector<T, Allocator>::Vector(std::initializer_list<T> init, const alloc_ptr &alloc)
            : Vector_<T>(), CPUTensor<T, Allocator>(init.size(), alloc)
        {
            std::copy(init.begin(), init.end(), this->data_);
        }

        template <typename T, class Allocator>
        Vector<T, Allocator> &Vector<T, Allocator>::operator=(const Vector &other)
        {
            if (this != &other)
            {
                CPUTensor<T, Allocator>::operator=(other);
            }
            return *this;
        };

        template <typename T, class Allocator>
        Vector<T, Allocator> &Vector<T, Allocator>::operator=(Vector &&other)
        {
            if (this != &other)
            {
                CPUTensor<T, Allocator>::operator=(std::move(other));
            }
            return *this;
        };

        template <typename T, class Allocator>
        Vector<T, Allocator>::~Vector(){

        };

        template <typename T, class Allocator>
        void Vector<T, Allocator>::print() const
        {
            for (size_t i = 0; i < this->size(); i++)
            {
                std::cout << this->data()[i] << ", ";
                std::cout << std::endl;
            }
        }

    }
} // namespace lahva