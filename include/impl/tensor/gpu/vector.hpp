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
        
    template<typename T>
    class Vector_ : public virtual GPUTensor_<T>, public virtual cpu::Vector_<T>
    {

    };
    
    template <class T, class Allocator = CudaHostAllocator<T>, class GPUAllocator = CudaDeviceAllocator<T>>
    class Vector : public GPUTensor<T, Allocator, GPUAllocator>, public virtual Vector_<T>
    {

    public:
        using size_type = std::size_t;
        using alloc_ptr = CPUAllocator<T>;
        using gpualloc_ptr = GPUAllocator;

        Vector() : GPUTensor<T, Allocator, GPUAllocator>() {};
        Vector(size_type count, const alloc_ptr &alloc = Allocator(), const gpualloc_ptr &gpualloc = GPUAllocator());
        template<typename U, typename V>
        Vector(size_type count, const CPUAllocator<U> &alloc = Allocator(), const GPUAllocator_<V> &gpualloc = GPUAllocator())
        : GPUTensor<T, Allocator, GPUAllocator>{count, static_cast<Allocator>(alloc), static_cast<GPUAllocator>(gpualloc)}  {};

        Vector(size_type count, const CudaRuntime &cudart, const gpualloc_ptr  &gpualloc = GPUAllocator());
        template<typename V>
        Vector(size_type count, const CudaRuntime &cudart, const GPUAllocator_<V>  &gpualloc = GPUAllocator())
        : GPUTensor<T, Allocator, GPUAllocator>{gpualloc} { std::cout << "Created here << std::endl";};
        Vector(const Vector &x);
        Vector(Vector &&x);
        Vector(size_type count, const T &value, const alloc_ptr &alloc = Allocator(), const gpualloc_ptr  &gpualloc = GPUAllocator());
        Vector(size_type count, T *ptr, bool take_onwership = true, const alloc_ptr &alloc = Allocator(), const gpualloc_ptr  &gpualloc = GPUAllocator());
        Vector(size_type count, const T *ptr, const alloc_ptr &alloc = Allocator(), const gpualloc_ptr  &gpualloc = GPUAllocator());
        Vector(std::initializer_list<T> init, const alloc_ptr &alloc = Allocator(), const gpualloc_ptr &gpualloc = GPUAllocator());
        ~Vector();
        Vector &operator=(const Vector &other);
        Vector &operator=(Vector &&other);

        T *begin() const { return &(this->data_[0]); };
        T *end() const { return &(this->data_[this->count_]); };

        void print() const;
    };

    
    ///////////////////////////////////////////////////////////////////////////
    // Vector class
    ///////////////////////////////////////////////////////////////////////////
    template <typename T, class Allocator, class GPUAllocator>
    Vector<T, Allocator, GPUAllocator>::Vector(size_type count, const alloc_ptr& alloc , const gpualloc_ptr& gpualloc ) : 
    GPUTensor<T, Allocator, GPUAllocator>{count, alloc, gpualloc} {};
    
    template <typename T, class Allocator, class GPUAllocator>
    Vector<T, Allocator, GPUAllocator>::Vector(size_type count, const CudaRuntime &cudart, const gpualloc_ptr& gpualloc ) : 
    GPUTensor<T, Allocator, GPUAllocator>{gpualloc}
    {
        this->gpu_buffer = true;
        this->count_ = count;
        this->is_owner_ = false;
        this->is_on_device_ = true;
    };

    template <typename T, class Allocator, class GPUAllocator>
    Vector<T, Allocator, GPUAllocator>::Vector(size_type count, const T &value, const alloc_ptr& alloc , const gpualloc_ptr& gpualloc ) : 
    Vector{count, alloc, gpualloc}
    {
        std::fill(this->data_, this->data_ + count, value);
    };

    template <typename T, class Allocator, class GPUAllocator>
    Vector<T, Allocator, GPUAllocator>::Vector(size_type count, T *ptr, bool take_ownership, const alloc_ptr& alloc , const gpualloc_ptr& gpualloc ) : 
    GPUTensor<T, Allocator, GPUAllocator>{gpualloc}
    {   
        this->data_ = ptr;
        this->count_ = count;
        this->is_owner_ = take_ownership;
    };

    template <typename T, class Allocator, class GPUAllocator>
    Vector<T, Allocator, GPUAllocator>::Vector(size_type count, const T *ptr, const alloc_ptr& alloc , const gpualloc_ptr& gpualloc ) : 
    Vector{count, alloc, gpualloc}
    {
        std::copy(ptr, ptr + count, this->data_);
    };

    template <class T, class Allocator, class GPUAllocator>
Vector<T, Allocator, GPUAllocator>::Vector(std::initializer_list<T> init, const alloc_ptr &alloc, const gpualloc_ptr &gpualloc)
    : GPUTensor<T, Allocator, GPUAllocator>(init.size(), alloc, gpualloc), Vector_<T>()
{
    std::copy(init.begin(), init.end(), this->data_);
}

    template <typename T, class Allocator, class GPUAllocator>
    Vector<T, Allocator, GPUAllocator>::Vector(const Vector &x) : 
    GPUTensor<T, Allocator, GPUAllocator>{static_cast<const GPUTensor<T, Allocator, GPUAllocator>&>(x)}
    {
    };

    template <typename T, class Allocator, class GPUAllocator>
    Vector<T, Allocator, GPUAllocator>::Vector(Vector &&x) : 
    GPUTensor<T, Allocator, GPUAllocator>{static_cast<GPUTensor<T, Allocator, GPUAllocator>&&>(x)}
    {
    };

    template <typename T, class Allocator, class GPUAllocator>
    Vector<T, Allocator, GPUAllocator> &Vector<T,Allocator, GPUAllocator>::operator=(const Vector &other)
    {
        if (this != &other)
        {
            GPUTensor<T, Allocator, GPUAllocator>::operator=(static_cast<const GPUTensor<T, Allocator, GPUAllocator>&>(other));
        }
        return *this;
    };

    template <typename T, class Allocator, class GPUAllocator>
    Vector<T, Allocator, GPUAllocator> &Vector<T, Allocator, GPUAllocator>::operator=(Vector &&other) 
    {
        if (this != &other)
        {
            GPUTensor<T, Allocator, GPUAllocator>::operator=(std::move(other));
        }
        return *this;
    };

    template <typename T, class Allocator, class GPUAllocator>
    Vector<T, Allocator, GPUAllocator>::~Vector()
    {
        
    };

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