#pragma once
#ifdef _CUDA
#include "runtime.hpp"
#endif

namespace lahva
{
    template <typename T>
    class CPUAllocator
    {
        using value_type = T;

    public:
        virtual value_type * // Use pointer if pointer is not a value_type*
        allocate(std::size_t n) const = 0;

        virtual void
        deallocate(value_type *ptr, std::size_t n) noexcept = 0; // Use pointer if pointer is not a value_type*

        virtual void
        deallocate(value_type *ptr) noexcept // Use pointer if pointer is not a value_type*
            = 0;
    };

    template <typename T>
    class StdAllocator : public CPUAllocator<T>
    {
        using value_type = T;

    public:
        StdAllocator() noexcept {};

        template <class U>
        StdAllocator(StdAllocator<U> const &) : StdAllocator<T>{} {}
        template <class U>
        StdAllocator(CPUAllocator<U> const &) : StdAllocator<T>{} {}
        value_type * // Use pointer if pointer is not a value_type*
        allocate(std::size_t n) const override 
        {
            return static_cast<T*>(
            ::operator new(n * sizeof(T)));
        }

        void
        deallocate(value_type *ptr, std::size_t n) noexcept override // Use pointer if pointer is not a value_type*
        {
            ::operator delete(ptr);
        }

        void
        deallocate(value_type *ptr) noexcept override // Use pointer if pointer is not a value_type*
        {
            ::operator delete(ptr);
        }
    };
#ifdef _CUDA

    template <typename T>
    class GPUAllocator_ : public CPUAllocator<T>
    {
    public:
        virtual void operator()(T *ptr) = 0;
        virtual void H2DCopy(void *d_ptr,const void *h_ptr, const size_t buffersize) const = 0;
        virtual void D2HCopy(void *d_ptr, void *h_ptr, const size_t buffersize) const = 0;
        virtual void setStream(cudaStream_t stream) const {};
    };

    template <typename T>
    class CudaHostAllocator : public CPUAllocator<T>
    {
    public:
        using value_type = T;
        CudaHostAllocator() noexcept {} // not required, unless used
        template <class U>
        CudaHostAllocator(CudaHostAllocator<U> const &) : CudaHostAllocator<T>{} {}
        template <class U>
        CudaHostAllocator(CPUAllocator<U> const &) : CudaHostAllocator<T>{} {}
        value_type * // Use pointer if pointer is not a value_type*
        allocate(std::size_t n) const override  
        {
            value_type *ptr;
            get_cuda_error(cudaMallocHost(&ptr, n * sizeof(value_type)));
            return ptr;
        }

        void
        deallocate(value_type *ptr, std::size_t n) noexcept // Use pointer if pointer is not a value_type*
        {
            get_cuda_error(cudaFreeHost(ptr));
        }

        void
        deallocate(value_type *ptr) noexcept // Use pointer if pointer is not a value_type*
        {
            get_cuda_error(cudaFreeHost(ptr));
        }
    };

    template <typename T>
    class CudaDeviceAllocator : public GPUAllocator_<T>
    {
    public:
        using value_type = T;
        CudaDeviceAllocator() noexcept {} // not required, unless used
        template <class U>
        CudaDeviceAllocator(CudaDeviceAllocator<U> const &) : CudaDeviceAllocator<T>{} {}
        template <class U>
        CudaDeviceAllocator(GPUAllocator_<U> const &) : CudaDeviceAllocator<T>{} {}

        value_type * // Use pointer if pointer is not a value_type*
        allocate(std::size_t n) const override
        {
            value_type *ptr;
            get_cuda_error(cudaMalloc((void **)&ptr, n * sizeof(value_type)));
            return ptr;
        }
        void
        deallocate(value_type *ptr, std::size_t n) noexcept // Use pointer if pointer is not a value_type*
        {
            get_cuda_error(cudaFree(ptr));
        }

        void
        deallocate(value_type *ptr) noexcept // Use pointer if pointer is not a value_type*
        {
            get_cuda_error(cudaFree(ptr));
        }

        void operator()(T *ptr)
        {
            if (ptr != nullptr)
            {
                this->deallocate(ptr);
            }
        }

        void H2DCopy(void *d_ptr,const void *h_ptr, const size_t buffersize) const
        {
            get_cuda_error(cudaMemcpy(d_ptr, h_ptr, buffersize, cudaMemcpyHostToDevice));
        };

        void D2HCopy(void *d_ptr, void *h_ptr, const size_t buffersize) const
        {
            get_cuda_error(cudaMemcpy(h_ptr, d_ptr, buffersize, cudaMemcpyDeviceToHost));
        };
    };

    template <typename T>
    class CudaDeviceAsyncAllocator : public GPUAllocator_<T>
    {
    public:
        using value_type = T;

        mutable cudaStream_t stream_ = 0;

        CudaDeviceAsyncAllocator() noexcept {} // not required, unless used
        CudaDeviceAsyncAllocator(const cudaStream_t stream) : stream_{stream} {};
        template <class U>
        CudaDeviceAsyncAllocator(CudaDeviceAsyncAllocator<U> const &) : CudaDeviceAsyncAllocator<T>{} {std::cout << "Calling the copy constrcut" << std::endl;}
        template <class U>
        CudaDeviceAsyncAllocator(GPUAllocator_<U> const &) : CudaDeviceAsyncAllocator<T>{} {std::cout << "Calling the copy constrcut" << std::endl;}

        value_type * // Use pointer if pointer is not a value_type*
        allocate(std::size_t n) const override 
        {
            value_type *ptr;
            get_cuda_error(cudaMallocAsync((void **)&ptr, n * sizeof(value_type), stream_));
            return ptr;
        }
        void
        deallocate(value_type *ptr, std::size_t n) noexcept // Use pointer if pointer is not a value_type*
        {
            get_cuda_error(cudaFreeAsync(ptr, stream_));
        }

        void
        deallocate(value_type *ptr) noexcept // Use pointer if pointer is not a value_type*
        {
            get_cuda_error(cudaFreeAsync(ptr, stream_));
        }

        void operator()(T *ptr)
        {
            if (ptr != nullptr)
            {
                this->deallocate(ptr);
            }
        }

        void H2DCopy(void *d_ptr, const void *h_ptr, const size_t buffersize) const
        {
            get_cuda_error(cudaMemcpyAsync(d_ptr, h_ptr, buffersize, cudaMemcpyHostToDevice, stream_));
        };

        void D2HCopy(void *d_ptr, void *h_ptr, const size_t buffersize) const
        {
            get_cuda_error(cudaMemcpyAsync(h_ptr, d_ptr, buffersize, cudaMemcpyDeviceToHost, stream_));
        };
        void setStream(cudaStream_t stream) const {stream_ = stream;};
    };
#endif
} // namespace lahva
