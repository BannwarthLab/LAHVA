/// @file allocators.hpp
/// @brief Memory allocator abstractions for CPU and GPU tensor storage.
///
/// Provides abstract allocator classes and concrete implementations for managing tensor memory
/// across CPU and GPU devices. Supports custom allocation strategies and memory pooling.

#pragma once
#include <cstddef>
#ifdef _CUDA
#include "runtime.hpp"
#endif

namespace lahva
{
    /// @brief Abstract base class for CPU memory allocators
    ///
    /// Provides interface for managing host (CPU) memory allocation and deallocation.
    ///
    /// @tparam T data type of elements to allocate
    template <typename T>
    class CPUAllocator
    {
        using value_type = T;

    public:
        /// @brief Allocate n elements of type T on CPU
        /// @param[in] n number of elements to allocate
        /// @return raw pointer to allocated memory (caller manages deallocation)
        /// @note Allocation may fail silently on some allocators; always check for null
        virtual value_type *
        allocate(std::size_t n) const = 0;

        /// @brief Deallocate previously allocated memory with known size
        /// @param[in] ptr raw pointer to memory to deallocate (must be valid)
        /// @param[in] n number of elements originally allocated (may be used for pooling)
        /// @note Size parameter allows allocators to implement memory pooling strategies
        virtual void
        deallocate(value_type *ptr, std::size_t n) noexcept = 0;

        /// @brief Deallocate previously allocated memory
        /// @param[in] ptr raw pointer to memory to deallocate (must be valid)
        /// @note This version is used when original size is not available
        virtual void
        deallocate(value_type *ptr) noexcept
            = 0;
    };

    /// @brief Standard C++ allocator using new/delete operators
    ///
    /// Simple allocator that uses global ::operator new and ::operator delete
    /// for memory management. This is the default allocator for CPU tensors
    /// when CUDA is not available or not required.
    ///
    /// @tparam T element data type
    template <typename T>
    class StdAllocator : public CPUAllocator<T>
    {
        using value_type = T;

    public:
        /// @brief Default constructor
        StdAllocator() noexcept {};

        /// @brief Converting constructor from StdAllocator of different type
        template <class U>
        StdAllocator(StdAllocator<U> const &) : StdAllocator<T>{} {}

        /// @brief Converting constructor from CPUAllocator of different type
        template <class U>
        StdAllocator(CPUAllocator<U> const &) : StdAllocator<T>{} {}

        /// @brief Allocate memory using ::operator new
        /// @param[in] n number of elements to allocate
        /// @return raw pointer to n*sizeof(T) bytes of uninitialized memory
        value_type *
        allocate(std::size_t n) const override
        {
            return static_cast<T*>(
            ::operator new(n * sizeof(T)));
        }

        /// @brief Deallocate memory using ::operator delete (size parameter ignored)
        /// @param[in] ptr raw pointer to memory to deallocate
        /// @param[in] n size parameter (unused, for API compatibility)
        void
        deallocate(value_type *ptr, std::size_t n) noexcept override
        {
            ::operator delete(ptr);
        }

        /// @brief Deallocate memory using ::operator delete
        /// @param[in] ptr raw pointer to memory to deallocate
        void
        deallocate(value_type *ptr) noexcept override
        {
            ::operator delete(ptr);
        }
    };
#ifdef _CUDA

    /// @brief Abstract base class for GPU memory allocators
    ///
    /// Extends CPUAllocator interface with GPU-specific functionality for device
    /// memory management and host-device data transfers. Concrete implementations
    /// provide synchronous or asynchronous allocation and transfer operations.
    ///
    /// @tparam T data type of elements to allocate
    template <typename T>
    class GPUAllocator_ : public CPUAllocator<T>
    {
    protected:
        /// @brief Currently selected CUDA device (-1 = uninitialized)
        mutable int device_ = -1;
    
    public:
        /// @brief Functor-style deletion operator for unique_ptr deleter
        /// @param[in] ptr pointer to device memory to delete (nullptr-safe)
        virtual void operator()(T *ptr) = 0;

        /// @brief Transfer data from host to GPU device memory
        /// @param[out] d_ptr device memory destination pointer
        /// @param[in] h_ptr host memory source pointer
        /// @param[in] buffersize number of bytes to transfer
        /// @note Behavior (sync/async) depends on concrete implementation
        virtual void H2DCopy(void *d_ptr,const void *h_ptr, const size_t buffersize) const = 0;

        /// @brief Transfer data from GPU device memory to host
        /// @param[in] d_ptr device memory source pointer
        /// @param[out] h_ptr host memory destination pointer
        /// @param[in] buffersize number of bytes to transfer
        /// @note Behavior (sync/async) depends on concrete implementation
        virtual void D2HCopy(void *d_ptr, void *h_ptr, const size_t buffersize) const = 0;

        /// @brief Set CUDA stream for asynchronous operations
        /// @param[in] stream shared_ptr to cudaStream_t for async ops
        /// @note Default implementation is empty for synchronous allocators
        virtual void setStream(std::shared_ptr<cudaStream_t>& stream) const {};

        /// @brief Set target CUDA device for subsequent operations
        /// @param[in] device CUDA device ID to use
        /// @note Must be called before allocate/deallocate on multi-GPU systems
        virtual void setDevice(int device) const {this->device_ = device;};

    };

    /// @brief Pinned host memory allocator for CUDA operations
    ///
    /// Allocates host memory using cudaMallocHost, which pins memory pages
    /// (prevents paging to disk). Pinned memory enables faster DMA transfers
    /// between host and GPU.
    ///
    /// @tparam T element data type
    template <typename T>
    class CudaHostAllocator : public CPUAllocator<T>
    {
    public:
        using value_type = T;

        /// @brief Default constructor
        CudaHostAllocator() noexcept {}

        /// @brief Converting constructor from CudaHostAllocator of different type
        template <class U>
        CudaHostAllocator(CudaHostAllocator<U> const &) : CudaHostAllocator<T>{} {}

        /// @brief Converting constructor from CPUAllocator of different type
        template <class U>
        CudaHostAllocator(CPUAllocator<U> const &) : CudaHostAllocator<T>{} {}

        /// @brief Allocate n*sizeof(T) bytes of pinned host memory
        /// @param[in] n number of elements to allocate
        /// @return raw pointer to pinned memory, backed by CUDA
        /// @throws May throw via get_cuda_error if allocation fails
        /// @note Memory is page-locked and suitable for DMA transfers
        value_type *
        allocate(std::size_t n) const override
        {
            value_type *ptr;
            get_cuda_error(cudaMallocHost(&ptr, n * sizeof(value_type)));
            return ptr;
        }

        /// @brief Free pinned host memory allocated via cudaMallocHost
        /// @param[in] ptr raw pointer to pinned memory (must be from allocate())
        /// @param[in] n size parameter (unused, for API compatibility)
        void
        deallocate(value_type *ptr, std::size_t n) noexcept override
        {
            get_cuda_error(cudaFreeHost(ptr));
        }

        /// @brief Free pinned host memory allocated via cudaMallocHost
        /// @param[in] ptr raw pointer to pinned memory (must be from allocate())
        void
        deallocate(value_type *ptr) noexcept override
        {
            get_cuda_error(cudaFreeHost(ptr));
        }
    };

    /// @brief Zero-copy mapped host memory allocator for CUDA operations
    ///
    /// Allocates host memory with cudaHostAllocMapped flag, making it directly
    /// accessible from GPU without explicit transfers. GPU accesses memory
    /// across PCIe/NVLink transparently via memory mapping.
    ///
    /// @tparam T element data type
    template <typename T>
    class CudaHostAllocatorMapped : public CPUAllocator<T>
    {
    public:
        using value_type = T;

        /// @brief Default constructor
        CudaHostAllocatorMapped() noexcept {}

        /// @brief Converting constructor from CudaHostAllocator of different type
        template <class U>
        CudaHostAllocatorMapped(CudaHostAllocator<U> const &) : CudaHostAllocator<T>{} {}

        /// @brief Converting constructor from CPUAllocator of different type
        template <class U>
        CudaHostAllocatorMapped(CPUAllocator<U> const &) : CudaHostAllocator<T>{} {}

        /// @brief Allocate n*sizeof(T) bytes of mapped host memory
        /// @param[in] n number of elements to allocate
        /// @return raw pointer to mapped host memory (GPU-accessible)
        /// @throws May throw via get_cuda_error if allocation fails
        /// @note GPU can directly access this memory without explicit copies
        value_type *
        allocate(std::size_t n) const override
        {
            value_type *ptr;
            get_cuda_error(cudaHostAlloc(&ptr, n * sizeof(value_type), cudaHostAllocMapped));
            return ptr;
        }

        /// @brief Free mapped host memory allocated via cudaHostAlloc
        /// @param[in] ptr raw pointer to mapped memory (must be from allocate())
        /// @param[in] n size parameter (unused, for API compatibility)
        void
        deallocate(value_type *ptr, std::size_t n) noexcept override
        {
            get_cuda_error(cudaFreeHost(ptr));
        }

        /// @brief Free mapped host memory allocated via cudaHostAlloc
        /// @param[in] ptr raw pointer to mapped memory (must be from allocate())
        void
        deallocate(value_type *ptr) noexcept override
        {
            get_cuda_error(cudaFreeHost(ptr));
        }
    };

    /// @brief Synchronous GPU device memory allocator with blocking transfers
    ///
    /// Allocates memory on GPU device using cudaMalloc with synchronous
    /// memory transfers via cudaMemcpy. All operations block until complete.
    ///
    /// @tparam T element data type
    template <typename T>
    class CudaDeviceAllocator : public GPUAllocator_<T>
    {
    public:
        using value_type = T;

        /// @brief Default constructor
        CudaDeviceAllocator() noexcept {}

        /// @brief Converting constructor from CudaDeviceAllocator of different type
        template <class U>
        CudaDeviceAllocator(CudaDeviceAllocator<U> const &) : CudaDeviceAllocator<T>{} {}

        /// @brief Converting constructor from GPUAllocator_ of different type
        template <class U>
        CudaDeviceAllocator(GPUAllocator_<U> const &) : CudaDeviceAllocator<T>{} {}

        /// @brief Allocate n*sizeof(T) bytes on GPU device via cudaMalloc
        /// @param[in] n number of elements to allocate
        /// @return raw pointer to GPU device memory
        /// @throws May throw via get_cuda_error if allocation fails or device invalid
        /// @note Sets device before allocation if device_ is set
        value_type *
        allocate(std::size_t n) const override
        {
            value_type *ptr;
            get_cuda_error(cudaSetDevice(this->device_));
            get_cuda_error(cudaMalloc((void **)&ptr, n * sizeof(value_type)));
            return ptr;
        }

        /// @brief Free GPU device memory via cudaFree
        /// @param[in] ptr raw pointer to GPU memory (must be from allocate())
        /// @param[in] n size parameter (unused, for API compatibility)
        /// @note Synchronously frees memory and returns when complete
        void
        deallocate(value_type *ptr, std::size_t n) noexcept override
        {
            get_cuda_error(cudaSetDevice(this->device_));
            get_cuda_error(cudaFree(ptr));
        }

        /// @brief Free GPU device memory via cudaFree
        /// @param[in] ptr raw pointer to GPU memory (must be from allocate())
        void
        deallocate(value_type *ptr) noexcept override
        {
            get_cuda_error(cudaSetDevice(this->device_));
            get_cuda_error(cudaFree(ptr));
        }

        /// @brief Deletion operator for use as unique_ptr deleter
        /// @param[in] ptr pointer to GPU memory to delete
        void operator()(T *ptr) override
        {
            if (ptr != nullptr)
            {
                this->deallocate(ptr);
            }
        }

        /// @brief Synchronous host-to-device memory transfer via cudaMemcpy
        /// @param[out] d_ptr destination GPU device memory pointer
        /// @param[in] h_ptr source host memory pointer (must be accessible to CPU)
        /// @param[in] buffersize bytes to transfer
        /// @note Blocks until transfer completes; GPU cannot start work until done
        void H2DCopy(void *d_ptr,const void *h_ptr, const size_t buffersize) const override
        {
            get_cuda_error(cudaSetDevice(this->device_));
            get_cuda_error(cudaMemcpy(d_ptr, h_ptr, buffersize, cudaMemcpyHostToDevice));
        };

        /// @brief Synchronous device-to-host memory transfer via cudaMemcpy
        /// @param[in] d_ptr source GPU device memory pointer
        /// @param[out] h_ptr destination host memory pointer
        /// @param[in] buffersize bytes to transfer
        /// @note Blocks until transfer completes; CPU cannot use data until done
        void D2HCopy(void *d_ptr, void *h_ptr, const size_t buffersize) const override
        {
            get_cuda_error(cudaSetDevice(this->device_));
            get_cuda_error(cudaMemcpy(h_ptr, d_ptr, buffersize, cudaMemcpyDeviceToHost));
        };
    };

    /// @brief Asynchronous GPU device memory allocator with non-blocking transfers
    ///
    /// Allocates GPU memory using cudaMallocAsync and performs non-blocking
    /// memory transfers via cudaMemcpyAsync. Supports CUDA streams for
    /// pipelining computation and communication.
    ///
    /// @tparam T element data type
    template <typename T>
    class CudaDeviceAsyncAllocator : public GPUAllocator_<T>
    {
    public:
        using value_type = T;

        /// @brief CUDA stream for asynchronous operations (nullptr = default stream)
        mutable std::shared_ptr<cudaStream_t> stream_ ;

        /// @brief Default constructor (no stream, falls back to synchronous)
        CudaDeviceAsyncAllocator() noexcept {}

        /// @brief Constructor with CUDA stream for asynchronous operations
        /// @param[in] stream shared_ptr to cudaStream_t for async H2D/D2H
        CudaDeviceAsyncAllocator( std::shared_ptr<cudaStream_t>& stream) : stream_{stream} {};

        /// @brief Converting constructor from CudaDeviceAsyncAllocator of different type
        template <class U>
        CudaDeviceAsyncAllocator(CudaDeviceAsyncAllocator<U> const &) : CudaDeviceAsyncAllocator<T>{} {};

        /// @brief Converting constructor from GPUAllocator_ of different type
        template <class U>
        CudaDeviceAsyncAllocator(GPUAllocator_<U> const &) : CudaDeviceAsyncAllocator<T>{} {};

        /// @brief Allocate n*sizeof(T) bytes on GPU device (async or sync)
        /// @param[in] n number of elements to allocate
        /// @return raw pointer to GPU device memory
        /// @throws May throw via get_cuda_error if allocation fails
        /// @note Uses cudaMallocAsync if stream set, cudaMalloc otherwise
        value_type *
        allocate(std::size_t n) const override
        {
            value_type *ptr;
            if (stream_)
            {
                get_cuda_error(cudaSetDevice(this->device_));
                get_cuda_error(cudaMallocAsync((void **)&ptr, n * sizeof(value_type), *stream_));
            }
            else
            {
                get_cuda_error(cudaSetDevice(this->device_));
                get_cuda_error(cudaMalloc((void **)&ptr, n * sizeof(value_type)));
            }

            return ptr;
        }

        /// @brief Free GPU device memory (async or sync)
        /// @param[in] ptr raw pointer to GPU memory (must be from allocate())
        /// @param[in] n size parameter (unused, for API compatibility)
        /// @note Uses cudaFreeAsync if stream set, cudaFree otherwise
        void
        deallocate(value_type *ptr, std::size_t n) noexcept override
        {
            if (stream_)
            {
                get_cuda_error(cudaSetDevice(this->device_));
                get_cuda_error(cudaFreeAsync(ptr, *stream_));
            }
            else
            {
                get_cuda_error(cudaSetDevice(this->device_));
                get_cuda_error(cudaFree(ptr));
            }

        }

        /// @brief Free GPU device memory (async or sync)
        /// @param[in] ptr raw pointer to GPU memory (must be from allocate())
        void
        deallocate(value_type *ptr) noexcept override
        {
            if (stream_)
            {
                get_cuda_error(cudaSetDevice(this->device_));
                get_cuda_error(cudaFreeAsync(ptr, *stream_));
            }
            else
            {
                get_cuda_error(cudaSetDevice(this->device_));
                get_cuda_error(cudaFree(ptr));
            }
        }

        /// @brief Deletion operator for use as unique_ptr deleter
        /// @param[in] ptr pointer to GPU memory to delete
        void operator()(T *ptr) override
        {
            if (ptr != nullptr)
            {
                this->deallocate(ptr);
            }
        }

        /// @brief Asynchronous host-to-device transfer (or synchronous if no stream)
        /// @param[out] d_ptr destination GPU device memory pointer
        /// @param[in] h_ptr source host memory pointer (should be pinned for speed)
        /// @param[in] buffersize bytes to transfer
        /// @note Uses cudaMemcpyAsync if stream set, cudaMemcpy otherwise
        /// @warning Host memory should be pinned (CudaHostAllocator) for performance
        void H2DCopy(void *d_ptr, const void *h_ptr, const size_t buffersize) const override
        {
            if (stream_ )
            {
                get_cuda_error(cudaSetDevice(this->device_));
                get_cuda_error(cudaMemcpyAsync(d_ptr, h_ptr, buffersize, cudaMemcpyHostToDevice, *stream_));
            }
            else
            {
                get_cuda_error(cudaSetDevice(this->device_));
                get_cuda_error(cudaMemcpy(d_ptr, h_ptr, buffersize, cudaMemcpyHostToDevice));
            }

        };

        /// @brief Asynchronous device-to-host transfer (or synchronous if no stream)
        /// @param[in] d_ptr source GPU device memory pointer
        /// @param[out] h_ptr destination host memory pointer
        /// @param[in] buffersize bytes to transfer
        /// @note Uses cudaMemcpyAsync if stream set, cudaMemcpy otherwise
        /// @warning Do not access h_ptr on CPU until stream synchronizes
        void D2HCopy(void *d_ptr, void *h_ptr, const size_t buffersize) const override
        {
            if (stream_)
            {
                get_cuda_error(cudaSetDevice(this->device_));
                get_cuda_error(cudaMemcpyAsync(h_ptr, d_ptr, buffersize, cudaMemcpyDeviceToHost, *stream_));
            }
            else
            {
                get_cuda_error(cudaSetDevice(this->device_));
                get_cuda_error(cudaMemcpy(h_ptr, d_ptr, buffersize, cudaMemcpyDeviceToHost));
            }
        };

        /// @brief Set CUDA stream for subsequent asynchronous operations
        /// @param[in] stream shared_ptr to cudaStream_t (can be nullptr for sync fallback)
        /// @note Setting stream enables cudaMallocAsync/cudaMemcpyAsync
        void setStream(std::shared_ptr<cudaStream_t>& stream)  const override {
             stream_ = stream;};
    };
#endif
} // namespace lahva
