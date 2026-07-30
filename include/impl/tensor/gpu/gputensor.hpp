/// @file gputensor.hpp
/// @brief GPU-based abstract tensor class for GPU device memory management.
///
/// Defines the Tensor_ abstract class providing GPU-specific tensor functionality.
/// Handles CUDA device memory allocation, transfers, and synchronization.
/// Base class for all specialized GPU tensor types (vectors, matrices, etc.).

#pragma once
#include <memory>

#include "impl/tensor/allocators.hpp"
#include "impl/tensor/cpu/tensor.hpp"
#include "runtime.hpp"

namespace lahva
{
    namespace gpu
    {
        template <typename T>
        class Tensor_ : public virtual lahva::Tensor_<T>
        {
        public:
            /// @brief copy data to device, by allocating a pointer and copying over
            /// @param cudart Cuda Runtime instance
            /// @return none
            virtual void copy2device(const CudaRuntime &cudart) const = 0;
            /// @brief copy data to host,
            /// @param cudart
            virtual void copy2host(const CudaRuntime &cudart) = 0;

            virtual inline bool alloc_on_device() const = 0;

            virtual T *gpu_data() const = 0;
            virtual T *gpu_data() = 0;
        };

        // Forward declaration for CopyTensors
        template <typename in, typename out>
        void CopyTensors(const unsigned long size, const in *d_in, out *d_out);

        /// @brief GPU-based tensor with dual memory management and CUDA integration
        /// Base class for GPU tensors providing automatic memory management on both CPU and GPU.
        ///
        /// @tparam T data type for tensor elements
        /// @tparam Allocator host (CPU) memory allocator type (default: CudaHostAllocator)
        /// @tparam GPUAllocator device (GPU) memory allocator type (default: CudaDeviceAllocator)
        template <typename T, typename Allocator = CudaHostAllocator<T>, typename GPUAllocator = CudaDeviceAllocator<T>>
        class Tensor : public lahva::Tensor<T, Allocator>, virtual public Tensor_<T>
        {
            using alloc_ptr = CPUAllocator<T>;
            using gpualloc_t = GPUAllocator;

        public:
            /// @brief Construct GPU tensor with specified element count
            /// @param[in] count number of elements in the tensor
            /// @param[in] cpualloc host (CPU) memory allocator
            /// @param[in] alloc device (GPU) memory allocator
            Tensor(size_t count, const alloc_ptr &cpualloc = Allocator(), const GPUAllocator &alloc = GPUAllocator())
                : lahva::Tensor<T, Allocator>{count, cpualloc}, gpualloc_{alloc} {};

            /// @brief Construct GPU tensor without allocating initial memory
            /// @param[in] cpualloc host (CPU) memory allocator
            /// @param[in] alloc device (GPU) memory allocator
            Tensor(const alloc_ptr &cpualloc = Allocator(), const GPUAllocator &alloc = GPUAllocator())
                : lahva::Tensor<T, Allocator>{cpualloc}, gpualloc_{alloc} {};

            /// @brief Construct GPU tensor with GPU allocator for device-only memory
            /// @param[in] alloc device (GPU) memory allocator
            Tensor(const GPUAllocator &alloc) : lahva::Tensor<T, Allocator>{}, gpualloc_{alloc}, gpu_buffer{true}
            {
                this->no_alloc = true;
            };

            /// @brief Destructor for GPU tensor, releases GPU memory and CPU memory (via base class)
            virtual ~Tensor() { this->device_ptr_.get_deleter() = this->gpualloc_; };

            /// @brief Copy constructor for GPU tensor
            /// @param[in] other source GPU tensor to copy
            Tensor(const Tensor &other) : lahva::Tensor<T, Allocator>{other},
                                          gpualloc_{other.get_gpuallocator()},
                                          is_on_device_{other.is_on_device_}
            {
                if (other.is_on_device_)
                {
                    this->device_ptr_.reset(gpualloc_.allocate(this->size()));
                    CopyTensors(other.size(), other.gpu_data(), this->gpu_data());
                }
            };

            /// @brief Move constructor for GPU tensor
            /// @param[in] other source GPU tensor to move from
            Tensor(Tensor &&other) : lahva::Tensor<T, Allocator>{std::move(other)}
            {
                this->gpualloc_ = other.get_gpuallocator();
                if (this->size() == other.size())
                {
                    is_on_device_ = other.is_on_device_;
                    this->device_ptr_ = std::move(other.device_ptr_);
                }
                else
                {
                    this->device_ptr_.reset();
                    this->is_on_device_ = false;
                    other.device_ptr_.reset();
                }

                other.is_on_device_ = false;
            };

            /// @brief Copy assignment operator
            /// @param[in] other source GPU tensor
            /// @return reference to this tensor
            Tensor<T, Allocator, GPUAllocator> &operator=(const Tensor<T, Allocator, GPUAllocator> &other)
            {
                if (this != &other)
                {
                    this->gpualloc_ = other.get_gpuallocator();
                    if (!other.gpu_buffer)
                    {
                        lahva::Tensor<T, Allocator>::operator=(other);
                    }
                    this->is_on_device_ = other.is_on_device_;
                    this->gpu_buffer = other.gpu_buffer;

                    if (other.is_on_device_)
                    {
                        this->device_ptr_.reset(gpualloc_.allocate(this->size()));
                        CopyTensors(other.size(), other.gpu_data(), this->gpu_data());
                    }
                }
                return *this;
            };

            /// @brief Move assignment operator
            /// @param[in] other source GPU tensor to move from
            /// @return reference to this tensor
            Tensor<T, Allocator, GPUAllocator> &operator=(Tensor<T, Allocator, GPUAllocator> &&other)
            {
                if (this != &other)
                {
                    if (!other.gpu_buffer)
                    {
                        lahva::Tensor<T, Allocator>::operator=(std::move(other));
                    }
                    else
                    {
                        this->count_ = other.count_;
                    }
                    this->gpu_buffer = other.gpu_buffer;
                    this->gpualloc_ = other.get_gpuallocator();
                    this->is_on_device_ = other.is_on_device_;
                    this->device_ptr_ = std::move(other.device_ptr_);

                    other.is_on_device_ = false;
                }
                return *this;
            };

        protected:
            /// @brief GPU memory allocator instance
            mutable GPUAllocator gpualloc_;

            /// @brief Pointer to data on GPU, managed as unique_ptr
            mutable std::unique_ptr<T, GPUAllocator> device_ptr_;

            /// @brief Flag tracking whether data is currently allocated on GPU device
            mutable bool is_on_device_ = false;

            /// @brief Flag indicating if buffer is GPU-only (no CPU copy)
            bool gpu_buffer = false;

            /// @brief Flag indicating if host memory is registered with CUDA
            mutable bool registered = false;

        public:
            /// @brief Register host memory for faster GPU transfers
            /// @param[in] cudart CUDA runtime instance
            void registerMem(const CudaRuntime &cudart) const
            {
                if (!registered)
                {
                    get_cuda_error(cudaHostRegister((void *)this->data(), this->size() * sizeof(T), cudaHostRegisterDefault));
                    registered = true;
                }
            };

            /// @brief Unregister host memory from CUDA
            /// @param[in] cudart CUDA runtime instance
            void unregisterMem(const CudaRuntime &cudart) const
            {
                if (registered)
                {
                    get_cuda_error(cudaHostUnregister((void *)this->data()));
                    registered = false;
                }
            };

            /// @brief Copy tensor data from host to GPU device
            /// @param[in] cudart CUDA runtime instance for GPU operations
            void copy2device(const CudaRuntime &cudart) const override;

            /// @brief Copy tensor data from GPU device to host
            /// @param[in] cudart CUDA runtime instance for GPU operations
            void copy2host(const CudaRuntime &cudart) override;

            /// @brief Check if data is allocated and present on GPU device
            /// @return true if tensor data is currently on GPU
            inline bool alloc_on_device() const override { return this->is_on_device_; };

            /// @brief Get pointer to GPU device memory
            /// @return const pointer to GPU device data
            T *gpu_data() const override { return this->device_ptr_.get(); };

            /// @brief Get pointer to GPU device memory
            /// @return non-const pointer to GPU device data
            T *gpu_data() override { return this->device_ptr_.get(); };

            /// @brief Release GPU device pointer without freeing memory
            void release_gpu_ptr() { device_ptr_.reset(); };

            /// @brief Update GPU memory with current host data
            /// @param[in] cudart CUDA runtime instance
            void updateGPUvalues(const CudaRuntime &cudart);

            /// @brief Allocate memory on GPU device
            /// @param[in] cudart CUDA runtime instance
            void allocateGPU(const CudaRuntime &) const;

            /// @brief Deallocate memory on GPU device
            /// @param[in] cudart CUDA runtime instance
            void deallocateGPU(const CudaRuntime &) const;

            /// @brief Get GPU memory allocator
            /// @return GPU allocator instance by move
            GPUAllocator get_gpuallocator() const { return std::move(gpualloc_); };

            /// @brief Get GPU memory allocator
            /// @return GPU allocator instance by reference
            GPUAllocator get_gpuallocator() { return gpualloc_; };
        };

        /// @brief Implementation: Allocate GPU device memory with CUDA runtime context
        template <typename T, typename Allocator, typename GPUAllocator>
        void Tensor<T, Allocator, GPUAllocator>::allocateGPU(const CudaRuntime &cudart) const
        {
            this->gpualloc_.setDevice(cudart.device_id());
            this->gpualloc_.setStream(cudart.getStreamPtr());
            this->device_ptr_.get_deleter() = this->gpualloc_;
            this->device_ptr_.reset(gpualloc_.allocate(this->size()));
        };

        /// @brief Implementation: Release GPU device memory with CUDA runtime context
        template <typename T, typename Allocator, typename GPUAllocator>
        void Tensor<T, Allocator, GPUAllocator>::deallocateGPU(const CudaRuntime &cudart) const
        {
            this->gpualloc_.setDevice(cudart.device_id());
            this->gpualloc_.setStream(cudart.getStreamPtr());
            this->device_ptr_.get_deleter() = this->gpualloc_;
            this->device_ptr_.reset();
            if (!gpu_buffer)
                this->is_on_device_ = false;
        };

        /// @brief Implementation: Invalidate GPU copy and transfer from host to device
        template <typename T, typename Allocator, typename GPUAllocator>
        void Tensor<T, Allocator, GPUAllocator>::updateGPUvalues(const CudaRuntime &cudart)
        {
            this->is_on_device_ = false;
            this->copy2device(cudart);
        }

        /// @brief Implementation: Transfer tensor data from host to GPU asynchronously
        template <typename T, typename Allocator, typename GPUAllocator>
        void Tensor<T, Allocator, GPUAllocator>::copy2device(const CudaRuntime &cudart) const
        {
            if (!this->is_on_device_)
            {

                if (!this->device_ptr_)
                {
                    this->allocateGPU(cudart);
                }
                this->is_on_device_ = true;
                if (this->data() != nullptr)
                {
                    this->gpualloc_.setDevice(cudart.device_id());
                    this->gpualloc_.setStream(cudart.getStreamPtr());
                    this->gpualloc_.H2DCopy(this->device_ptr_.get(), this->data(), this->size() * sizeof(T));
                }
            }

            if (!this->device_ptr_)
            {
                this->allocateGPU(cudart);
            }
        };

        /// @brief Implementation: Transfer tensor data from GPU to host and optionally deallocate
        template <typename T, typename Allocator, typename GPUAllocator>
        void Tensor<T, Allocator, GPUAllocator>::copy2host(const CudaRuntime &cudart)
        {
            if (this->is_on_device_)
            {
                this->gpualloc_.setDevice(cudart.device_id());
                this->gpualloc_.setStream(cudart.getStreamPtr());
                gpualloc_.D2HCopy(this->device_ptr_.get(), this->data(), this->size() * sizeof(T));
                this->is_on_device_ = false;
                if ((cudart.criticalMem()) && (cudart.criticalSize(this->size() * sizeof(T))))
                {
                    this->deallocateGPU(cudart);
                }
                cudart.synchronize();
            }
        };

    } // namespace gpu
} // namespace lahva
