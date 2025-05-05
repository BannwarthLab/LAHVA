#pragma once
#include <memory>
#include "impl/tensor/allocators.hpp"
#include "impl/tensor/cpu/tensor.hpp"
#include "runtime.hpp"
#include "additional-level1.hpp"
namespace lahva
{
    namespace gpu
    {

        template <typename T>
        class GPUTensor_ : public virtual Tensor<T>
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

            // virtual const std::shared_ptr<GPUAllocator_<T>> get_gpuallocator() const = 0;
        };

        template <typename T, typename Allocator = CudaHostAllocator<T>, typename GPUAllocator = CudaDeviceAllocator<T>>
        class GPUTensor : public CPUTensor<T, Allocator>, virtual public GPUTensor_<T>
        {
            using alloc_ptr = CPUAllocator<T>;
            using gpualloc_t = GPUAllocator;

        public:
            GPUTensor(size_t count, const alloc_ptr &cpualloc = Allocator(), const GPUAllocator &alloc = GPUAllocator()) 
            : CPUTensor<T, Allocator>{count, cpualloc}, gpualloc_{alloc} { };
            GPUTensor(const alloc_ptr &cpualloc = Allocator(), const GPUAllocator &alloc = GPUAllocator()) 
            : CPUTensor<T, Allocator>{cpualloc}, gpualloc_{alloc} {};
            GPUTensor(const GPUAllocator &alloc) : CPUTensor<T, Allocator>{}, gpualloc_{alloc}, gpu_buffer{true} 
            {};
            virtual ~GPUTensor() {this->device_ptr_.get_deleter() = this->gpualloc_;};
            GPUTensor(const GPUTensor &other) : CPUTensor<T, Allocator>{other},
                                                gpualloc_{other.get_gpuallocator()},
                                                is_on_device_{other.is_on_device_} 
            {
                if (other.is_on_device_)
                {
                    this->device_ptr_.reset(gpualloc_.allocate(this->size()));
                    CopyTensors(other.size(), other.gpu_data(), this->gpu_data());
                }
            };
            GPUTensor(GPUTensor &&other) : CPUTensor<T, Allocator>{std::move(other)}
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
            GPUTensor<T, Allocator, GPUAllocator> &operator=(const GPUTensor<T, Allocator, GPUAllocator> &other)
            {
                if (this != &other)
                {
                    this->gpualloc_ = other.get_gpuallocator();
                    if (!other.gpu_buffer)
                    {
                        CPUTensor<T, Allocator>::operator=(other);
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

            GPUTensor<T, Allocator, GPUAllocator> &operator=(GPUTensor<T, Allocator, GPUAllocator> &&other)
            {
                if (this != &other)
                {
                    if (!other.gpu_buffer)
                    {
                        CPUTensor<T, Allocator>::operator=(std::move(other));
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
            mutable GPUAllocator gpualloc_;

            /// @brief pointer to data on GPU, as unique ptr
            mutable std::unique_ptr<T, GPUAllocator> device_ptr_;
            /// @brief marker to keep track if data is on GPU
            mutable bool is_on_device_ = false;
            bool gpu_buffer = false;
            mutable bool registered = false;

        public:
            void registerMem(const CudaRuntime &cudart) const
            {
                if (!registered)
                {
                    get_cuda_error(cudaHostRegister((void *)this->data(), this->size() * sizeof(T), cudaHostRegisterDefault));
                    registered = true;
                }
            };

            void unregisterMem(const CudaRuntime &cudart) const
            {
                if (registered)
                {
                    get_cuda_error(cudaHostUnregister((void *)this->data()));
                    registered = false;
                }
            };
            /// @brief copy data to device, by allocating a pointer and copying over
            /// @param cudart Cuda Runtime instance
            /// @return none
            void copy2device(const CudaRuntime &cudart) const override;
            /// @brief copy data to host,
            /// @param cudart
            void copy2host(const CudaRuntime &cudart) override;

            inline bool alloc_on_device() const override { return this->is_on_device_; };

            T *gpu_data() const override { return this->device_ptr_.get(); };
            T *gpu_data() override { return this->device_ptr_.get(); };

            void release_gpu_ptr() { device_ptr_.reset(); };

            void updateGPUvalues(const CudaRuntime &cudart);
            void allocateGPU(const CudaRuntime &) const;
            //void deallocateGPU() const;
            void deallocateGPU(const CudaRuntime &) const;
            GPUAllocator get_gpuallocator() const { return std::move(gpualloc_); };
            GPUAllocator get_gpuallocator() { return gpualloc_; };
        };

        template <typename T, typename Allocator, typename GPUAllocator>
        void GPUTensor<T, Allocator, GPUAllocator>::allocateGPU(const CudaRuntime &cudart) const
        {
            this->gpualloc_.setDevice(cudart.device_id());
            this->gpualloc_.setStream(cudart.getStreamPtr());
            this->device_ptr_.get_deleter() = this->gpualloc_;
            this->device_ptr_.reset(gpualloc_.allocate(this->size()));
        };

        template <typename T, typename Allocator, typename GPUAllocator>
        void GPUTensor<T, Allocator, GPUAllocator>::deallocateGPU(const CudaRuntime& cudart) const
        {
            this->gpualloc_.setDevice(cudart.device_id());
            this->gpualloc_.setStream(cudart.getStreamPtr());
            this->device_ptr_.get_deleter() = this->gpualloc_;
            this->device_ptr_.reset();
            if (!gpu_buffer)
                this->is_on_device_ = false;
        };

        template <typename T, typename Allocator, typename GPUAllocator>
        void GPUTensor<T, Allocator, GPUAllocator>::updateGPUvalues(const CudaRuntime &cudart)
        {
            this->is_on_device_ = false;
            this->copy2device(cudart);
        }

        template <typename T, typename Allocator, typename GPUAllocator>
        void GPUTensor<T, Allocator, GPUAllocator>::copy2device(const CudaRuntime &cudart) const
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

        template <typename T, typename Allocator, typename GPUAllocator>
        void GPUTensor<T, Allocator, GPUAllocator>::copy2host(const CudaRuntime &cudart)
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
