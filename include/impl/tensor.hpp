#ifndef TCGMTENSOR_TENSOR_HPP
#define TCGMTENSOR_TENSOR_HPP

#include <memory>
#include <iterator>

#include <iostream>
#ifdef _CUDA
    #include "runtime.hpp"
#endif

namespace tcgmtensor
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
        virtual const size_t size() const = 0;
        /// @brief return pointer to the underlying data
        /// @return ptr to begin of buffer
        virtual const T *data() const = 0;
    };
#ifdef _CUDA
    /// @brief abstract base class for GPU compatibility handling transfer and gpu ptr 
    /// @tparam T type of values to be stored in tensor
    template <typename T>
    class GPUTensor : public Tensor<T>
    {
    public:
        /// @brief deleter function used to destroy GPU ptr
        struct deleter
        {
            void operator()(T *ptr)
            {
                if (ptr != nullptr)
                {
                    cudaError_t cuda_error = (cudaFree(ptr));
                    get_cuda_error(cuda_error);
                }
            }
        };

    protected:
        /// @brief pointer to data on GPU, as unique ptr
        mutable std::unique_ptr<T, deleter> device_ptr_;
        /// @brief marker to keep track if data is on GPU
        mutable bool is_on_device_ = false;

    public:
        /// @brief copy data to device, by allocating a pointer and copying over
        /// @param cudart Cuda Runtime instance
        /// @return none
        const void copy2device(const CudaRuntime &cudart) const;
        /// @brief copy data to host, 
        /// @param cudart 
        void copy2host(const CudaRuntime &cudart);
        inline bool alloc_on_device() const { return this->is_on_device_; };
        const T *gpu_data() const { return device_ptr_.get(); };
        T *gpu_data() { return device_ptr_.get(); };
        void release_gpu_ptr() { device_ptr_.reset();};
        virtual void copyGPUTensor(const GPUTensor<T> &other);
        virtual void moveGPUTensor(GPUTensor<T> &&other);
    };

    template <typename T>
    T *allocate(size_t count = 1)
    {
        T *ptr = 0;
        size_t bytes = 0;
        bytes = count * sizeof(T);
        cudaError_t istat = cudaMalloc((void **)&ptr, bytes);
        get_cuda_error(istat);
        return ptr;
    };

    template <typename T>
    void free(T *ptr)
    {
        if (ptr)
        {
            cudaError_t cuda_error = (cudaFree(ptr));
            get_cuda_error(cuda_error);
        }
    }
#else
    template <typename T>
    class GPUTensor : public Tensor<T>
    {
        protected:
            mutable std::unique_ptr<T> device_ptr_;
            mutable bool is_on_device_ = false;
        public:
            virtual void copyGPUTensor(const GPUTensor<T> &other) {};
            virtual void moveGPUTensor(GPUTensor<T> &&other) {};
        
    };
#endif
} // namespace tcgmtensor

#endif