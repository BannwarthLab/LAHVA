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
    template <typename T>
    class Tensor
    {
    public:
        virtual size_t size() = 0;
        virtual T *data() = 0;
        virtual const size_t size() const = 0;
        virtual const T *data() const = 0;
    };
#ifdef _CUDA
    template <typename T>
    class GPUTensor : public Tensor<T>
    {
    public:
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
        mutable std::unique_ptr<T, deleter> device_ptr_;
        mutable bool is_on_device_ = false;

    public:
        const void copy2device(const CudaRuntime &cudart) const;
        void copy2host(const CudaRuntime &cudart);
        inline bool alloc_on_device() const { return this->is_on_device_; };
        const T *gpu_data() const { return device_ptr_.get(); };
        T *gpu_data() { return device_ptr_.get(); };
        void release_gpu_ptr() { device_ptr_.release(); };
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
    }
#endif
} // namespace tcgmtensor

#endif