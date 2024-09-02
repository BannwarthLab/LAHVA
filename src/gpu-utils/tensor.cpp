#include "impl/tensor.hpp"

#ifdef _CUDA
#include "runtime.hpp"
void *operator new(size_t size)
{
    void *ptr;
    get_cuda_error(cudaMallocHost(&ptr, size));
    return ptr;
}

void *operator new[](size_t size)
{
    void *ptr;
    get_cuda_error(cudaMallocHost(&ptr, size));
    return ptr;
}

void operator delete(void *ptr)
{
    get_cuda_error(cudaFreeHost(ptr));
}

void operator delete[](void *ptr)
{
    get_cuda_error(cudaFreeHost(ptr));
}
#endif

namespace tcgmtensor
{
    template <typename T>
    void GPUTensor<T>::allocateGPU(const CudaRuntime &cudart) const
    {
        if (cudart.asyncCopy())
        {
            this->device_ptr_.reset(allocate<T>(this->size(), cudart.getStream()));
        }
        else
        {
            this->device_ptr_.reset(allocate<T>(this->size()));
        }
    };

    template <typename T>
    void GPUTensor<T>::deallocateGPU(const CudaRuntime &cudart) const
    {
        if (cudart.asyncCopy())
        {
            get_cuda_error(cudaFreeAsync(device_ptr_.release(), cudart.getStream()));

        }
    };

    template<typename T>
    void GPUTensor<T>::updateGPUvalues(const CudaRuntime& cudart)
    {
        this->is_on_device_ = false;
        this->copy2device(cudart);
    }

    template <typename T>
    const void GPUTensor<T>::copy2device(const CudaRuntime &cudart) const
    {
        if (!this->is_on_device_)
        {
            cudaError_t stat_;
            // stat_ = cudaSetDevice(cudart.device_id());
            // get_cuda_error(stat_);
            if (!this->device_ptr_)
            {
                this->allocateGPU(cudart);
            }
            this->is_on_device_ = true;
            if (cudart.asyncCopy())
            {
                stat_ = cudaMemcpyAsync(this->device_ptr_.get(), this->data(), this->size() * sizeof(T), cudaMemcpyHostToDevice, cudart.getStream());
            }
            else
            {
                stat_ = cudaMemcpy(this->device_ptr_.get(), this->data(), this->size() * sizeof(T), cudaMemcpyHostToDevice);
            };
            get_cuda_error(stat_);
        }
    };

    template <typename T>
    void GPUTensor<T>::copy2host(const CudaRuntime &cudart)
    {
        if (this->is_on_device_)
        {
            cudaError_t stat_;
            // stat_ = cudaSetDevice(cudart.device_id());
            // get_cuda_error(stat_);
            if (cudart.asyncCopy())
            {
                stat_ = cudaMemcpyAsync(this->data(), this->device_ptr_.get(), this->size() * sizeof(T), cudaMemcpyDeviceToHost, cudart.getStream());
            }
            else
            {
                stat_ = cudaMemcpy(this->data(), this->device_ptr_.get(), this->size() * sizeof(T), cudaMemcpyDeviceToHost);
            }
            get_cuda_error(stat_);
            this->is_on_device_ = false;
        }
    };

    // copy operations
    template <typename T>
    void GPUTensor<T>::copyGPUTensor(const GPUTensor<T> &other)
    {
        this->is_on_device_ = false;
    }

    // move operations
    template <typename T>
    void GPUTensor<T>::moveGPUTensor(GPUTensor<T> &&other)
    {
        this->device_ptr_ = std::move(other.device_ptr_);
        this->is_on_device_ = other.is_on_device_;
    }

    template class GPUTensor<double>;
    template class GPUTensor<float>;
    template class GPUTensor<int>;
    template class GPUTensor<int64_t>;

} // namespace tcgmtensor
