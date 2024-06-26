#include "impl/tensor.hpp"
#include "runtime.hpp"

namespace tcgmtensor
{
    
    template <typename T>
    const void GPUTensor<T>::copy2device(const CudaRuntime &cudart) const
    {
        if (!this->is_on_device_)
        {
            cudaError_t stat_;
            stat_ = cudaSetDevice(cudart.device_id());
            get_cuda_error(stat_);
            if (!this->device_ptr_) this->device_ptr_.reset(allocate<T>(this->size()));
            this->is_on_device_ = true;
            if (cudart.asyncCopy())
            {
                cudaError_t stat = cudaMemcpyAsync(this->device_ptr_.get(), this->data(), this->size() * sizeof(T), cudaMemcpyHostToDevice, cudart.getStream());
                get_cuda_error(stat);
            }
            else
            {
                cudaError_t stat = cudaMemcpy(this->device_ptr_.get(), this->data(), this->size() * sizeof(T), cudaMemcpyHostToDevice);
                get_cuda_error(stat);
            };
            
        }
    };

    template <typename T>
    void GPUTensor<T>::copy2host(const CudaRuntime &cudart)
    {
        if (this->is_on_device_)
        {
            cudaError_t stat_;
            stat_ = cudaSetDevice(cudart.device_id());
            get_cuda_error(stat_);
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

} // namespace tcgmtensor

