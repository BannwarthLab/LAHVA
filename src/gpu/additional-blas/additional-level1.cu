#include "impl/blas/gpu/additional-level1.hpp"
#include "../../gpu-utils/utils.hpp"
#include "reductions/reduction.cuh"
#include "reductions/common.cuh"
namespace lahva
{
    namespace gpu
    {
        template<typename T, class op>
        void ApplyKernel(const CudaRuntime& cudart, GPUTensor_<T>& in, op operation)
        {
            check_device_alloc(cudart, in);

            ApplyKernel_<<<cudart.gridSize(in.size(), 1), cudart.blockSize(), 0, cudart.getStream()>>>(in.size(), in.gpu_data(), operation);
        }

        template<typename T>
        T Sum_(const CudaRuntime& cudart, const GPUTensor_<T>& in, GPUTensor_<T>& res)
        {
            check_device_alloc(cudart, in);
            check_device_alloc(cudart, res);
            unsigned long long blockSize = cudart.blockSize();
            GPUReduction<T, add_rn<T>>(cudart, in.size(), in.gpu_data(), res.gpu_data(), blockSize, add_rn<T>());
            
            res.copy2host(cudart);
            cudart.synchronize();
            return res[0];
        }

        template<typename T>
        T MaxElement_(const CudaRuntime& cudart, const GPUTensor_<T>& in, GPUTensor_<T>& res)
        {
            check_device_alloc(cudart, in);
            check_device_alloc(cudart, res);
            unsigned long long blockSize = cudart.blockSize();
            GPUReduction<T, max_<T>>(cudart, in.size(), in.gpu_data(), res.gpu_data(), blockSize);
            
            res.copy2host(cudart);
            cudart.synchronize();
           
            return res[0];
        }

        template<typename T>
        T MinElement_(const CudaRuntime& cudart, const GPUTensor_<T>& in, GPUTensor_<T>& res)
        {
            check_device_alloc(cudart, in);
            check_device_alloc(cudart, res);
            unsigned long long blockSize = cudart.blockSize();
            GPUReduction<T, min_<T>>(cudart, in.size(), in.gpu_data(), res.gpu_data(), blockSize);
            
            res.copy2host(cudart);
            cudart.synchronize();
            
            return res[0];
        }


        template float MaxElement_<float>(const CudaRuntime& cudart, const GPUTensor_<float>& in, GPUTensor_<float>& res);
        template double MaxElement_<double>(const CudaRuntime& cudart, const GPUTensor_<double>& in, GPUTensor_<double>& res);
        template float MinElement_<float>(const CudaRuntime& cudart, const GPUTensor_<float>& in, GPUTensor_<float>& res);
        template double MinElement_<double>(const CudaRuntime& cudart, const GPUTensor_<double>& in, GPUTensor_<double>& res);
        template float Sum_<float>(const CudaRuntime& cudart, const GPUTensor_<float>& in, GPUTensor_<float>& res);
        template double Sum_<double>(const CudaRuntime& cudart, const GPUTensor_<double>& in, GPUTensor_<double>& res);
        template void ApplyKernel<float, fabs_gpu<float>>(const CudaRuntime& cudart, GPUTensor_<float>& in, fabs_gpu<float> operation);
        template void ApplyKernel<double, fabs_gpu<double>>(const CudaRuntime& cudart, GPUTensor_<double>& in, fabs_gpu<double> operation);
    } // namespace gpu
    
} // namespace lahva
