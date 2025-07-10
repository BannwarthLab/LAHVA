#include "linalg.hpp"
#include "runtime.hpp"
#include "impl/blas/gpu/additional-level1.hpp"
#include "../gpu-utils/utils.hpp"
#include "reductions/reduction.cuh"
#include "reductions/common.cuh"
#include "additional-level1.cuh"
#include "./additional-level1.hpp"
#include "add-vectors.hpp"
namespace lahva
{
    namespace gpu
    {
        __global__ void AddVector_(unsigned long long ndim, const double alpha, const float *a, double *b)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index < ndim)
            {
                b[index] += static_cast<double>(a[index]*alpha);
            }
            
        };

        __global__ void AddVector_(unsigned long long ndim, const double alpha, const double *a, float *b)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index < ndim)
            {
                b[index] += static_cast<float>(a[index]*alpha);
            }
            
        };

         __global__ void AddVector_(unsigned long long ndim, const double alpha, const __half *a, double *b)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index < ndim)
            {
                b[index] += static_cast<double>(a[index])*alpha;
            }
            
        };

        void AddVector(const CudaRuntime& cudart, unsigned long long ndim, const double alpha, const GPUTensor_<float>& a, GPUTensor_<double>& b)
        {
            check_device_alloc(cudart, a);
            check_device_alloc(cudart, b);
            unsigned long long blockSize = cudart.blockSize();
            AddVector_<<<cudart.gridSize(ndim, 1), blockSize, 0, cudart.getStream()>>>(ndim, alpha, a.gpu_data(), b.gpu_data());
        }

        void AddVector(const CudaRuntime& cudart, unsigned long long ndim, const double alpha, const GPUTensor_<double>& a, GPUTensor_<float>& b)
        {
            check_device_alloc(cudart, a);
            check_device_alloc(cudart, b);
            unsigned long long blockSize = cudart.blockSize();
            AddVector_<<<cudart.gridSize(ndim, 1), blockSize, 0, cudart.getStream()>>>(ndim, alpha, a.gpu_data(), b.gpu_data());
        }
        
        void AddVector(const CudaRuntime& cudart, unsigned long long ndim, const double alpha, const GPUTensor_<__half>& a, GPUTensor_<double>& b)
        {
            check_device_alloc(cudart, a);
            check_device_alloc(cudart, b);
            unsigned long long blockSize = cudart.blockSize();
            AddVector_<<<cudart.gridSize(ndim, 1), blockSize, 0, cudart.getStream()>>>(ndim, alpha, a.gpu_data(), b.gpu_data());
        }

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
            //check_device_alloc(cudart, res);
            T* res_data;
            cudaHostGetDevicePointer(&res_data, res.data(), 0);
            unsigned long long blockSize = cudart.blockSize();
            GPUReduction<T, add_rn<T>>(cudart, in.size(), in.gpu_data(), res_data, blockSize, add_rn<T>());
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

        template<typename in, typename out>
        __global__ void CopyTensors_(unsigned long size, const in* d_in, out* d_out)
        {
            unsigned long idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size)
                d_out[idx] = d_in[idx];
        }

        template<typename in, typename out>
        void CopyTensors(const unsigned long size, const in* d_in, out* d_out)
        {
            unsigned int blockSize = 512;
            int gridSize = (int)ceil(((float)size/blockSize));
            CopyTensors_<in, out><<<gridSize, blockSize, 0, 0>>>(size, d_in, d_out);
        }

        template float MaxElement_<float>(const CudaRuntime& cudart, const GPUTensor_<float>& in, GPUTensor_<float>& res);
        template double MaxElement_<double>(const CudaRuntime& cudart, const GPUTensor_<double>& in, GPUTensor_<double>& res);
        template float MinElement_<float>(const CudaRuntime& cudart, const GPUTensor_<float>& in, GPUTensor_<float>& res);
        template double MinElement_<double>(const CudaRuntime& cudart, const GPUTensor_<double>& in, GPUTensor_<double>& res);
        template float Sum_<float>(const CudaRuntime& cudart, const GPUTensor_<float>& in, GPUTensor_<float>& res);
        template double Sum_<double>(const CudaRuntime& cudart, const GPUTensor_<double>& in, GPUTensor_<double>& res);
        template void ApplyKernel<float, fabs_gpu<float>>(const CudaRuntime& cudart, GPUTensor_<float>& in, fabs_gpu<float> operation);
        template void ApplyKernel<double, fabs_gpu<double>>(const CudaRuntime& cudart, GPUTensor_<double>& in, fabs_gpu<double> operation);
        template void CopyTensors<double, double>(const unsigned long size, const double* d_in, double* d_out);
        template void CopyTensors<float, float>(const unsigned long size, const float* d_in, float* d_out);
        template void CopyTensors<double, float>(const unsigned long size, const double* d_in, float* d_out);
        template void CopyTensors<float, double>(const unsigned long size, const float* d_in, double* d_out);
        template void CopyTensors<int, int>(const unsigned long size, const int* d_in, int* d_out);
        template void CopyTensors<__half, __half>(const unsigned long size, const __half* d_in, __half* d_out);
    } // namespace gpu
    
} // namespace lahva
