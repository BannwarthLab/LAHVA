#pragma once 
#include "linalg.hpp"

namespace lahva
{
    namespace gpu
    {
        void AddVector(const CudaRuntime& cudart, unsigned long long ndim, const double a, const GPUTensor_<double>& x, GPUTensor_<float>& y);
        void AddVector(const CudaRuntime& cudart, unsigned long long ndim, const double a, const GPUTensor_<float>& x, GPUTensor_<double>& y);
        void AddVector(const CudaRuntime& cudart, unsigned long long ndim, const double a, const GPUTensor_<__half>& x, GPUTensor_<double>& y);
    } // namespace gpu
    
} // namespace lahva
