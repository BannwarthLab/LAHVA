/// @file add-vectors.hpp
/// @brief Mixed-precision vector addition kernels for GPU computation.
///
/// Provides functions for adding vectors with type conversion between different
/// precisions (double, float, half-precision).

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
