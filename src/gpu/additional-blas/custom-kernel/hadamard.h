#pragma once
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "runtime.hpp"

namespace lahva
{
    namespace gpu
    {
        void sHadamard(const CudaRuntime& cudart, float* vecinout, const float* vecin, size_t ndim2);
        void dHadamard(const CudaRuntime& cudart, double* vecinout, const double* vecin, size_t ndim2);
        void sHadamardcopy(const CudaRuntime& cudart, float* vecout, const float* vecin1, const float* vecin2, size_t ndim2);
        void dHadamardcopy(const CudaRuntime& cudart, double* vecout, const double* vecin1, const double* vecin2, size_t ndim2);
        void sHadamardcopy(const CudaRuntime& cudart, float* vecout, const float* vecin1, const float* vecin2, size_t ndim2, size_t ndim);
        void dHadamardcopy(const CudaRuntime& cudart, double* vecout, const double* vecin1, const double* vecin2, size_t ndim2, size_t ndim);
    } // namespace gpu
    
} // namespace lahva
