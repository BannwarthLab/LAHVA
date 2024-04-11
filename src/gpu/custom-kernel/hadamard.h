#pragma once
#include <cuda_runtime.h>
#include "cublas_v2.h"

namespace tcgmtensor
{
    namespace gpu
    {
        void sHadamard(float* vecinout, const float* vecin, size_t ndim2);
        void dHadamard(double* vecinout, const double* vecin, size_t ndim2);
        void sHadamardcopy(float* vecout, const float* vecin1, const float* vecin2, size_t ndim2);
        void dHadamardcopy(double* vecout, const double* vecin1, const double* vecin2, size_t ndim2);
    } // namespace gpu
    
} // namespace tcgmtensor
