#pragma once

#include "linalg.hpp"
#include "runtime.hpp"
namespace lahva
{
    namespace gpu
    {
        void SymmetrizedON2ScalingProductGPU(const CudaRuntime& cudart, const Vector_<double>& diag1, const Matrix_<float>& matrix1,
                                             const Vector_<double>& diag2, const Matrix_<float>& matrix2, Matrix_<float>& matrix_out);
    } // namespace gpu
    
} // namespace lahva