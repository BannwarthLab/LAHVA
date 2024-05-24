#pragma once
#include "linalg.hpp"
#include "runtime.hpp"
namespace tcgmtensor
{
    namespace gpu
    {
        void SymmetrizedON2ScalingProductGPU(const CudaRuntime& cudart, const Vector<double>& diag1, const Matrix<float>& matrix1,
                                             const Vector<double>& diag2, const Matrix<float>& matrix2, Matrix<float>& matrix_out);
    } // namespace gpu
    
} // namespace tcgmtensor