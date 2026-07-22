/// @file additional-level2.hpp
/// @brief GPU-accelerated additional Level-2 operations beyond standard BLAS.
///
/// Provides GPU kernel declarations for supplementary matrix-vector operations
/// operating on GPU matrix and vector objects. These extend standard BLAS Level-2 functionality.
/// Each function is provided for double and float precision with GPU runtime management.

#pragma once
#include "const.h"
#include "linalg.hpp"
#include "runtime.hpp"

namespace lahva
{
    namespace gpu
    {
        template<typename T>
        void AddMatrices(const CudaRuntime& cudart, const char* Ta, const char* Tb, const T alpha, const Matrix_<T>& a, const Matrix_<T>& b,
        const T beta, Matrix_<T>& c);
        template<typename T>
        void AddMatrices(const CudaRuntime& cudart, const Matrix_<T>& a, const Matrix_<T>& b, Matrix_<T>& c,
        const T alpha = 1.0 , const T beta = 1.0, const char* Ta = "N", const char* Tb = "N");

        void SymmetrizedON2ScalingProductGPU(const CudaRuntime& cudart, const Vector_<double>& diag1, const Matrix_<float>& matrix1,
                                             const Vector_<double>& diag2, const Matrix_<float>& matrix2, Matrix_<float>& matrix_out);
    
    
    } // namespace gpu
    
}