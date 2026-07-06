/// @file mixed-precision.h
/// @brief GPU kernels for mixed-precision matrix operations.
///
/// Provides custom GPU kernel for symmetrized matrix operations with diagonal scaling
/// using mixed-precision arithmetic.

#pragma once

#include "linalg.hpp"
#include "runtime.hpp"
namespace lahva
{
    namespace gpu
    {
        /// @brief Computes symmetrized diagonal-scaled matrix product: out = sym(diag1*matrix1 + diag2*matrix2^T).
        ///
        /// Performs scaled matrix operations with diagonal scaling vectors and symmetrization
        /// for use in mixed-precision iterative refinement algorithms.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param diag1 First diagonal scaling vector (double-precision).
        /// @param matrix1 First input matrix (single-precision).
        /// @param diag2 Second diagonal scaling vector (double-precision).
        /// @param matrix2 Second input matrix (single-precision).
        /// @param matrix_out Output matrix (single-precision), replaced with result.
        void SymmetrizedON2ScalingProductGPU(const CudaRuntime& cudart, const Vector_<double>& diag1, const Matrix_<float>& matrix1,
                                             const Vector_<double>& diag2, const Matrix_<float>& matrix2, Matrix_<float>& matrix_out);
    } // namespace gpu

} // namespace lahva