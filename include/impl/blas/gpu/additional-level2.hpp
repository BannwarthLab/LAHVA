#ifndef LAHVA_ADD_LEVEL2_GPU_HPP
#define LAHVA_ADD_LEVEL2_GPU_HPP
#include "linalg.hpp"
#include "const.h"
#include "runtime.hpp"

namespace lahva
{
    namespace gpu
    {
        /// @brief Adds two GPU matrices with optional transposition.
        ///
        /// Performs C = alpha * op(A) + beta * op(B) where op is specified by Ta and Tb.
        ///
        /// @tparam T Numerical type of matrix elements (double, float, complex types).
        /// @param cudart CUDA runtime instance
        /// @param Ta Operation on matrix A ("N" for no transpose, "T" for transpose).
        /// @param Tb Operation on matrix B ("N" for no transpose, "T" for transpose).
        /// @param alpha Scalar multiplier for matrix A.
        /// @param a First input GPU matrix (Matrix_<T>).
        /// @param b Second input GPU matrix (Matrix_<T>).
        /// @param beta Scalar multiplier for matrix B.
        /// @param c Output GPU matrix to store result (destination / input-output, Matrix_<T>).
        template<typename T>
        void AddMatrices(const CudaRuntime& cudart, const char* Ta, const char* Tb, const T alpha, const Matrix_<T>& a, const Matrix_<T>& b,
        const T beta, Matrix_<T>& c);

        /// @brief Adds two GPU matrices with optional transposition (convenience overload).
        ///
        /// Performs C = alpha * op(A) + beta * op(B) with default parameters.
        ///
        /// @tparam T Numerical type of matrix elements (double, float, complex types).
        /// @param cudart CUDA runtime instance
        /// @param a First input GPU matrix (Matrix_<T>).
        /// @param b Second input GPU matrix (Matrix_<T>).
        /// @param c Output GPU matrix to store result (destination / input-output, Matrix_<T>).
        /// @param alpha Scalar multiplier for A (default: 1.0).
        /// @param beta Scalar multiplier for B (default: 1.0).
        /// @param Ta Operation on A (default: "N" for no transpose).
        /// @param Tb Operation on B (default: "N" for no transpose).
        template<typename T>
        void AddMatrices(const CudaRuntime& cudart, const Matrix_<T>& a, const Matrix_<T>& b, Matrix_<T>& c,
        const T alpha = 1.0 , const T beta = 1.0, const char* Ta = "N", const char* Tb = "N");

        /// @brief Computes scaled symmetric product using ON2 scaling (GPU).
        ///
        /// Computes matrix_out = scaling1 * matrix1 * scaling2 * matrix2 where scaling factors are diagonal.
        /// Optimized for symmetric matrices with explicit scaling vectors.
        ///
        /// @param cudart CUDA runtime instance
        /// @param diag1 Diagonal scaling vector for first matrix (Vector_<double>).
        /// @param matrix1 First input matrix in single-precision (Matrix_<float>).
        /// @param diag2 Diagonal scaling vector for second matrix (Vector_<double>).
        /// @param matrix2 Second input matrix in single-precision (Matrix_<float>).
        /// @param matrix_out Output matrix with computed product (destination / input-output, Matrix_<float>).
        void SymmetrizedON2ScalingProductGPU(const CudaRuntime& cudart, const Vector_<double>& diag1, const Matrix_<float>& matrix1,
                                             const Vector_<double>& diag2, const Matrix_<float>& matrix2, Matrix_<float>& matrix_out);
    
    
    } // namespace gpu

}   // namespace lahva
#endif