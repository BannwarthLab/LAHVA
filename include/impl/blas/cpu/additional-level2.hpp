/// @file additional-level2.hpp
/// @brief C++-style additional Level-2 operations beyond standard BLAS.
///
// Additional Level-2 BLAS-like operations (matrix * matrix) - C++ template declarations
// These functions are C++-style wrappers providing additional matrix operations
// beyond standard BLAS Level-2 operations. These wrappers operate on Matrix_ objects and
// dispatch to optimized CPU implementations. Each overload is provided for double and
// float precision; both precisions are documented explicitly.

#ifndef LAHVA_ADD_LEVEL2_CPU_HPP
#define LAHVA_ADD_LEVEL2_CPU_HPP
#include "linalg.hpp"
#include "const.h"

namespace lahva
{
    namespace cpu
    {

        /// @brief Add two matrices: C = alpha*op(A) + beta*op(B)
        ///
        /// Performs scaled matrix addition with optional transposes: C = alpha*op(A) + beta*op(B),
        /// where op(A) is A or A^T depending on the transpose flag. Matrices A and B are not modified.
        ///
        /// @tparam T Numerical element type (double, float).
        /// @param Ta Transpose option for matrix A: "N" (no transpose, A), "T" (transpose, A^T).
        /// @param Tb Transpose option for matrix B: "N" (no transpose, B), "T" (transpose, B^T).
        /// @param alpha Scalar multiplier for op(A).
        /// @param a Input matrix A.
        /// @param beta Scalar multiplier for op(B).
        /// @param b Input matrix B.
        /// @param c Output matrix C (overwritten with the result).
        template <typename T>
        void AddMatrices(const char *Ta, const char *Tb, const T alpha, const Matrix_<T> &a, const T beta,
                         const Matrix_<T> &b, Matrix_<T> &c);

        /// @brief Add two matrices with default parameters: C = alpha*op(A) + beta*op(B)
        ///
        /// Convenience overload with default transpose and scalar parameters.
        /// Performs: C = alpha*op(A) + beta*op(B) with optional transposes.
        ///
        /// @tparam T Numerical element type (double, float).
        /// @param a Input matrix A.
        /// @param b Input matrix B.
        /// @param c Output matrix C (overwritten with the result).
        /// @param alpha Scalar multiplier for op(A). Default: 1.0.
        /// @param beta Scalar multiplier for op(B). Default: 1.0.
        /// @param Ta Transpose option for matrix A: "N" (no transpose, A), "T" (transpose, A^T). Default: "N".
        /// @param Tb Transpose option for matrix B: "N" (no transpose, B), "T" (transpose, B^T). Default: "N".
        template <typename T>
        void AddMatrices(const Matrix_<T> &a, const Matrix_<T> &b, Matrix_<T> &c,
                         const T alpha = 1.0, const T beta = 1.0, const char *Ta = "N", const char *Tb = "N");

        /// @brief Runtime-dispatching overload that allows passing a CPURuntime first.
        ///
        /// The runtime parameter is ignored for CPU implementations but this overload
        /// mirrors the signatures used for GPU/CUDA dispatching.
        template <typename... Args>
        void AddMatrices(const CPURuntime &rt_, Args &&...args)
        {
            (AddMatrices(args...));
        }

    } // namespace gpu

}
#endif