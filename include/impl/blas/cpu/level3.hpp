/// @file level3.hpp
/// @brief C++-style BLAS Level-3 operations (matrix-matrix operations).
///
// Level-3 BLAS-like operations (matrix * matrix) - C++ template declarations
// These functions are C++-style wrappers operating on Matrix_<> and LowTriMatrix_<> objects.
// They provide convenient overloads for different scalar types and dispatch to optimized
// CPU BLAS kernels. Each overload is provided for double and float precision; both precisions
// are documented explicitly.

#pragma once
#include "const.h"
#include "linalg.hpp"

namespace lahva
{
    namespace cpu
    {
        /// @brief Multiply two matrices of type Matrix_<T> and store the result in `c`.
        ///
        /// Performs C = alpha * op(A) * op(B) + beta * C where op(X) is either X, X^T or X^H
        /// depending on the transpose flags `Ta` and `Tb`. Wrapper to BLAS functions dgemm (double),
        /// sgemm (float), zgemm (complex_double), or cgemm (complex_float). This C++-style wrapper
        /// performs size and transpose checks on Matrix_<T> instances and forwards the call to the
        /// lower-level pointer-based MatrixMatrixProduct implementation declared in the C-style header.
        ///
        /// @tparam T Numeric element type (double, float, complex_double, complex_float).
        /// @param Ta    Transpose option for A: "N" (no transpose, A), "T" (transpose, A^T), "C" (conjugate-transpose, A^H).
        /// @param Tb    Transpose option for B: "N" (no transpose, B), "T" (transpose, B^T), "C" (conjugate-transpose, B^H).
        /// @param alpha Scaling factor for op(A)*op(B).
        /// @param a     Left-hand input matrix (Matrix_<T>).
        /// @param b     Right-hand input matrix (Matrix_<T>).
        /// @param beta  Scaling factor applied to existing contents of `c`.
        /// @param c     Output matrix which will receive the result.
        template <typename T>
        void MatrixMatrixProduct(const char *Ta, const char *Tb, const T alpha, const Matrix_<T> &a, const Matrix_<T> &b,
                                 const T beta, Matrix_<T> &c);

        /// @brief Convenience overload allowing scalar types differing from matrix element type.
        ///
        /// Performs C = alpha * op(A) * op(B) + beta * C where scalar types are cast to the matrix
        /// element type `T`. Dispatches to the appropriate BLAS function (dgemm, sgemm, zgemm, or cgemm)
        /// based on `T`. Useful when calling with double literals against float matrices or other
        /// mixed-precision convenience calls.
        ///
        /// @tparam T      Matrix element type.
        /// @tparam Scalar Scalar type for `alpha`.
        /// @tparam Scalar2 Scalar type for `beta`.
        /// @param Ta    Transpose option for A: "N" (no transpose, A), "T" (transpose, A^T), "C" (conjugate-transpose, A^H).
        /// @param Tb    Transpose option for B: "N" (no transpose, B), "T" (transpose, B^T), "C" (conjugate-transpose, B^H).
        /// @param alpha Scaling factor for op(A)*op(B) (will be cast to T).
        /// @param a     Left-hand input matrix (Matrix_<T>).
        /// @param b     Right-hand input matrix (Matrix_<T>).
        /// @param beta  Scaling factor applied to existing contents of `c` (will be cast to T).
        /// @param c     Output matrix which will receive the result (Matrix_<T>).
        template <typename T, typename Scalar, typename Scalar2>
        void MatrixMatrixProduct(const char *Ta, const char *Tb, const Scalar alpha, const Matrix_<T> &a, const Matrix_<T> &b,
                                 const Scalar2 beta, Matrix_<T> &c)
        {
            MatrixMatrixProduct(Ta, Tb, (T)alpha, a, b, (T)beta, c);
        };

        /// @brief Convenience overload that uses default transpose flags and scalar defaults.
        ///
        /// Performs C = alpha * op(A) * op(B) + beta * C with `Ta`/`Tb` defaulting to "N"
        /// (no transpose) and alpha/beta defaulting to 1/0 respectively. Dispatches to the appropriate
        /// BLAS function (dgemm, sgemm, zgemm, or cgemm) based on `T`. Useful for the common case C = A*B.
        ///
        /// @tparam T Matrix element type.
        /// @tparam Scalar Scalar type for `alpha` (defaults to 1.0 if omitted).
        /// @tparam Scalar2 Scalar type for `beta` (defaults to 0.0 if omitted).
        /// @param a     Left-hand input matrix (Matrix_<T>).
        /// @param b     Right-hand input matrix (Matrix_<T>).
        /// @param c     Output matrix which will receive the result (Matrix_<T>).
        /// @param alpha Scaling factor for op(A)*op(B) (will be cast to T). Default: 1.0.
        /// @param beta  Scaling factor applied to existing contents of `c` (will be cast to T). Default: 0.0.
        /// @param Ta    Transpose option for A: "N" (no transpose, A), "T" (transpose, A^T), "C" (conjugate-transpose, A^H). Default: "N"
        /// @param Tb    Transpose option for B: "N" (no transpose, B), "T" (transpose, B^T), "C" (conjugate-transpose, B^H). Default: "N"
        template <typename T, typename Scalar, typename Scalar2>
        void MatrixMatrixProduct(const Matrix_<T> &a, const Matrix_<T> &b, Matrix_<T> &c,
                                 const Scalar alpha = 1.0, const Scalar2 beta = 0.0, const char *Ta = "N", const char *Tb = "N")
        {
            MatrixMatrixProduct(Ta, Tb, (T)alpha, a, b, (T)beta, c);
        };

        /// @brief Block-diagonal matrix multiply: C = alpha * op(A) * op(B) + beta * C
        ///
        /// Performs C = alpha * op(A) * op(B) + beta * C where A is block-diagonal,
        /// and op(X) is X, X^T, or X^H depending on transpose flags.
        ///
        /// @param Ta    Transpose for A: "N" (none), "T" (transpose), "C" (conjugate-transpose).
        /// @param Tb    Transpose for B: "N" (none), "T" (transpose), "C" (conjugate-transpose).
        /// @param alpha Scaling factor for op(A)*op(B).
        /// @param a     Block-diagonal input matrix.
        /// @param b     Dense input matrix.
        /// @param beta  Scaling factor for existing c.
        /// @param c     Output matrix (modified).
        template <typename T>
        void MatrixMatrixProduct(const char *Ta, const char *Tb, const T alpha, const BlockDiagMatrix_<T> &a, const Matrix_<T> &b,
                                 const T beta, Matrix_<T> &c);

        /// @brief Block-diagonal multiply with default parameters, convenience overload.
        ///
        /// Performs C = alpha * op(A) * op(B) + beta * C with defaults: Ta="N", Tb="N".
        ///
        /// @param a     Block-diagonal input matrix.
        /// @param b     Dense input matrix.
        /// @param c     Output matrix (modified).
        /// @param alpha Scaling factor for A*B (default: 1.0).
        /// @param beta  Scaling factor for existing c (default: 0.0).
        /// @param Ta    Transpose for A (default: "N").
        /// @param Tb    Transpose for B (default: "N").
        template <typename T, typename Scalar = T, typename Scalar2 = T>
        void MatrixMatrixProduct(const BlockDiagMatrix_<T> &a, const Matrix_<T> &b, Matrix_<T> &c,
                                const Scalar alpha = 1.0, const Scalar2 beta = 0.0, const char *Ta = "N", const char *Tb = "N")
        {
            MatrixMatrixProduct(Ta, Tb, (T)alpha, a, b, (T)beta, c);
        };

        /// @brief Block-diagonal matrix multiply: C = alpha * op(A) * op(B) + beta * C
        ///
        /// Performs C = alpha * op(A) * op(B) + beta * C where B is block-diagonal,
        /// and op(X) is X, X^T, or X^H depending on transpose flags.
        ///
        /// @param Ta    Transpose for A: "N" (none), "T" (transpose), "C" (conjugate-transpose).
        /// @param Tb    Transpose for B: "N" (none), "T" (transpose), "C" (conjugate-transpose).
        /// @param alpha Scaling factor for op(A)*op(B).
        /// @param a     Block-diagonal input matrix.
        /// @param b     Dense input matrix.
        /// @param beta  Scaling factor for existing c.
        /// @param c     Output matrix (modified).
        template <typename T>
        void MatrixMatrixProduct(const char *Ta, const char *Tb, const T alpha, const Matrix_<T> &a, const BlockDiagMatrix_<T> &b,
                                 const T beta, Matrix_<T> &c);

        /// @brief Block-diagonal multiply with default parameters, convenience overload.
        ///
        /// Performs C = alpha * op(A) * op(B) + beta * C with defaults: Ta="N", Tb="N".
        ///
        /// @param a     Block-diagonal input matrix.
        /// @param b     Dense input matrix.
        /// @param c     Output matrix (modified).
        /// @param alpha Scaling factor for A*B (default: 1.0).
        /// @param beta  Scaling factor for existing c (default: 0.0).
        /// @param Ta    Transpose for A (default: "N").
        /// @param Tb    Transpose for B (default: "N").
        template <typename T, typename Scalar = T, typename Scalar2 = T>
        void MatrixMatrixProduct(const Matrix_<T> &a, const BlockDiagMatrix_<T> &b, Matrix_<T> &c,
                                const Scalar alpha = 1.0, const Scalar2 beta = 0.0, const char *Ta = "N", const char *Tb = "N")
        {
            MatrixMatrixProduct(Ta, Tb, (T)alpha, a, b, (T)beta, c);
        };

        /// @brief Symmetric matrix-matrix multiply (double precision), wrapper to BLAS function dsymm.
        ///
        /// Performs C = alpha * A * B + beta * C when `side == CblasLeft`, or
        /// C = alpha * B * A + beta * C when `side == CblasRight`. A is assumed symmetric and stored
        /// in lower triangular packed format.
        ///
        /// @param side  Which side the symmetric matrix A appears on (CblasLeft or CblasRight).
        /// @param alpha Scaling factor for the product involving A.
        /// @param a     Symmetric matrix A (Matrix_<double>). If side == CblasLeft, A has shape (m x m); if side == CblasRight, A has shape (n x n).
        /// @param b     Matrix B (Matrix_<double>), shape (m x n).
        /// @param beta  Scaling factor for existing contents of C.
        /// @param c     Output matrix C (destination / input-output, Matrix_<double>), shape (m x n).
        void SymMatrixMatrixProduct(const CBLAS_SIDE side, const double alpha, const Matrix_<double> &a, const Matrix_<double> &b,
                                    const double beta, Matrix_<double> &c);

        /// @brief Convenience overload for double-precision symmetric multiply using defaults, wrapper to BLAS function dsymm.
        ///
        /// Performs C = alpha * A * B + beta * C when side == CblasLeft (default),
        /// or C = alpha * B * A + beta * C when side == CblasRight. Defaults to left-side application
        /// and alpha=1.0, beta=0.0.
        ///
        /// @param a     Symmetric matrix A (Matrix_<double>). If side == CblasLeft (default), A has shape (m x m).
        /// @param b     Matrix B (Matrix_<double>), shape (m x m) when side == CblasLeft.
        /// @param c     Output matrix C (destination / input-output, Matrix_<double>), same shape as B.
        /// @param alpha Scaling factor for the product involving A. Default: 1.0
        /// @param beta  Scaling factor for existing contents of C. Default: 0.0
        /// @param side  Which side the symmetric matrix A appears on (CblasLeft or CblasRight). Default: CblasLeft
        inline void SymMatrixMatrixProduct(const Matrix_<double> &a, const Matrix_<double> &b, Matrix_<double> &c,
                                           const double alpha = 1.0, const double beta = 0.0, const CBLAS_SIDE side = CblasLeft)
        {
            SymMatrixMatrixProduct(side, alpha, a, b, beta, c);
        };
        /// @brief Symmetric matrix-matrix multiply (single precision), wrapper to BLAS function ssymm.
        ///
        /// Performs C = alpha * A * B + beta * C when side == CblasLeft,
        /// or C = alpha * B * A + beta * C when side == CblasRight. A is assumed symmetric and stored
        /// in lower triangular packed format (only lower triangle elements stored). Float variant.
        ///
        /// @param side  Which side the symmetric matrix A appears on (CblasLeft or CblasRight).
        /// @param alpha Scaling factor for the product involving A.
        /// @param a     Symmetric matrix A (Matrix_<float>). If side == CblasLeft, A has shape (m x m); if side == CblasRight, A has shape (n x n).
        /// @param b     Matrix B (Matrix_<float>), shape (m x n).
        /// @param beta  Scaling factor for existing contents of C.
        /// @param c     Output matrix C (destination / input-output, Matrix_<float>), shape (m x n).
        void SymMatrixMatrixProduct(const CBLAS_SIDE side, const float alpha, const Matrix_<float> &a, const Matrix_<float> &b,
                                    const float beta, Matrix_<float> &c);

        /// @brief Convenience overload for single-precision symmetric multiply using defaults, wrapper to BLAS function ssymm.
        ///
        /// Performs C = alpha * A * B + beta * C when side == CblasLeft (default),
        /// or C = alpha * B * A + beta * C when side == CblasRight. Defaults to left-side application
        /// and alpha=1.0, beta=0.0.
        ///
        /// @param a     Symmetric matrix A (Matrix_<float>). If side == CblasLeft (default), A has shape (m x m).
        /// @param b     Matrix B (Matrix_<float>), shape (m x n).
        /// @param c     Output matrix C (destination / input-output, Matrix_<float>), shape (m x n).
        /// @param alpha Scaling factor for the product involving A. Default: 1.0
        /// @param beta  Scaling factor for existing contents of C. Default: 0.0
        /// @param side  Which side the symmetric matrix A appears on (CblasLeft or CblasRight). Default: CblasLeft
        inline void SymMatrixMatrixProduct(const Matrix_<float> &a, const Matrix_<float> &b, Matrix_<float> &c,
                                           const float alpha = 1.0, const float beta = 0.0, const CBLAS_SIDE side = CblasLeft)
        {
            SymMatrixMatrixProduct(side, alpha, a, b, beta, c);
        };

        /// @brief Runtime-dispatching overload that allows passing a CPURuntime first.
        ///
        /// The runtime parameter is ignored for CPU implementations but this overload
        /// mirrors the signatures used for GPU/CUDA dispatching.
        template <typename... Args>
        void MatrixMatrixProduct(const CPURuntime &rt_, Args &&...args)
        {
            (MatrixMatrixProduct(args...));
        }

        /// @brief Runtime-dispatching overload for symmetric multiply (CPURuntime first).
        ///
        /// The runtime parameter is ignored for CPU implementations but this overload
        /// mirrors the signatures used for GPU/CUDA dispatching.
        template <typename... Args>
        void SymMatrixMatrixProduct(const CPURuntime &rt_, Args &&...args)
        {
            (SymMatrixMatrixProduct(args...));
        }

    }
}
