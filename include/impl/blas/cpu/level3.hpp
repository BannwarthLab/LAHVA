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
    void MatrixMatrixProduct(const char* Ta, const char* Tb, const double alpha, const Matrix_<double>& a, const Matrix_<double>& b,
                             const double beta, Matrix_<double>& c);
    void MatrixMatrixProduct(const Matrix_<double>& a, const Matrix_<double>& b, Matrix_<double>& c,
                             const double alpha = 1.0 , const double beta = 0.0, const char* Ta = "N", const char* Tb = "N");
    void MatrixMatrixProduct(const char* Ta, const char* Tb, const float alpha, const Matrix_<float>& a, const Matrix_<float>& b,
                             const float beta, Matrix_<float>& c);
    void MatrixMatrixProduct(const Matrix_<float>& a, const Matrix_<float>& b, Matrix_<float>& c,
                             const float alpha = 1.0 , const float beta = 0.0, const char* Ta = "N", const char* Tb = "N");

    void MatrixMatrixProduct(const char* Ta, const char* Tb, const complex_double alpha, const Matrix_<complex_double>& a, 
                             const Matrix_<complex_double>& b, const complex_double beta, Matrix_<complex_double>& c);
    void MatrixMatrixProduct(const Matrix_<complex_double>& a, const Matrix_<complex_double>& b, Matrix_<complex_double>& c,
                             const complex_double alpha = 1.0 , const complex_double beta = 0.0, const char* Ta = "N", const char* Tb = "N");
    void MatrixMatrixProduct(const char* Ta, const char* Tb, const complex_float alpha, const Matrix_<complex_float>& a, 
                             const Matrix_<complex_float>& b, const complex_float beta, Matrix_<complex_float>& c);
    void MatrixMatrixProduct(const Matrix_<complex_float>& a, const Matrix_<complex_float>& b, Matrix_<complex_float>& c,
                             const complex_float alpha = 1.0 , const complex_float beta = 0.0, const char* Ta = "N", const char* Tb = "N");

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
    void SymMatrixMatrixProduct(const Matrix_<double>& a, const Matrix_<double>& b, Matrix_<double>& c,
                                const double alpha = 1.0 , const double beta = 0.0, const CBLAS_SIDE side = CblasLeft);
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
    void SymMatrixMatrixProduct(const Matrix_<float>& a, const Matrix_<float>& b, Matrix_<float>& c,
                                const float alpha = 1.0 , const float beta = 0.0, const CBLAS_SIDE side = CblasLeft);

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
