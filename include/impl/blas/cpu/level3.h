/// @file level3.h
/// @brief C-style BLAS Level-3 operations (matrix-matrix operations).
///
// Level-3 BLAS-like operations (matrix * matrix) - C-style declarations
// These functions are thin wrappers around optimized CPU BLAS kernels. They operate on
// raw pointers and use simple scalar parameters for flexibility. Each overload is
// provided for double and float precision; both precisions are documented explicitly.

#pragma once
#include "const.h"

namespace lahva
{
    namespace cpu
    {
        /// @brief General matrix-matrix product (double-precision) with explicit transpose flags, wrapper to BLAS function dgemm.
        ///
        /// Performs C = alpha * op(A) * op(B) + beta * C where op(X) is either X or X^T
        /// depending on the transpose flags `Ta` and `Tb` ("N" = no-transpose, "T" = transpose).
        ///
        /// @param Ta   Pointer to a char indicating whether matrix A is transposed: "N" (no transpose), "T" (transpose).
        /// @param Tb   Pointer to a char indicating whether matrix B is transposed: "N" (no transpose), "T" (transpose).
        /// @param m    Number of rows of op(A) and C.
        /// @param n    Number of columns of op(B) and C.
        /// @param k    Number of columns of op(A) / rows of op(B).
        /// @param alpha Scaling factor for the product op(A)*op(B).
        /// @param a    Pointer to matrix A data.
        /// @param b    Pointer to matrix B data.
        /// @param beta Scaling factor for the existing contents of C.
        /// @param c    Pointer to output matrix C data (destination / input-output).
        void MatrixMatrixProduct(const char *Ta, const char *Tb, const size_t m, const size_t n, const size_t k,
                                 const double alpha, const double *a, const double *b, const double beta, double *c);
    void MatrixMatrixProduct(const size_t m, const size_t n, const size_t k, const double* a, const double* b, double* c,
                             const double alpha = 1.0 , const double beta = 0.0, const char* Ta = "N", const char* Tb = "N");

        /// @brief General matrix-matrix product (single-precision) with explicit transpose flags, wrapper to BLAS function sgemm.
        ///
        /// Performs C = alpha * op(A) * op(B) + beta * C where op(X) is either X or X^T
        /// depending on the transpose flags `Ta` and `Tb` ("N" = no-transpose, "T" = transpose).
        /// Single-precision variant of the double-precision overload.
        ///
        /// @param Ta   Pointer to a char indicating whether matrix A is transposed: "N" (no transpose), "T" (transpose).
        /// @param Tb   Pointer to a char indicating whether matrix B is transposed: "N" (no transpose), "T" (transpose).
        /// @param m    Number of rows of op(A) and C.
        /// @param n    Number of columns of op(B) and C.
        /// @param k    Number of columns of op(A) / rows of op(B).
        /// @param alpha Scaling factor for the product op(A)*op(B).
        /// @param a    Pointer to matrix A data (float precision).
        /// @param b    Pointer to matrix B data (float precision).
        /// @param beta Scaling factor for the existing contents of C.
        /// @param c    Pointer to output matrix C data (float precision).
        void MatrixMatrixProduct(const char *Ta, const char *Tb, const size_t m, const size_t n, const size_t k,
                                 const float alpha, const float *a, const float *b, const float beta, float *c);
    void MatrixMatrixProduct(const size_t m, const size_t n, const size_t k, const float* a, const float* b, float* c,
                             const float alpha = 1.0 , const float beta = 0.0, const char* Ta = "N", const char* Tb = "N");

        /// @brief Symmetric matrix-matrix product (double-precision) with explicit side, wrapper to BLAS function dsymm.
        ///
        /// Performs C = alpha * A * B + beta * C or C = alpha * B * A + beta * C depending on `side`.
        /// A is assumed symmetric and stored in lower triangular packed format.
        ///
        /// @param side  Which side the symmetric matrix A appears on (CblasLeft or CblasRight).
        /// @param m     Number of rows of C.
        /// @param n     Number of columns of C.
        /// @param alpha Scaling factor for the product involving A.
        /// @param a     Pointer to symmetric matrix A data.
        /// @param b     Pointer to matrix B data.
        /// @param beta  Scaling factor for the existing contents of C.
        /// @param c     Pointer to output matrix C data (destination / input-output).
        void SymMatrixMatrixProduct(const CBLAS_SIDE side, const size_t m, const size_t n, const double alpha, const double *a,
                                    const double *b, const double beta, double *c);
    void SymMatrixMatrixProduct(const size_t m, const size_t n, const double* a, const double* b, double* c,
                                const double alpha = 1.0 , const double beta = 0.0, const CBLAS_SIDE side = CblasLeft);

        /// @brief Symmetric matrix-matrix product (single-precision) with explicit side, wrapper to BLAS function ssymm.
        ///
        /// Performs C = alpha * A * B + beta * C when side == CblasLeft,
        /// or C = alpha * B * A + beta * C when side == CblasRight. A is assumed symmetric and stored
        /// in lower triangular packed format.
        /// Single-precision variant of the double-precision symmetric multiply.
        ///
        /// @param side  Which side the symmetric matrix A appears on (CblasLeft or CblasRight).
        /// @param m     Number of rows of C.
        /// @param n     Number of columns of C.
        /// @param alpha Scaling factor for the product involving A.
        /// @param a     Pointer to symmetric matrix A data (stored in lower triangular packed format). If side == CblasLeft, A has shape (m x m); if side == CblasRight, A has shape (n x n).
        /// @param b     Pointer to matrix B data, shape (m x n).
        /// @param beta  Scaling factor for the existing contents of C.
        /// @param c     Pointer to output matrix C data (destination / input-output), shape (m x n).
        void SymMatrixMatrixProduct(const CBLAS_SIDE side, const size_t m, const size_t n, const float alpha, const float *a,
                                    const float *b, const float beta, float *c);
    void SymMatrixMatrixProduct(const size_t m, const size_t n, const float* a, const float* b, float* c,
                                const float alpha = 1.0 , const float beta = 0.0, const CBLAS_SIDE side = CblasLeft);
    }
}