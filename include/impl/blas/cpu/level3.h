#ifndef LAHVA_C_LEVEL_3_H
#define LAHVA_C_LEVEL_3_H
#include "const.h"

// Level-3 BLAS-like operations (matrix * matrix) - C-style declarations
// These functions are thin wrappers around optimized CPU BLAS kernels. They operate on
// raw pointers and use simple scalar parameters for flexibility. Each overload is
// provided for double and float precision; both precisions are documented explicitly.

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

        /// @brief General matrix-matrix product (double-precision) with defaults, wrapper to BLAS function dgemm.
        ///
        /// Performs C = alpha * op(A) * op(B) + beta * C with default alpha = 1.0, beta = 0.0,
        /// and no transposition (op(A) = A, op(B) = B).
        ///
        /// @param m    Number of rows of op(A) and C.
        /// @param n    Number of columns of op(B) and C.
        /// @param k    Number of columns of op(A) / rows of op(B).
        /// @param a    Pointer to matrix A data.
        /// @param b    Pointer to matrix B data.
        /// @param c    Pointer to output matrix C data (destination / input-output).
        /// @param alpha Scaling factor for the product op(A)*op(B). Default: 1.0
        /// @param beta  Scaling factor for the existing contents of C. Default: 0.0
        /// @param Ta    Transpose flag for A: "N" (no transpose), "T" (transpose). Default: "N"
        /// @param Tb    Transpose flag for B: "N" (no transpose), "T" (transpose). Default: "N"
        inline void MatrixMatrixProduct(const size_t m, const size_t n, const size_t k, const double *a, const double *b, double *c,
                                        const double alpha = 1.0, const double beta = 0.0, const char *Ta = "N", const char *Tb = "N")
        {
            MatrixMatrixProduct(Ta, Tb, m, n, k, alpha, a, b, beta, c);
        };

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

        /// @brief General matrix-matrix product (single-precision) with defaults, wrapper to BLAS function sgemm.
        ///
        /// Performs C = alpha * op(A) * op(B) + beta * C with default alpha = 1.0f, beta = 0.0f,
        /// and no transposition (op(A) = A, op(B) = B).
        ///
        /// @param m    Number of rows of op(A) and C.
        /// @param n    Number of columns of op(B) and C.
        /// @param k    Number of columns of op(A) / rows of op(B).
        /// @param a    Pointer to matrix A data (float).
        /// @param b    Pointer to matrix B data (float).
        /// @param c    Pointer to output matrix C data (float) (destination / input-output).
        /// @param alpha Scaling factor for the product op(A)*op(B). Default: 1.0f
        /// @param beta  Scaling factor for the existing contents of C. Default: 0.0f
        /// @param Ta    Transpose flag for A: "N" (no transpose), "T" (transpose). Default: "N"
        /// @param Tb    Transpose flag for B: "N" (no transpose), "T" (transpose). Default: "N"
        inline void MatrixMatrixProduct(const size_t m, const size_t n, const size_t k, const float *a, const float *b, float *c,
                                        const float alpha = 1.0, const float beta = 0.0, const char *Ta = "N", const char *Tb = "N")
        {
            MatrixMatrixProduct(Ta, Tb, m, n, k, alpha, a, b, beta, c);
        };

        /// @brief General matrix-matrix product (double-precision complex) with explicit transpose flags.
        ///
        /// Complex double-precision variant (wrapper to zgemm). Performs
        /// C = alpha * op(A) * op(B) + beta * C where op(X) is either X or X^H
        /// depending on the transpose flags `Ta` and `Tb` ("N" = no-transpose,
        /// "T" = transpose, "C" = conjugate-transpose). Uses `complex_double`.
        ///
        /// @param Ta   Transpose option for A: "N" (no transpose, A), "T" (transpose, A^T), "C" (conjugate-transpose, A^H).
        /// @param Tb   Transpose option for B: "N" (no transpose, B), "T" (transpose, B^T), "C" (conjugate-transpose, B^H).
        /// @param m    Number of rows of op(A) and C.
        /// @param n    Number of columns of op(B) and C.
        /// @param k    Number of columns of op(A) / rows of op(B).
        /// @param alpha Complex scaling factor for op(A)*op(B).
        /// @param a    Pointer to complex-double matrix A data.
        /// @param b    Pointer to complex-double matrix B data.
        /// @param beta  Complex scaling factor for the existing contents of C.
        /// @param c    Pointer to complex-double output matrix C data.
        void MatrixMatrixProduct(const char *Ta, const char *Tb, const size_t m, const size_t n, const size_t k,
                                 const complex_double alpha, const complex_double *a, const complex_double *b, const complex_double beta, complex_double *c);

        /// @brief General matrix-matrix product (double-precision complex) with defaults, wrapper to BLAS function zgemm.
        ///
        /// Performs C = alpha * op(A) * op(B) + beta * C with default alpha = 1.0, beta = 0.0,
        /// and no transposition (op(A) = A, op(B) = B). Complex-double variant.
        ///
        /// @param m     Number of rows of op(A) and C.
        /// @param n     Number of columns of op(B) and C.
        /// @param k     Number of columns of op(A) / rows of op(B).
        /// @param a     Pointer to complex-double matrix A data.
        /// @param b     Pointer to complex-double matrix B data.
        /// @param c     Pointer to complex-double output matrix C data (destination / input-output).
        /// @param alpha Complex scaling factor for the product op(A)*op(B). Default: 1.0
        /// @param beta  Complex scaling factor for the existing contents of C. Default: 0.0
        /// @param Ta    Transpose option for A: "N" (no transpose, A), "T" (transpose, A^T), "C" (conjugate-transpose, A^H). Default: "N"
        /// @param Tb    Transpose option for B: "N" (no transpose, B), "T" (transpose, B^T), "C" (conjugate-transpose, B^H). Default: "N"
        inline void MatrixMatrixProduct(const size_t m, const size_t n, const size_t k, const complex_double *a, const complex_double *b, complex_double *c,
                                        const complex_double alpha = 1.0, const complex_double beta = 0.0, const char *Ta = "N", const char *Tb = "N")
        {
            MatrixMatrixProduct(Ta, Tb, m, n, k, alpha, a, b, beta, c);
        };

        /// @brief General matrix-matrix product (single-precision complex) with explicit transpose flags.
        ///
        /// Performs C = alpha * op(A) * op(B) + beta * C where op(X) is either X or X^H
        /// depending on the transpose flags `Ta` and `Tb` ("N" = no-transpose, "T" = transpose,
        /// "C" = conjugate-transpose). Complex single-precision variant (wrapper to cgemm).
        ///
        /// @param Ta   Transpose option for A: "N" (no transpose, A), "T" (transpose, A^T), "C" (conjugate-transpose, A^H).
        /// @param Tb   Transpose option for B: "N" (no transpose, B), "T" (transpose, B^T), "C" (conjugate-transpose, B^H).
        /// @param m    Number of rows of op(A) and C.
        /// @param n    Number of columns of op(B) and C.
        /// @param k    Number of columns of op(A) / rows of op(B).
        /// @param alpha Complex scaling factor for op(A)*op(B).
        /// @param a    Pointer to complex-float matrix A data.
        /// @param b    Pointer to complex-float matrix B data.
        /// @param beta  Complex scaling factor for the existing contents of C.
        /// @param c    Pointer to complex-float output matrix C data.
        void MatrixMatrixProduct(const char *Ta, const char *Tb, const size_t m, const size_t n, const size_t k,
                                 const complex_float alpha, const complex_float *a, const complex_float *b, const complex_float beta, complex_float *c);

        /// @brief General matrix-matrix product (single-precision complex) with defaults, wrapper to BLAS function cgemm.
        ///
        /// Performs C = alpha * op(A) * op(B) + beta * C with default alpha = 1.0, beta = 0.0,
        /// and no transposition (op(A) = A, op(B) = B). Complex-float variant.
        ///
        /// @param m     Number of rows of op(A) and C.
        /// @param n     Number of columns of op(B) and C.
        /// @param k     Number of columns of op(A) / rows of op(B).
        /// @param a     Pointer to complex-float matrix A data.
        /// @param b     Pointer to complex-float matrix B data.
        /// @param c     Pointer to complex-float output matrix C data (destination / input-output).
        /// @param alpha Complex scaling factor for the product op(A)*op(B). Default: 1.0
        /// @param beta  Complex scaling factor for the existing contents of C. Default: 0.0
        /// @param Ta    Transpose option for A: "N" (no transpose, A), "T" (transpose, A^T), "C" (conjugate-transpose, A^H). Default: "N"
        /// @param Tb    Transpose option for B: "N" (no transpose, B), "T" (transpose, B^T), "C" (conjugate-transpose, B^H). Default: "N"
        inline void MatrixMatrixProduct(const size_t m, const size_t n, const size_t k, const complex_float *a, const complex_float *b, complex_float *c,
                                        const complex_float alpha = 1.0, const complex_float beta = 0.0, const char *Ta = "N", const char *Tb = "N")
        {
            MatrixMatrixProduct(Ta, Tb, m, n, k, alpha, a, b, beta, c);
        };

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

        /// @brief Symmetric matrix-matrix product (double-precision) with defaults, wrapper to BLAS function dsymm.
        ///
        /// Performs C = alpha * A * B + beta * C when side == CblasLeft (default),
        /// or C = alpha * B * A + beta * C when side == CblasRight. A is assumed symmetric and stored
        /// in lower triangular packed format.
        ///
        /// @param m     Number of rows of C.
        /// @param n     Number of columns of C.
        /// @param a     Pointer to symmetric matrix A data (stored in lower triangular packed format).
        /// @param b     Pointer to matrix B data. If side == CblasLeft, B has shape (n x n); if side == CblasRight, B has shape (m x m).
        /// @param c     Pointer to output matrix C data (destination / input-output), shape (m x n).
        /// @param alpha Scaling factor for the product involving A. Default: 1.0
        /// @param beta  Scaling factor for the existing contents of C. Default: 0.0
        /// @param side  Which side the symmetric matrix A appears on (CblasLeft or CblasRight). Default: CblasLeft
        inline void SymMatrixMatrixProduct(const size_t m, const size_t n, const double *a, const double *b, double *c,
                                           const double alpha = 1.0, const double beta = 0.0, const CBLAS_SIDE side = CblasLeft)
        {
            SymMatrixMatrixProduct(side, m, n, alpha, a, b, beta, c);
        };

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

        /// @brief Symmetric matrix-matrix product (single-precision) with defaults, wrapper to BLAS function ssymm.
        ///
        /// Performs C = alpha * A * B + beta * C when side == CblasLeft (default),
        /// or C = alpha * B * A + beta * C when side == CblasRight. A is assumed symmetric and stored
        /// in lower triangular packed format.
        /// Single-precision variant with default scalars.
        ///
        /// @param m     Number of rows of C.
        /// @param n     Number of columns of C.
        /// @param a     Pointer to symmetric matrix A data (stored in lower triangular packed format).
        /// @param b     Pointer to matrix B data. If side == CblasLeft, B has shape (n x n); if side == CblasRight, B has shape (m x m).
        /// @param c     Pointer to output matrix C data (destination / input-output), shape (m x n).
        /// @param alpha Scaling factor for the product involving A. Default: 1.0
        /// @param beta  Scaling factor for the existing contents of C. Default: 0.0
        /// @param side  Which side the symmetric matrix A appears on (CblasLeft or CblasRight). Default: CblasLeft
        inline void SymMatrixMatrixProduct(const size_t m, const size_t n, const float *a, const float *b, float *c,
                                           const float alpha = 1.0, const float beta = 0.0, const CBLAS_SIDE side = CblasLeft)
        {
            SymMatrixMatrixProduct(side, m, n, alpha, a, b, beta, c);
        };
    }
}
#endif