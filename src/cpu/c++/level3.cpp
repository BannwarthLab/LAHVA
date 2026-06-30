/// @file level3.cpp
/// @brief CPU C++ template implementations of Level-3 BLAS operations.
///
/// Provides C++ template functions for Level-3 BLAS matrix-matrix operations including
/// matrix multiplication. Supports optional transposition and scalar factors. Functions
/// accept Matrix<T> types and forward to lower-level pointer-based CBLAS GEMM routines.

#include "../utils/utils.hpp"
#include "impl/blas/cpu/level3.h"
#include "impl/blas/cpu/level3.hpp"
#include "linalg.hpp"

namespace lahva
{
    namespace cpu
    {
        
    /// @brief C++-style wrapper that multiplies two matrices stored in Matrix_<T> and
    ///        forwards the call to the lower-level pointer-based routine.
    ///
    /// This template adapts the C-style MatrixMatrixProduct that takes raw pointers
    /// into a convenient Matrix_<T>-based interface. It supports real and complex
    /// element types (T == double/float/complex_double/complex_float).
    ///
    /// @tparam T Numeric element type of matrices (matches Matrix_<T> template).
    /// @param Ta    Transpose option for A: "N" (no transpose, A), "T" (transpose, A^T), "C" (conjugate-transpose, A^H).
    /// @param Tb    Transpose option for B: "N" (no transpose, B), "T" (transpose, B^T), "C" (conjugate-transpose, B^H).
    /// @param alpha Scaling factor applied to op(A)*op(B).
    /// @param a     Left-hand input matrix (Matrix_<T>).
    /// @param b     Right-hand input matrix (Matrix_<T>).
    /// @param beta  Scaling factor applied to existing contents of c.
    /// @param c     Output matrix (Matrix_<T>) which receives the result.
        template<typename T>
        void MatrixMatrixProduct(const char *Ta, const char *Tb, const T alpha, const Matrix_<T> &a, const Matrix_<T> &b,
                                 const T beta, Matrix_<T> &c)
        {
            CBLAS_TRANSPOSE transa = get_trans(Ta);
            CBLAS_TRANSPOSE transb = get_trans(Tb);
            int m, n, k;
            std::tie(m, n, k) = check_size_mm(a, b, c, transa, transb);
            MatrixMatrixProduct(Ta, Tb, m, n, k, alpha, a.data(), b.data(), beta, c.data());
        };

        template void MatrixMatrixProduct(const char *Ta, const char *Tb, const double alpha, const Matrix_<double> &a, const Matrix_<double> &b,
                                 const double beta, Matrix_<double> &c);
        template void MatrixMatrixProduct(const char *Ta, const char *Tb, const float alpha, const Matrix_<float> &a, const Matrix_<float> &b,
                                 const float beta, Matrix_<float> &c); 
        template void MatrixMatrixProduct(const char *Ta, const char *Tb, const complex_double alpha, const Matrix_<complex_double> &a, const Matrix_<complex_double> &b,
                                 const complex_double beta, Matrix_<complex_double> &c);
        template void MatrixMatrixProduct(const char *Ta, const char *Tb, const complex_float alpha, const Matrix_<complex_float> &a, const Matrix_<complex_float> &b,
                                 const complex_float beta, Matrix_<complex_float> &c);

    /// @brief Symmetric matrix-matrix multiply (double precision).
    ///
    /// Performs C = alpha * A * B + beta * C when `side == CblasLeft`, or
    /// C = alpha * B * A + beta * C when `side == CblasRight`. Matrix A is
    /// assumed to be symmetric and stored according to the project's triangular
    /// convention (see `tri` in `const.h`). This wrapper adapts Matrix_<double>
    /// to the cblas_dsymm call.
    ///
    /// @param side  Side on which the symmetric matrix A appears (CblasLeft/CblasRight).
    /// @param alpha Scaling factor for the product involving A.
    /// @param a     Symmetric matrix A (Matrix_<double>).
    /// @param b     Matrix B (Matrix_<double>).
    /// @param beta  Scaling factor for existing contents of C.
    /// @param c     Output matrix C (Matrix_<double>), receives the result.
        void SymMatrixMatrixProduct(const CBLAS_SIDE side, const double alpha, const Matrix_<double> &a, const Matrix_<double> &b,
                                    const double beta, Matrix_<double> &c)
        {
            int m, n, k;
            std::tie(m, n, k) = check_size_mm(a, b, c);
            BLAS_INT lda = get_leading(m, k);
            BLAS_INT ldb = get_leading(k, n);
            BLAS_INT ldc = get_leading(m, n);
            cblas_dsymm(major, side, tri, m, n, alpha, a.data(), lda, b.data(), ldb, beta, c.data(), ldc);
        };

    /// @brief Symmetric matrix-matrix multiply (single precision).
    ///
    /// Float variant of SymMatrixMatrixProduct. Performs the same operation
    /// as the double-precision variant but calls cblas_ssymm under the hood.
    ///
    /// @param side  Side on which the symmetric matrix A appears (CblasLeft/CblasRight).
    /// @param alpha Scaling factor for the product involving A (float).
    /// @param a     Symmetric matrix A (Matrix_<float>).
    /// @param b     Matrix B (Matrix_<float>).
    /// @param beta  Scaling factor for existing contents of C (float).
    /// @param c     Output matrix C (Matrix_<float>), receives the result.
        void SymMatrixMatrixProduct(const CBLAS_SIDE side, const float alpha, const Matrix_<float> &a, const Matrix_<float> &b,
                                    const float beta, Matrix_<float> &c)
        {
            int m, n, k;
            std::tie(m, n, k) = check_size_mm(a, b, c);

            BLAS_INT lda = get_leading(m, k);
            BLAS_INT ldb = get_leading(k, n);
            BLAS_INT ldc = get_leading(m, n);

            cblas_ssymm(major, side, tri, m, n, alpha, a.data(), lda, b.data(), ldb, beta, c.data(), ldc);
        };

    } // namespace cpu
} // namespace lahva
