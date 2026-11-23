#include "impl/blas/cpu/level3.h"
/// @file level3.cpp
/// @brief CPU C-level Level-3 BLAS function implementations.
///
/// Low-level C implementations of Level-3 BLAS operations (matrix-matrix multiplication)
/// with runtime type dispatching to appropriate GEMM routines (dgemm, sgemm, etc.).

#include "../utils/utils.hpp"

namespace lahva
{
    namespace cpu
    {
    void MatrixMatrixProduct(const char* Ta, const char* Tb, const size_t m, const size_t n, const size_t k, 
                             const double alpha, const double* a, const double* b, const double beta, double* c) 
        {
            CBLAS_TRANSPOSE transa = get_trans(Ta);
            CBLAS_TRANSPOSE transb = get_trans(Tb);

            BLAS_INT m_ = m;
            BLAS_INT n_ = n;
            BLAS_INT k_ = k;

        BLAS_INT lda = get_leading(m_, k_);
        BLAS_INT ldb = get_leading(k_, n_);
            BLAS_INT ldc = get_leading(m_, n_);

                cblas_dgemm(major, transa, transb, m_, n_, k_, alpha, a, lda, b, ldb, beta, c, ldc);
        };

    void MatrixMatrixProduct(const size_t m, const size_t n, const size_t k, const double* a, const double* b, double* c,
                             const double alpha, const double beta, const char* Ta, const char* Tb)
        {
        CBLAS_TRANSPOSE transa = get_trans(Ta);
        CBLAS_TRANSPOSE transb = get_trans(Tb);

        BLAS_INT m_ = m;
        BLAS_INT n_ = n;
        BLAS_INT k_ = k;

        BLAS_INT lda = get_leading(m_, k_);
        BLAS_INT ldb = get_leading(k_, n_);
        BLAS_INT ldc = get_leading(m_, n_);

        cblas_dgemm(major, transa, transb, m_, n_, k_, alpha, a, lda, b, ldb, beta, c, ldc);
        };


    void MatrixMatrixProduct(const char* Ta, const char* Tb, const size_t m, const size_t n, const size_t k, 
                             const float alpha, const float* a, const float* b, const float beta, float* c)
        {
        CBLAS_TRANSPOSE transa = get_trans(Ta);
        CBLAS_TRANSPOSE transb = get_trans(Tb);

        BLAS_INT m_ = m;
        BLAS_INT n_ = n;
        BLAS_INT k_ = k;

        BLAS_INT lda = get_leading(m_, k_);
        BLAS_INT ldb = get_leading(k_, n_);
        BLAS_INT ldc = get_leading(m_, n_);

        cblas_sgemm(major, transa, transb, m_, n_, k_, alpha, a, lda, b, ldb, beta, c, ldc);
        };

    void MatrixMatrixProduct(const size_t m, const size_t n, const size_t k, const float* a, const float* b, float* c,
                             const float alpha, const float beta, const char* Ta, const char* Tb)
        {
        CBLAS_TRANSPOSE transa = get_trans(Ta);
        CBLAS_TRANSPOSE transb = get_trans(Tb);

        BLAS_INT m_ = m;
        BLAS_INT n_ = n;
        BLAS_INT k_ = k;

        BLAS_INT lda = get_leading(m_, k_);
        BLAS_INT ldb = get_leading(k_, n_);
        BLAS_INT ldc = get_leading(m_, n_);

        cblas_sgemm(major, transa, transb, m_, n_, k_, alpha, a, lda, b, ldb, beta, c, ldc);
        };

        /// @brief Symmetric matrix-matrix multiply (double-precision) pointer-based implementation.
        ///
        /// Performs C = alpha * A * B + beta * C when `side == CblasLeft`, or
        /// C = alpha * B * A + beta * C when `side == CblasRight`. Matrix A is
        /// assumed to be symmetric and stored in triangular format.
        ///
        /// @param side Side on which the symmetric matrix A appears (CblasLeft/CblasRight).
        /// @param m Number of rows of output matrix C.
        /// @param n Number of columns of output matrix C.
        /// @param alpha Scaling factor for the product involving A (double precision).
        /// @param a Pointer to symmetric matrix A data (double precision).
        /// @param b Pointer to matrix B data (double precision).
        /// @param beta Scaling factor for existing contents of C (double precision).
        /// @param c Pointer to output matrix C data (double precision, overwritten).
        void SymMatrixMatrixProduct(const CBLAS_SIDE side, const size_t m, const size_t n, const double alpha, const double *a,
                                    const double *b, const double beta, double *c)
        {
            BLAS_INT m_ = m;
            BLAS_INT n_ = n;
            BLAS_INT lda, ldb, ldc;

            if (side == CblasLeft)
            {
                lda = get_leading(m_, m_);
                ldb = get_leading(m_, n_);
                ldc = get_leading(m_, n_);
            }
            else
            {
                lda = get_leading(n_, n_);
                ldb = get_leading(m_, n_);
                ldc = get_leading(m_, n_);
            }

            cblas_dsymm(major, side, tri, m_, n_, alpha, a, lda, b, ldb, beta, c, ldc);
        };
    void SymMatrixMatrixProduct(const size_t m, const size_t n, const double* a, const double* b, double* c,
                                const double alpha, const double beta, const CBLAS_SIDE side)
    {

        BLAS_INT m_ = m;
        BLAS_INT n_ = n;
        BLAS_INT lda, ldb, ldc;

        if (side == CblasLeft)
        {
            lda = get_leading(m_, m_);
            ldb = get_leading(m_, n_);
            ldc = get_leading(m_, n_);
        }
        else
        {
            lda = get_leading(n_, n_);
            ldb = get_leading(m_, n_);
            ldc = get_leading(m_, n_); 
        }

        cblas_dsymm(major, side, tri, m_, n_, alpha, a, lda, b, ldb, beta, c, ldc);
    };

        /// @brief Symmetric matrix-matrix multiply (single-precision) pointer-based implementation.
        ///
        /// Performs C = alpha * A * B + beta * C when `side == CblasLeft`, or
        /// C = alpha * B * A + beta * C when `side == CblasRight`. Matrix A is
        /// assumed to be symmetric and stored in triangular format.
        ///
        /// @param side Side on which the symmetric matrix A appears (CblasLeft/CblasRight).
        /// @param m Number of rows of output matrix C.
        /// @param n Number of columns of output matrix C.
        /// @param alpha Scaling factor for the product involving A (single precision).
        /// @param a Pointer to symmetric matrix A data (single precision).
        /// @param b Pointer to matrix B data (single precision).
        /// @param beta Scaling factor for existing contents of C (single precision).
        /// @param c Pointer to output matrix C data (single precision, overwritten).
        void SymMatrixMatrixProduct(const CBLAS_SIDE side, const size_t m, const size_t n, const float alpha, const float *a,
                                    const float *b, const float beta, float *c)
        {
            BLAS_INT m_ = m;
            BLAS_INT n_ = n;
            BLAS_INT lda, ldb, ldc;

            if (side == CblasLeft)
            {
                lda = get_leading(m_, m_);
                ldb = get_leading(m_, n_);
                ldc = get_leading(m_, n_);
            }
            else
            {
                lda = get_leading(n_, n_);
                ldb = get_leading(m_, n_);
                ldc = get_leading(m_, n_);
            }

            cblas_ssymm(major, side, tri, m_, n_, alpha, a, lda, b, ldb, beta, c, ldc);
        };
    void SymMatrixMatrixProduct(const size_t m, const size_t n, const float* a, const float* b, float* c,
                                const float alpha, const float beta, const CBLAS_SIDE side)
    {
        BLAS_INT m_ = m;
        BLAS_INT n_ = n;
        BLAS_INT lda, ldb, ldc;

        if (side == CblasLeft)
        {
            lda = get_leading(m_, m_);
            ldb = get_leading(m_, n_);
            ldc = get_leading(m_, n_);
        }
        else
        {
            lda = get_leading(n_, n_);
            ldb = get_leading(m_, n_);
            ldc = get_leading(m_, n_); 
        }

        cblas_ssymm(major, side, tri, m_, n_, alpha, a, lda, b, ldb, beta, c, ldc);
    }; 
    } // namespace cpu
} // namespace lahva
