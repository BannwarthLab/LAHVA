#include "impl/blas/cpu/level3.h"
#include "../utils/utils.hpp"

namespace lahva
{
    namespace cpu
    {
        /// @brief Type-generic pointer-based matrix-matrix multiply dispatcher.
        ///
        /// Dispatches to the appropriate BLAS GEMM routine depending on the
        /// element type T. Supported types: double -> dgemm, float -> sgemm,
        /// complex_double -> zgemm, complex_float -> cgemm.
        ///
        /// @tparam T Element type of matrices (double/float/complex_double/complex_float).
        /// @param Ta Transpose option for A ("N","T","C").
        /// @param Tb Transpose option for B ("N","T","C").
        /// @param m Number of rows of op(A) and C.
        /// @param n Number of columns of op(B) and C.
        /// @param k Number of columns of op(A) / rows of op(B).
        /// @param alpha Scaling factor for op(A)*op(B).
        /// @param a Pointer to matrix A data.
        /// @param b Pointer to matrix B data.
        /// @param beta Scaling factor for existing contents of C.
        /// @param c Pointer to output matrix C data (destination / input-output).
        template <typename T>
        void MatrixMatrixProductMeta(const char *Ta, const char *Tb, const size_t m, const size_t n, const size_t k,
                                     const T alpha, const T *a, const T *b, const T beta, T *c)
        {
            CBLAS_TRANSPOSE transa = get_trans(Ta);
            CBLAS_TRANSPOSE transb = get_trans(Tb);

            BLAS_INT m_ = m;
            BLAS_INT n_ = n;
            BLAS_INT k_ = k;

            BLAS_INT lda = get_leading(m_, k_);
            BLAS_INT ldb = get_leading(k_, n_);
            BLAS_INT ldc = get_leading(m_, n_);

            if constexpr (std::is_same_v<T, double>)
            {
                cblas_dgemm(major, transa, transb, m_, n_, k_, alpha, a, lda, b, ldb, beta, c, ldc);
            }
            else if constexpr (std::is_same_v<T, float>)
            {
                cblas_sgemm(major, transa, transb, m_, n_, k_, alpha, a, lda, b, ldb, beta, c, ldc);
            }
            else if constexpr (std::is_same_v<T, complex_double>)
            {
                cblas_zgemm(major, transa, transb, m_, n_, k_, &alpha, a, lda, b, ldb, &beta, c, ldc);
            }
            else if constexpr (std::is_same_v<T, complex_float>)
            {
                cblas_cgemm(major, transa, transb, m_, n_, k_, &alpha, a, lda, b, ldb, &beta, c, ldc);
            }
        };

        template void MatrixMatrixProductMeta(const char *Ta, const char *Tb, const size_t m, const size_t n, const size_t k,
                                              const double alpha, const double *a, const double *b, const double beta, double *c);
        template void MatrixMatrixProductMeta(const char *Ta, const char *Tb, const size_t m, const size_t n, const size_t k,
                                              const float alpha, const float *a, const float *b, const float beta, float *c);
        template void MatrixMatrixProductMeta(const char *Ta, const char *Tb, const size_t m, const size_t n, const size_t k,
                                              const complex_double alpha, const complex_double *a, const complex_double *b, const complex_double beta, complex_double *c);
        template void MatrixMatrixProductMeta(const char *Ta, const char *Tb, const size_t m, const size_t n, const size_t k,
                                              const complex_float alpha, const complex_float *a, const complex_float *b, const complex_float beta, complex_float *c);

        /// @brief Single-precision pointer-based GEMM entry point.
        ///
        /// Calls the templated dispatcher which forwards to cblas_sgemm. Parameters
        /// follow BLAS conventions: Ta/Tb specify transpose options, alpha/beta are
        /// scalars, and a/b/c are pointers to matrices in the project storage layout.
        ///
        /// @param Ta Transpose option for A ("N","T","C").
        /// @param Tb Transpose option for B ("N","T","C").
        /// @param m Number of rows of op(A) and C.
        /// @param n Number of columns of op(B) and C.
        /// @param k Number of columns of op(A) / rows of op(B).
        /// @param alpha Scalar multiplier for op(A)*op(B).
        /// @param a Pointer to matrix A data (float).
        /// @param b Pointer to matrix B data (float).
        /// @param beta Scalar multiplier for existing contents of C (float).
        /// @param c Pointer to output matrix C data (float).
        void MatrixMatrixProduct(const char *Ta, const char *Tb, const size_t m, const size_t n, const size_t k, const float alpha,
                                 const float *a, const float *b, const float beta, float *c)
        {
            MatrixMatrixProductMeta(Ta, Tb, m, n, k, alpha, a, b, beta, c);
        };

        /// @brief Double-precision pointer-based GEMM entry point.
        ///
        /// Forwards to the templated dispatcher which will call cblas_dgemm.
        ///
        /// @param Ta Transpose option for A ("N","T","C").
        /// @param Tb Transpose option for B ("N","T","C").
        /// @param m Number of rows of op(A) and C.
        /// @param n Number of columns of op(B) and C.
        /// @param k Number of columns of op(A) / rows of op(B).
        /// @param alpha Scalar multiplier for op(A)*op(B).
        /// @param a Pointer to matrix A data (double).
        /// @param b Pointer to matrix B data (double).
        /// @param beta Scalar multiplier for existing contents of C (double).
        /// @param c Pointer to output matrix C data (double).
        void MatrixMatrixProduct(const char *Ta, const char *Tb, const size_t m, const size_t n, const size_t k, const double alpha,
                                 const double *a, const double *b, const double beta, double *c)
        {
            MatrixMatrixProductMeta(Ta, Tb, m, n, k, alpha, a, b, beta, c);
        };

        /// @brief Complex-double pointer-based GEMM entry point (zgemm wrapper).
        ///
        /// For complex types BLAS typically expects scalar pointers; the dispatcher
        /// handles passing addresses where needed.
        ///
        /// @param Ta Transpose option for A ("N","T","C").
        /// @param Tb Transpose option for B ("N","T","C").
        /// @param m Number of rows of op(A) and C.
        /// @param n Number of columns of op(B) and C.
        /// @param k Number of columns of op(A) / rows of op(B).
        /// @param alpha Complex scalar multiplier for op(A)*op(B).
        /// @param a Pointer to complex-double matrix A data.
        /// @param b Pointer to complex-double matrix B data.
        /// @param beta Complex scalar multiplier for existing contents of C.
        /// @param c Pointer to complex-double output matrix C data.
        void MatrixMatrixProduct(const char *Ta, const char *Tb, const size_t m, const size_t n, const size_t k, const complex_double alpha,
                                 const complex_double *a, const complex_double *b, const complex_double beta, complex_double *c)
        {
            MatrixMatrixProductMeta(Ta, Tb, m, n, k, alpha, a, b, beta, c);
        };

        /// @brief Complex-float pointer-based GEMM entry point (cgemm wrapper).
        ///
        /// See complex-double variant for parameter meanings.
        void MatrixMatrixProduct(const char *Ta, const char *Tb, const size_t m, const size_t n, const size_t k,
                                 const complex_float alpha, const complex_float *a, const complex_float *b, const complex_float beta, complex_float *c)
        {
            MatrixMatrixProductMeta(Ta, Tb, m, n, k, alpha, a, b, beta, c);
        };

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
    }
}
