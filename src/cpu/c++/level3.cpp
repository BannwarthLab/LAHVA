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

        void MatrixMatrixProduct(const char *Ta, const char *Tb, const double alpha, const Matrix_<double> &a,
                                 const Matrix_<double> &b, const double beta, Matrix_<double> &c)
        {
            CBLAS_TRANSPOSE transa = get_trans(Ta);
            CBLAS_TRANSPOSE transb = get_trans(Tb);

            int m, n, k;
            std::tie(m, n, k) = check_size_mm(a, b, c, transa, transb);

            BLAS_INT lda = get_leading(m, k);
            BLAS_INT ldb = get_leading(k, n);
            BLAS_INT ldc = get_leading(m, n);

            cblas_dgemm(major, transa, transb, m, n, k, alpha, a.data(), lda, b.data(), ldb, beta, c.data(), ldc);
        };

        void MatrixMatrixProduct(const Matrix_<double> &a, const Matrix_<double> &b, Matrix_<double> &c,
                                 const double alpha, const double beta, const char *Ta, const char *Tb)
        {
            CBLAS_TRANSPOSE transa = get_trans(Ta);
            CBLAS_TRANSPOSE transb = get_trans(Tb);

            int m, n, k;
            std::tie(m, n, k) = check_size_mm(a, b, c, transa, transb);

            BLAS_INT lda = get_leading(m, k);
            BLAS_INT ldb = get_leading(k, n);
            BLAS_INT ldc = get_leading(m, n);

            cblas_dgemm(major, transa, transb, m, n, k, alpha, a.data(), lda, b.data(), ldb, beta, c.data(), ldc);
        };

        void MatrixMatrixProduct(const char *Ta, const char *Tb, const float alpha, const Matrix_<float> &a, const Matrix_<float> &b,
                                 const float beta, Matrix_<float> &c)
        {
            CBLAS_TRANSPOSE transa = get_trans(Ta);
            CBLAS_TRANSPOSE transb = get_trans(Tb);

            int m, n, k;
            std::tie(m, n, k) = check_size_mm(a, b, c, transa, transb);

            BLAS_INT lda = get_leading(m, k);
            BLAS_INT ldb = get_leading(k, n);
            BLAS_INT ldc = get_leading(m, n);

            cblas_sgemm(major, transa, transb, m, n, k, alpha, a.data(), lda, b.data(), ldb, beta, c.data(), ldc);
        };

        void MatrixMatrixProduct(const Matrix_<float> &a, const Matrix_<float> &b, Matrix_<float> &c,
                                 const float alpha, const float beta, const char *Ta, const char *Tb)
        {
            CBLAS_TRANSPOSE transa = get_trans(Ta);
            CBLAS_TRANSPOSE transb = get_trans(Tb);

            int m, n, k;
            std::tie(m, n, k) = check_size_mm(a, b, c, transa, transb);

            BLAS_INT lda = get_leading(m, k);
            BLAS_INT ldb = get_leading(k, n);
            BLAS_INT ldc = get_leading(m, n);

            cblas_sgemm(major, transa, transb, m, n, k, alpha, a.data(), lda, b.data(), ldb, beta, c.data(), ldc);
        };

        void MatrixMatrixProduct(const char *Ta, const char *Tb, const complex_double alpha, const Matrix_<complex_double> &a,
                                 const Matrix_<complex_double> &b, const complex_double beta, Matrix_<complex_double> &c)
        {
            CBLAS_TRANSPOSE transa = get_trans(Ta);
            CBLAS_TRANSPOSE transb = get_trans(Tb);

            int m, n, k;
            std::tie(m, n, k) = check_size_mm(a, b, c, transa, transb);

            BLAS_INT lda = get_leading(m, k);
            BLAS_INT ldb = get_leading(k, n);
            BLAS_INT ldc = get_leading(m, n);

            cblas_zgemm(major, transa, transb, m, n, k, &alpha, a.data(), lda, b.data(), ldb, &beta, c.data(), ldc);
        };

        void MatrixMatrixProduct(const Matrix_<complex_double> &a, const Matrix_<complex_double> &b, Matrix_<complex_double> &c,
                                 const complex_double alpha, const complex_double beta, const char *Ta, const char *Tb)
        {
            CBLAS_TRANSPOSE transa = get_trans(Ta);
            CBLAS_TRANSPOSE transb = get_trans(Tb);

            int m, n, k;
            std::tie(m, n, k) = check_size_mm(a, b, c, transa, transb);

            BLAS_INT lda = get_leading(m, k);
            BLAS_INT ldb = get_leading(k, n);
            BLAS_INT ldc = get_leading(m, n);

            cblas_zgemm(major, transa, transb, m, n, k, &alpha, a.data(), lda, b.data(), ldb, &beta, c.data(), ldc);
        };

        void MatrixMatrixProduct(const char *Ta, const char *Tb, const complex_float alpha, const Matrix_<complex_float> &a, 
                                 const Matrix_<complex_float> &b, const complex_float beta, Matrix_<complex_float> &c)
        {
            CBLAS_TRANSPOSE transa = get_trans(Ta);
            CBLAS_TRANSPOSE transb = get_trans(Tb);

            int m, n, k;
            std::tie(m, n, k) = check_size_mm(a, b, c, transa, transb);

            BLAS_INT lda = get_leading(m, k);
            BLAS_INT ldb = get_leading(k, n);
            BLAS_INT ldc = get_leading(m, n);

            cblas_cgemm(major, transa, transb, m, n, k, &alpha, a.data(), lda, b.data(), ldb, &beta, c.data(), ldc);
        };

        void MatrixMatrixProduct(const Matrix_<complex_float> &a, const Matrix_<complex_float> &b, Matrix_<complex_float> &c,
                                 const complex_float alpha, const complex_float beta, const char *Ta, const char *Tb)
        {
            CBLAS_TRANSPOSE transa = get_trans(Ta);
            CBLAS_TRANSPOSE transb = get_trans(Tb);

            int m, n, k;
            std::tie(m, n, k) = check_size_mm(a, b, c, transa, transb);

            BLAS_INT lda = get_leading(m, k);
            BLAS_INT ldb = get_leading(k, n);
            BLAS_INT ldc = get_leading(m, n);

            cblas_cgemm(major, transa, transb, m, n, k, &alpha, a.data(), lda, b.data(), ldb, &beta, c.data(), ldc);
        };

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

        void SymMatrixMatrixProduct(const Matrix_<double> &a, const Matrix_<double> &b, Matrix_<double> &c,
                                    const double alpha, const double beta, const CBLAS_SIDE side)
        {
            int m, n, k;
            std::tie(m, n, k) = check_size_mm(a, b, c);
            BLAS_INT lda = get_leading(m, k);
            BLAS_INT ldb = get_leading(k, n);
            BLAS_INT ldc = get_leading(m, n);
            cblas_dsymm(major, side, tri, m, n, alpha, a.data(), lda, b.data(), ldb, beta, c.data(), ldc);
        };

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

        void SymMatrixMatrixProduct(const Matrix_<float> &a, const Matrix_<float> &b, Matrix_<float> &c,
                                    const float alpha, const float beta, const CBLAS_SIDE side)
        {
            int m, n, k;
            std::tie(m, n, k) = check_size_mm(a, b, c);

            BLAS_INT lda = get_leading(m, k);
            BLAS_INT ldb = get_leading(k, n);
            BLAS_INT ldc = get_leading(m, n);

            cblas_ssymm(major, side, tri, m, n, alpha, a.data(), lda, b.data(), ldb, beta, c.data(), ldc);
        };
    }
} // namespace lahva
