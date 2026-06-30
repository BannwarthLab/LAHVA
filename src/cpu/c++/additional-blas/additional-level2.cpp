/// @file additional-level2.cpp
/// @brief CPU C++ implementations of extended/custom Level-2 BLAS operations.
///
/// Implements additional matrix-vector and matrix-matrix operations beyond standard BLAS
/// Level-2, including matrix addition with optional transposes. Supports OpenMP parallelization
/// for multi-threaded execution.

#include "../../utils/utils.hpp"
#include "impl/blas/cpu/additional-level2.hpp"
#include "impl/blas/cpu/level1.hpp"
#include "linalg.hpp"

namespace lahva
{
    namespace cpu
    {
        /// @brief Add two double-precision matrices: C = alpha*op(A) + beta*op(B)
        ///
        /// Performs scaled matrix addition with optional transposes: C = alpha*op(A) + beta*op(B),
        /// where op(A) is A or A^T depending on the transpose flag. Matrices A and B are not modified.
        ///
        /// @param Ta Transpose option for matrix A: "N" (no transpose, A), "T" (transpose, A^T).
        /// @param Tb Transpose option for matrix B: "N" (no transpose, B), "T" (transpose, B^T).
        /// @param alpha Scalar multiplier for op(A) (double precision).
        /// @param a Input matrix A (double precision).
        /// @param beta Scalar multiplier for op(B) (double precision).
        /// @param b Input matrix B (double precision).
        /// @param c Output matrix C (overwritten with the result, double precision).
        template <>
        void AddMatrices<double>(const char *Ta, const char *Tb, const double alpha, const Matrix_<double> &a, const double beta,
                                 const Matrix_<double> &b, Matrix_<double> &c)
        {
            CBLAS_TRANSPOSE transa = get_trans(Ta);
            CBLAS_TRANSPOSE transb = get_trans(Tb);

            int m, n;
            std::tie(m, n) = check_same_shape_mm(a, b, c, transa, transb);

            BLAS_INT lda = get_leading(m, n, transa);
            BLAS_INT ldb = get_leading(m, n, transb);
            BLAS_INT ldc = get_leading(m, n);
#if defined(_MKL_H_)
            mkl_domatadd(major_char, *Ta, *Tb, m, n, alpha, a.data(), lda, beta, b.data(), ldb, c.data(), ldc);

#elif defined(OPENBLAS_VERSION)
            std::cout << "OPENBLAS_VERSION" << std::endl;
            // copy B into C, potetially transpose
            cblas_domatcopy(major, transb, b.shape().first, b.shape().second, beta, b.data(), ldb, c.data(), ldc);
            // copy A inplace, if transpose
            if (transa == CblasTrans)
            {
                Matrix<double> tmp(c.shape(), 0.0);
                cblas_domatcopy(major, transa, a.shape().first, a.shape().second, alpha, a.data(), lda, tmp.data(), ldc);
                AddVectors(1.0, tmp, c);
            }
            else
            {
                AddVectors(alpha, a, c);
            }
#elif defined(__ACCELERATE__)
            appleblas_dgeadd(major, transa, transb, m, n, alpha, a.data(), lda, beta, b.data(), ldb, c.data(), ldc);

#else
            if (transa == CblasNoTrans && transb == CblasNoTrans)
            {
#pragma omp parallel for collapse(2)
                for (int i = 0; i < m; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        c(i, j) = alpha * a(i, j) + beta * b(i, j);
                    }
                }
            }
            else if (transa == CblasNoTrans && transb == CblasTrans)
            {
#pragma omp parallel for collapse(2)
                for (int i = 0; i < m; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        c(i, j) = alpha * a(i, j) + beta * b(j, i);
                    }
                }
            }
            else if (transa == CblasTrans && transb == CblasNoTrans)
            {
#pragma omp parallel for collapse(2)
                for (int i = 0; i < m; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        c(i, j) = alpha * a(j, i) + beta * b(i, j);
                    }
                }
            }
            else
            {
#pragma omp parallel for collapse(2)
                for (int i = 0; i < m; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        c(i, j) = alpha * a(j, i) + beta * b(j, i);
                    }
                }
            }

#endif
        };

        /// @brief Add two single-precision matrices: C = alpha*op(A) + beta*op(B)
        ///
        /// Performs scaled matrix addition with optional transposes: C = alpha*op(A) + beta*op(B),
        /// where op(A) is A or A^T depending on the transpose flag. Matrices A and B are not modified.
        ///
        /// @param Ta Transpose option for matrix A: "N" (no transpose, A), "T" (transpose, A^T).
        /// @param Tb Transpose option for matrix B: "N" (no transpose, B), "T" (transpose, B^T).
        /// @param alpha Scalar multiplier for op(A) (single precision).
        /// @param a Input matrix A (single precision).
        /// @param beta Scalar multiplier for op(B) (single precision).
        /// @param b Input matrix B (single precision).
        /// @param c Output matrix C (overwritten with the result, single precision).
        template <>
        void AddMatrices<float>(const char *Ta, const char *Tb, const float alpha, const Matrix_<float> &a, const float beta,
                                const Matrix_<float> &b, Matrix_<float> &c)
        {
            CBLAS_TRANSPOSE transa = get_trans(Ta);
            CBLAS_TRANSPOSE transb = get_trans(Tb);

            int m, n;
            std::tie(m, n) = check_same_shape_mm(a, b, c, transa, transb);

            BLAS_INT lda = get_leading(m, n, transa);
            BLAS_INT ldb = get_leading(m, n, transb);
            BLAS_INT ldc = get_leading(m, n);

#if defined(_MKL_H_)
            mkl_somatadd(major_char, *Ta, *Tb, m, n, alpha, a.data(), lda, beta, b.data(), ldb, c.data(), ldc);
#elif defined(OPENBLAS_VERSION)
            // copy B into C, potetially transpose
            cblas_somatcopy(major, transb, b.shape().first, b.shape().second, beta, b.data(), ldb, c.data(), ldc);
            // copy A inplace, if transpose
            if (transa == CblasTrans)
            {
                Matrix<float> tmp(c.shape(), 0.0);
                cblas_somatcopy(major, transa, a.shape().first, a.shape().second, alpha, a.data(), lda, tmp.data(), ldc);
                AddVectors(1.0, tmp, c);
            }
            else
            {
                AddVectors(alpha, a, c);
            }
#elif defined(__ACCELERATE__)
            appleblas_sgeadd(major, transa, transb, m, n, alpha, a.data(), lda, beta, b.data(), ldb, c.data(), ldc);
#else
            if (transa == CblasNoTrans && transb == CblasNoTrans)
            {
#pragma omp parallel for collapse(2)
                for (int i = 0; i < m; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        c(i, j) = alpha * a(i, j) + beta * b(i, j);
                    }
                }
            }
            else if (transa == CblasNoTrans && transb == CblasTrans)
            {
#pragma omp parallel for collapse(2)
                for (int i = 0; i < m; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        c(i, j) = alpha * a(i, j) + beta * b(j, i);
                    }
                }
            }
            else if (transa == CblasTrans && transb == CblasNoTrans)
            {
#pragma omp parallel for collapse(2)
                for (int i = 0; i < m; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        c(i, j) = alpha * a(j, i) + beta * b(i, j);
                    }
                }
            }
            else
            {
#pragma omp parallel for collapse(2)
                for (int i = 0; i < m; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        c(i, j) = alpha * a(j, i) + beta * b(j, i);
                    }
                }
            }
#endif
        };

        /// @brief Add two double-precision matrices with default parameters: C = alpha*op(A) + beta*op(B)
        ///
        /// Convenience overload with optional transpose and scalar parameters.
        /// Performs: C = alpha*op(A) + beta*op(B) with optional transposes.
        ///
        /// @param a Input matrix A (double precision).
        /// @param b Input matrix B (double precision).
        /// @param c Output matrix C (overwritten with the result, double precision).
        /// @param alpha Scalar multiplier for op(A) (double precision). Default: 1.0.
        /// @param beta Scalar multiplier for op(B) (double precision). Default: 1.0.
        /// @param Ta Transpose option for matrix A: "N" (no transpose, A), "T" (transpose, A^T). Default: "N".
        /// @param Tb Transpose option for matrix B: "N" (no transpose, B), "T" (transpose, B^T). Default: "N".
        template <>
        void AddMatrices<double>(const Matrix_<double> &a, const Matrix_<double> &b, Matrix_<double> &c,
                                 const double alpha, const double beta, const char *Ta, const char *Tb)
        {
            CBLAS_TRANSPOSE transa = get_trans(Ta);
            CBLAS_TRANSPOSE transb = get_trans(Tb);

            int m, n;
            std::tie(m, n) = check_same_shape_mm(a, b, c, transa, transb);

            BLAS_INT lda = get_leading(m, n, transa);
            BLAS_INT ldb = get_leading(m, n, transb);
            BLAS_INT ldc = get_leading(m, n);

#if defined(_MKL_H_)
            mkl_domatadd(major_char, *Ta, *Tb, m, n, alpha, a.data(), lda, beta, b.data(), ldb, c.data(), ldc);
#elif defined(OPENBLAS_VERSION)
            // copy B into C, potetially transpose
            cblas_domatcopy(major, transb, b.shape().first, b.shape().second, beta, b.data(), ldb, c.data(), ldc);
            // copy A inplace, if transpose
            if (transa == CblasTrans)
            {
                Matrix<double> tmp(c.shape(), 0.0);
                cblas_domatcopy(major, transa, a.shape().first, a.shape().second, alpha, a.data(), lda, tmp.data(), ldc);
                AddVectors(1.0, tmp, c);
            }
            else
            {
                AddVectors(alpha, a, c);
            }
#elif defined(__ACCELERATE__)
            appleblas_dgeadd(major, transa, transb, m, n, alpha, a.data(), lda, beta, b.data(), ldb, c.data(), ldc);
#else
            if (transa == CblasNoTrans && transb == CblasNoTrans)
            {
#pragma omp parallel for collapse(2)
                for (int i = 0; i < m; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        c(i, j) = alpha * a(i, j) + beta * b(i, j);
                    }
                }
            }
            else if (transa == CblasNoTrans && transb == CblasTrans)
            {
#pragma omp parallel for collapse(2)
                for (int i = 0; i < m; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        c(i, j) = alpha * a(i, j) + beta * b(j, i);
                    }
                }
            }
            else if (transa == CblasTrans && transb == CblasNoTrans)
            {
#pragma omp parallel for collapse(2)
                for (int i = 0; i < m; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        c(i, j) = alpha * a(j, i) + beta * b(i, j);
                    }
                }
            }
            else
            {
#pragma omp parallel for collapse(2)
                for (int i = 0; i < m; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        c(i, j) = alpha * a(j, i) + beta * b(j, i);
                    }
                }
            }

#endif
        };

        /// @brief Add two single-precision matrices with default parameters: C = alpha*op(A) + beta*op(B)
        ///
        /// Convenience overload with optional transpose and scalar parameters.
        /// Performs: C = alpha*op(A) + beta*op(B) with optional transposes.
        ///
        /// @param a Input matrix A (single precision).
        /// @param b Input matrix B (single precision).
        /// @param c Output matrix C (overwritten with the result, single precision).
        /// @param alpha Scalar multiplier for op(A) (single precision). Default: 1.0.
        /// @param beta Scalar multiplier for op(B) (single precision). Default: 1.0.
        /// @param Ta Transpose option for matrix A: "N" (no transpose, A), "T" (transpose, A^T). Default: "N".
        /// @param Tb Transpose option for matrix B: "N" (no transpose, B), "T" (transpose, B^T). Default: "N".
        template <>
        void AddMatrices<float>(const Matrix_<float> &a, const Matrix_<float> &b, Matrix_<float> &c,
                                const float alpha, const float beta, const char *Ta, const char *Tb)
        {
            CBLAS_TRANSPOSE transa = get_trans(Ta);
            CBLAS_TRANSPOSE transb = get_trans(Tb);

            int m, n;
            std::tie(m, n) = check_same_shape_mm(a, b, c, transa, transb);

            BLAS_INT lda = get_leading(m, n, transa);
            BLAS_INT ldb = get_leading(m, n, transb);
            BLAS_INT ldc = get_leading(m, n);

#if defined(_MKL_H_)
            mkl_somatadd(major_char, *Ta, *Tb, m, n, alpha, a.data(), lda, beta, b.data(), ldb, c.data(), ldc);
#elif defined(OPENBLAS_VERSION)
            // copy B into C, potetially transpose
            cblas_somatcopy(major, transb, b.shape().first, b.shape().second, beta, b.data(), ldb, c.data(), ldc);
            // copy A inplace, if transpose
            if (transa == CblasTrans)
            {
                Matrix<float> tmp(c.shape(), 0.0);
                cblas_somatcopy(major, transa, a.shape().first, a.shape().second, alpha, a.data(), lda, tmp.data(), ldc);
                AddVectors(1.0, tmp, c);
            }
            else
            {
                AddVectors(alpha, a, c);
            }
#elif defined(__ACCELERATE__)
            appleblas_sgeadd(major, transa, transb, m, n, alpha, a.data(), lda, beta, b.data(), ldb, c.data(), ldc);
#else
            if (transa == CblasNoTrans && transb == CblasNoTrans)
            {
#pragma omp parallel for collapse(2)
                for (int i = 0; i < m; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        c(i, j) = alpha * a(i, j) + beta * b(i, j);
                    }
                }
            }
            else if (transa == CblasNoTrans && transb == CblasTrans)
            {
#pragma omp parallel for collapse(2)
                for (int i = 0; i < m; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        c(i, j) = alpha * a(i, j) + beta * b(j, i);
                    }
                }
            }
            else if (transa == CblasTrans && transb == CblasNoTrans)
            {
#pragma omp parallel for collapse(2)
                for (int i = 0; i < m; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        c(i, j) = alpha * a(j, i) + beta * b(i, j);
                    }
                }
            }
            else
            {
#pragma omp parallel for collapse(2)
                for (int i = 0; i < m; i++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        c(i, j) = alpha * a(j, i) + beta * b(j, i);
                    }
                }
            }

#endif
        };
    } // namespace cpu
} // namespace lahva
