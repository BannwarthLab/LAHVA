#include "linalg.hpp"
#include "impl/blas/cpu/additional-level2.hpp"
#include "../../utils/utils.hpp"
#include "impl/blas/cpu/level1.hpp"

namespace lahva
{
    namespace cpu
    {
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
    }
}
