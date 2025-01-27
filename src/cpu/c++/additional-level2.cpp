#include "impl/blas/cpu/additional-level2.hpp"
#include "../../utils/utils.hpp"
namespace lahva
{
    namespace cpu
    {
        template <>
        void AddMatrices<double>(const char *Ta, const char *Tb, const double alpha, const Matrix_<double> &a, const Matrix_<double> &b,
                                 const double beta, Matrix_<double> &c)
        {
            CBLAS_TRANSPOSE transa = get_trans(Ta);
            CBLAS_TRANSPOSE transb = get_trans(Tb);

            auto [m, n] = check_same_shape_mm(a, b, c, transa, transb);

            BLAS_INT lda = get_leading(m, n, transa);
            BLAS_INT ldb = get_leading(m, n, transb);
            BLAS_INT ldc = get_leading(m, n);
#ifdef W_MKL
            mkl_domatadd(major_char, *Ta, *Tb, m, n, alpha, a.data(), lda, beta, b.data(), ldb, c.data(), ldc);
#endif

#ifdef OPENBLAS_GENERIC
            // copy B into C, potetially transpose
            cblas_domatcopy(major_char, *Tb, b.shape().first, b.shape().second, b.data(), ldb, c.data(), ldc);
            // copy A inplay, if transpose
            if (transa == CblasTrans)
            {
                cblas_dimatcopy(major_char, *Ta, a.shape().first, a.shape().second, a.data(), lda);
            }
            cblas_dgeadd(major_char, m, n, alpha, a.data(), lda, beta, c.data(), ldc);
#endif
        };

        template <>
        void AddMatrices<float>(const char *Ta, const char *Tb, const float alpha, const Matrix_<float> &a, const Matrix_<float> &b,
                                const float beta, Matrix_<float> &c)
        {
            CBLAS_TRANSPOSE transa = get_trans(Ta);
            CBLAS_TRANSPOSE transb = get_trans(Tb);

            auto [m, n] = check_same_shape_mm(a, b, c, transa, transb);

            BLAS_INT lda = get_leading(m, n, transa);
            BLAS_INT ldb = get_leading(m, n, transb);
            BLAS_INT ldc = get_leading(m, n);

#ifdef W_MKL
            mkl_somatadd(major_char, *Ta, *Tb, m, n, alpha, a.data(), lda, beta, b.data(), ldb, c.data(), ldc);
#endif

#ifdef OPENBLAS_GENERIC
            // copy B into C, potetially transpose
            cblas_somatcopy(major_char, *Tb, b.shape().first, b.shape().second, b.data(), ldb, c.data(), ldc);
            // copy A inplay, if transpose
            if (transa == CblasTrans)
            {
                cblas_simatcopy(major_char, *Ta, a.shape().first, a.shape().second, a.data(), lda);
            }
            cblas_sgeadd(major_char, m, n, alpha, a.data(), lda, beta, c.data(), ldc);
#endif
        };

        template <>
        void AddMatrices<double>(const Matrix_<double> &a, const Matrix_<double> &b, Matrix_<double> &c,
                                 const double alpha, const double beta, const char *Ta, const char *Tb)
        {
            CBLAS_TRANSPOSE transa = get_trans(Ta);
            CBLAS_TRANSPOSE transb = get_trans(Tb);

            auto [m, n] = check_same_shape_mm(a, b, c, transa, transb);

            BLAS_INT lda = get_leading(m, n, transa);
            BLAS_INT ldb = get_leading(m, n, transb);
            BLAS_INT ldc = get_leading(m, n);

#ifdef W_MKL
            mkl_domatadd(major_char, *Ta, *Tb, m, n, alpha, a.data(), lda, beta, b.data(), ldb, c.data(), ldc);
#endif

#ifdef OPENBLAS_GENERIC
            // copy B into C, potetially transpose
            cblas_domatcopy(major_char, *Tb, b.shape().first, b.shape().second, b.data(), ldb, c.data(), ldc);
            // copy A inplay, if transpose
            if (transa == CblasTrans)
            {
                cblas_dimatcopy(major_char, *Ta, a.shape().first, a.shape().second, a.data(), lda);
            }
            cblas_dgeadd(major_char, m, n, alpha, a.data(), lda, beta, c.data(), ldc);
#endif
        };

        template <>
        void AddMatrices<float>(const Matrix_<float> &a, const Matrix_<float> &b, Matrix_<float> &c,
                                const float alpha, const float beta, const char *Ta, const char *Tb)
        {
            CBLAS_TRANSPOSE transa = get_trans(Ta);
            CBLAS_TRANSPOSE transb = get_trans(Tb);

            auto [m, n] = check_same_shape_mm(a, b, c, transa, transb);

            BLAS_INT lda = get_leading(m, n, transa);
            BLAS_INT ldb = get_leading(m, n, transb);
            BLAS_INT ldc = get_leading(m, n);

#ifdef W_MKL
            mkl_somatadd(major_char, *Ta, *Tb, m, n, alpha, a.data(), lda, beta, b.data(), ldb, c.data(), ldc);
#endif

#ifdef OPENBLAS_GENERIC
            // copy B into C, potetially transpose
            cblas_somatcopy(major_char, *Tb, b.shape().first, b.shape().second, b.data(), ldb, c.data(), ldc);
            // copy A inplay, if transpose
            if (transa == CblasTrans)
            {
                cblas_simatcopy(major_char, *Ta, a.shape().first, a.shape().second, a.data(), lda);
            }
            cblas_sgeadd(major_char, m, n, alpha, a.data(), lda, beta, c.data(), ldc);
#endif
        };
    }
}
