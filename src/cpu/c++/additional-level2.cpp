#include "impl/blas/cpu/additional-level2.hpp"
#include "../../utils/utils.hpp"
namespace tcgmtensor
{
    namespace cpu
    {
        template<>
        void AddMatrices<double>(const char* Ta, const char* Tb, const double alpha, const Matrix_<double>& a, const Matrix_<double>& b,
        const double beta, Matrix_<double>& c)
        {
            CBLAS_TRANSPOSE transa = get_trans(Ta);
            CBLAS_TRANSPOSE transb = get_trans(Tb);

            auto [m, n, k] = check_size_mm(a, b, c, transa, transb);

            BLAS_INT lda = get_leading(m, k);
            BLAS_INT ldb = get_leading(k, n);
            BLAS_INT ldc = get_leading(m, n);

            mkl_domatadd(major_char, *Ta, *Tb, m, n, alpha, a.data(), lda, beta, b.data(), ldb, c.data(), ldc);
        };

        template<>
        void AddMatrices<float>(const char* Ta, const char* Tb, const float alpha, const Matrix_<float>& a, const Matrix_<float>& b,
        const float beta, Matrix_<float>& c)
        {
            CBLAS_TRANSPOSE transa = get_trans(Ta);
            CBLAS_TRANSPOSE transb = get_trans(Tb);

            auto [m, n, k] = check_size_mm(a, b, c, transa, transb);

            BLAS_INT lda = get_leading(m, k);
            BLAS_INT ldb = get_leading(k, n);
            BLAS_INT ldc = get_leading(m, n);

            mkl_somatadd(major_char, *Ta, *Tb, m, n, alpha, a.data(), lda, beta, b.data(), ldb, c.data(), ldc);
        };

        template<>
        void AddMatrices<double>(const Matrix_<double>& a, const Matrix_<double>& b, Matrix_<double>& c,
        const double alpha, const double beta, const char* Ta, const char* Tb)
        {
            CBLAS_TRANSPOSE transa = get_trans(Ta);
            CBLAS_TRANSPOSE transb = get_trans(Tb);

            auto [m, n, k] = check_size_mm(a, b, c, transa, transb);

            BLAS_INT lda = get_leading(m, k);
            BLAS_INT ldb = get_leading(k, n);
            BLAS_INT ldc = get_leading(m, n);

            mkl_domatadd(major_char, *Ta, *Tb, m, n, alpha, a.data(), lda, beta, b.data(), ldb, c.data(), ldc);
        };

        template<>
        void AddMatrices<float>(const Matrix_<float>& a, const Matrix_<float>& b, Matrix_<float>& c,
        const float alpha , const float beta, const char* Ta, const char* Tb)
        {
            CBLAS_TRANSPOSE transa = get_trans(Ta);
            CBLAS_TRANSPOSE transb = get_trans(Tb);

            auto [m, n, k] = check_size_mm(a, b, c, transa, transb);

            BLAS_INT lda = get_leading(m, k);
            BLAS_INT ldb = get_leading(k, n);
            BLAS_INT ldc = get_leading(m, n);

            mkl_somatadd(major_char, *Ta, *Tb, m, n, alpha, a.data(), lda, beta, b.data(), ldb, c.data(), ldc);
        };
    }
}
