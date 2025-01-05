#include "impl/blas/cpu/level3.hpp"
#include "../../utils/utils.hpp"
namespace lahva
{
    namespace cpu{ 
    
    void MatrixMatrixProduct(const char* Ta, const char* Tb, const double alpha, const Matrix_<double>& a, 
                             const Matrix_<double>& b, const double beta, Matrix_<double>& c)
    {
        CBLAS_TRANSPOSE transa = get_trans(Ta);
        CBLAS_TRANSPOSE transb = get_trans(Tb);

        auto [m, n, k] = check_size_mm(a, b, c, transa, transb);

        BLAS_INT lda = get_leading(m, k);
        BLAS_INT ldb = get_leading(k, n);
        BLAS_INT ldc = get_leading(m, n);

        cblas_dgemm(major, transa, transb, m, n, k, alpha, a.data(), lda, b.data(), ldb, beta, c.data(), ldc);
    };

    void MatrixMatrixProduct(const Matrix_<double>& a, const Matrix_<double>& b, Matrix_<double>& c,
                             const double alpha, const double beta, const char* Ta, const char* Tb)
    {
        CBLAS_TRANSPOSE transa = get_trans(Ta);
        CBLAS_TRANSPOSE transb = get_trans(Tb);

        auto [m, n, k] = check_size_mm(a, b, c, transa, transb);

        BLAS_INT lda = get_leading(m, k);
        BLAS_INT ldb = get_leading(k, n);
        BLAS_INT ldc = get_leading(m, n);

        cblas_dgemm(major, transa, transb, m, n, k, alpha, a.data(), lda, b.data(), ldb, beta, c.data(), ldc);
    };
                             
    void MatrixMatrixProduct(const char* Ta, const char* Tb, const float alpha, const Matrix_<float>& a, const Matrix_<float>& b,
                             const float beta, Matrix_<float>& c)
    {
        CBLAS_TRANSPOSE transa = get_trans(Ta);
        CBLAS_TRANSPOSE transb = get_trans(Tb);

        auto [m, n, k] = check_size_mm(a, b, c, transa, transb);

        BLAS_INT lda = get_leading(m, k);
        BLAS_INT ldb = get_leading(k, n);
        BLAS_INT ldc = get_leading(m, n);

        cblas_sgemm(major, transa, transb, m, n, k, alpha, a.data(), lda, b.data(), ldb, beta, c.data(), ldc);
    };

    void MatrixMatrixProduct(const Matrix_<float>& a, const Matrix_<float>& b, Matrix_<float>& c,
                             const float alpha, const float beta, const char* Ta, const char* Tb)
    {
        CBLAS_TRANSPOSE transa = get_trans(Ta);
        CBLAS_TRANSPOSE transb = get_trans(Tb);

        auto [m, n, k] = check_size_mm(a, b, c, transa, transb);

        BLAS_INT lda = get_leading(m, k);
        BLAS_INT ldb = get_leading(k, n);
        BLAS_INT ldc = get_leading(m, n);

        cblas_sgemm(major, transa, transb, m, n, k, alpha, a.data(), lda, b.data(), ldb, beta, c.data(), ldc);
    };

    void SymMatrixMatrixProduct(const CBLAS_SIDE side, const double alpha, const Matrix_<double>& a, const Matrix_<double>& b,
                                const double beta, Matrix_<double>& c)
    {
        auto [m, n, k] = check_size_mm(a, b, c);
        BLAS_INT lda = get_leading(m, k);
        BLAS_INT ldb = get_leading(k, n);
        BLAS_INT ldc = get_leading(m, n);
        cblas_dsymm(major, side, tri, m, n, alpha, a.data(), lda, b.data(), ldb, beta, c.data(), ldc);
    };

    void SymMatrixMatrixProduct(const Matrix_<double>& a, const Matrix_<double>& b, Matrix_<double>& c,
                                const double alpha, const double beta, const CBLAS_SIDE side)
    {
        auto [m, n, k] = check_size_mm(a, b, c);
        BLAS_INT lda = get_leading(m, k);
        BLAS_INT ldb = get_leading(k, n);
        BLAS_INT ldc = get_leading(m, n);
        cblas_dsymm(major, side, tri, m, n, alpha, a.data(), lda, b.data(), ldb, beta, c.data(), ldc);
    };

    void SymMatrixMatrixProduct(const CBLAS_SIDE side, const float alpha, const Matrix_<float>& a, const Matrix_<float>& b,
                                const float beta, Matrix_<float>& c)
    {
        auto [m, n, k] = check_size_mm(a, b, c);

        BLAS_INT lda = get_leading(m, k);
        BLAS_INT ldb = get_leading(k, n);
        BLAS_INT ldc = get_leading(m, n);

        cblas_ssymm(major, side, tri, m, n, alpha, a.data(), lda, b.data(), ldb, beta, c.data(), ldc);
    };

    void SymMatrixMatrixProduct(const Matrix_<float>& a, const Matrix_<float>& b, Matrix_<float>& c,
                                const float alpha, const float beta, const CBLAS_SIDE side)
    {
        auto [m, n, k] = check_size_mm(a, b, c);

        BLAS_INT lda = get_leading(m, k);
        BLAS_INT ldb = get_leading(k, n);
        BLAS_INT ldc = get_leading(m, n);

        cblas_ssymm(major, side, tri, m, n, alpha, a.data(), lda, b.data(), ldb, beta, c.data(), ldc);
    }; 
    }
} // namespace lahva
