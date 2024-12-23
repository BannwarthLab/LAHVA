#include "impl/blas/cpu/level2.hpp"
#include "../../utils/utils.hpp"

namespace tcgmtensor{
    namespace cpu{
    /*! @brief Simple interface to DGEMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
    or \f$\vec{y}=alpha*\mathbf{A}^\intercal*\vec{x}+beta*\vec{y}\f$ or 
    \f$\vec{y}=alpha*conj(\mathbf{A}^\intercal)*\vec{x}+beta*\vec{y}\f$  for specified stride
        @param[in] T character representing the equation execute "N", "T", or "C"
        @param[in] alpha double value by which A*x is scaled 
        @param[in] a pointer to the A matrix in column-major ordering
        @param[in] x pointer to the Vector_ x
        @param[in] incx stride of Vector_ x
        @param[in] beta double value by which y is scaled
        @param[in,out] y pointer to the y Vector_ values
        @param[in] incy stride of Vector_ y
    */
    void MatrixVectorProduct(const char* T, const double alpha, const Matrix_<double>& a, 
                             const Vector_<double>& x, const size_t incx, const double beta, Vector_<double>& y, const size_t incy)
        {
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;
            CBLAS_TRANSPOSE trans = get_trans(T);
            auto [nrow, ncol] = check_size_mv(a, x, y, trans);
            BLAS_INT lda = get_leading(nrow, ncol);

            cblas_dgemv(major, trans, nrow, ncol, alpha, a.data(), lda, x.data(), inx, beta, y.data(), iny);
        };

     /*! @brief Simple interface to DGEMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
    or \f$\vec{y}=alpha*\mathbf{A}^\intercal*\vec{x}+beta*\vec{y}\f$ or 
    \f$\vec{y}=alpha*conj(\mathbf{A}^\intercal)*\vec{x}+beta*\vec{y}\f$ for specified stride
        @param[in] a pointer to the A matrix in column-major ordering
        @param[in] x pointer to the Vector_ x
        @param[in,out] y pointer to the y Vector_ values
        @param[in] T (optional, default value "N") character representing the equation execute "N", "T", or "C"
        @param[in] alpha (optional, default 1.0) double value by which A*x is scaled
        @param[in] beta (optional, default 0.0) double value by which y is scaled
        @param[in] incx (optional, default 1) stride of Vector_ x
        @param[in] incy (optional, default 1) stride of Vector_ y
    */
    void MatrixVectorProduct(const Matrix_<double>& a, const Vector_<double>& x, Vector_<double>& y, const char* T, 
                             const double alpha, const double beta, const size_t incx, const size_t incy)
        {
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;
            CBLAS_TRANSPOSE trans = get_trans(T);
            auto [nrow, ncol] = check_size_mv(a, x, y, trans);
            BLAS_INT lda = get_leading(nrow, ncol);
            cblas_dgemv(major, trans, nrow, ncol, alpha, a.data(), lda, x.data(), inx, beta, y.data(), iny);
        };

    /*! @brief Simple interface to DGEMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
    or \f$\vec{y}=alpha*\mathbf{A}^\intercal*\vec{x}+beta*\vec{y}\f$ or 
    \f$\vec{y}=alpha*conj(\mathbf{A}^\intercal)*\vec{x}+beta*\vec{y}\f$  for specified stride
        @param[in] T character representing the equation execute "N", "T", or "C"
        @param[in] alpha double value by which A*x is scaled 
        @param[in] a pointer to the A matrix in column-major ordering
        @param[in] x pointer to the Vector_ x
        @param[in] incx stride of Vector_ x
        @param[in] beta double value by which y is scaled
        @param[in,out] y pointer to the y Vector_ values
        @param[in] incy stride of Vector_ y
    */
    void MatrixVectorProduct(const char* T, const float alpha, const Matrix_<float>& a, 
                             const Vector_<float>& x, const size_t incx, const float beta, Vector_<float>& y, const size_t incy)
        {
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;
            CBLAS_TRANSPOSE trans = get_trans(T);
            auto [nrow, ncol] = check_size_mv(a, x, y, trans);
            BLAS_INT lda = get_leading(nrow, ncol);
            cblas_sgemv(major, trans, nrow, ncol, alpha, a.data(), lda, x.data(), inx, beta, y.data(), iny);
        };

     /*! @brief Simple interface to SGEMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
    or \f$\vec{y}=alpha*\mathbf{A}^\intercal*\vec{x}+beta*\vec{y}\f$ or 
    \f$\vec{y}=alpha*conj(\mathbf{A}^\intercal)*\vec{x}+beta*\vec{y}\f$ for specified stride
        @param[in] a pointer to the A matrix in column-major ordering
        @param[in] x pointer to the Vector_ x
        @param[in,out] y pointer to the y Vector_ values
        @param[in] T (optional, default value "N") character representing the equation execute "N", "T", or "C"
        @param[in] alpha (optional, default 1.0) double value by which A*x is scaled
        @param[in] beta (optional, default 0.0) double value by which y is scaled
        @param[in] incx (optional, default 1) stride of Vector_ x
        @param[in] incy (optional, default 1) stride of Vector_ y
    */
    void MatrixVectorProduct(const Matrix_<float>& a, const Vector_<float>& x, Vector_<float>& y, const char* T, 
                             const float alpha, const float beta, const size_t incx, const size_t incy)
        {
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;
            CBLAS_TRANSPOSE trans = get_trans(T);
            auto [nrow, ncol] = check_size_mv(a, x, y, trans);
            BLAS_INT lda = get_leading(nrow, ncol);
            cblas_sgemv(major, trans, nrow, ncol, alpha, a.data(), lda, x.data(), inx, beta, y.data(), iny);
        };

    /*! @brief Simple interface to DSYMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$ 
    for specified stride
        @param[in] alpha double value by which A*x is scaled 
        @param[in] a pointer to the A matrix in column-major ordering
        @param[in] x pointer to the Vector_ x
        @param[in] incx stride of Vector_ x
        @param[in] beta double value by which y is scaled
        @param[in,out] y pointer to the y Vector_ values
        @param[in] incy stride of Vector_ y
    */ 
    void SymMatrixVectorProduct(const double alpha, const Matrix_<double>& a, 
                                const Vector_<double>& x, const size_t incx, const double beta, Vector_<double>& y, const size_t incy)
        {
            auto [nrow, ncol] = check_size_mv(a, x, y);
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;

            BLAS_INT lda = get_leading(nrow, ncol);

            cblas_dsymv(major, tri, ncol, alpha, a.data(), lda, x.data(), inx, beta, y.data(), iny);
        };

    /*! @brief Simple interface to DSYMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$ 
    for specified stride
        @param[in] a pointer to the A matrix in column-major ordering
        @param[in] x pointer to the Vector_ x
        @param[in] incx stride of Vector_ x
        @param[in,out] y pointer to the y Vector_ values
        @param[in] alpha (optional, default 1.0) double value by which A*x is scaled 
        @param[in] beta (optional, default 0.0) double value by which y is scaled
        @param[in] incy (optional, default 1) stride of Vector_ y
    */ 
    void SymMatrixVectorProduct(const Matrix_<double>& a, const Vector_<double>& x, Vector_<double>& y,
                                const double alpha, const double beta, const size_t incx, const size_t incy)
        {
            auto [nrow, ncol] = check_size_mv(a, x, y); 
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;

            BLAS_INT lda = get_leading(nrow, ncol);

            cblas_dsymv(major, tri, ncol, alpha, a.data(), lda, x.data(), inx, beta, y.data(), iny);
        };

    /*! @brief Simple interface to DSYMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$ 
    for specified stride
        @param[in] alpha float value by which A*x is scaled 
        @param[in] a pointer to the A matrix in column-major ordering
        @param[in] x pointer to the Vector_ x
        @param[in] incx stride of Vector_ x
        @param[in] beta float value by which y is scaled
        @param[in,out] y pointer to the y Vector_ values
        @param[in] incy stride of Vector_ y
    */ 
    void SymMatrixVectorProduct(const float alpha, const Matrix_<float>& a, 
                                const Vector_<float>& x, const size_t incx, const float beta, Vector_<float>& y, const size_t incy)
        {
            auto [nrow, ncol] = check_size_mv(a, x, y); 
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;

            BLAS_INT lda = get_leading(nrow, ncol);

            cblas_ssymv(major, tri, ncol, alpha, a.data(), lda, x.data(), inx, beta, y.data(), iny);
        };

    /*! @brief Simple interface to SSYMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$ 
    for specified stride
        @param[in] a pointer to the A matrix in column-major ordering
        @param[in] x pointer to the Vector_ x
        @param[in] incx stride of Vector_ x
        @param[in,out] y pointer to the y Vector_ values
        @param[in] alpha (optional, default 1.0) float value by which A*x is scaled 
        @param[in] beta (optional, default 0.0) float value by which y is scaled
        @param[in] incy (optional, default 1) stride of Vector_ y
    */ 
    void SymMatrixVectorProduct(const Matrix_<float>& a, const Vector_<float>& x, Vector_<float>& y,
                                const float alpha, const float beta, const size_t incx, const size_t incy)
        {
            auto [nrow, ncol] = check_size_mv(a, x, y);
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;

            BLAS_INT lda = get_leading(nrow, ncol);

            cblas_ssymv(major, tri, ncol, alpha, a.data(), lda, x.data(), inx, beta, y.data(), iny);
        };

    /*! @brief Simple interface to DSYMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$ 
    for specified stride
        @param[in] alpha double value by which A*x is scaled 
        @param[in] a pointer to the A matrix in column-major ordering
        @param[in] x pointer to the Vector_ x
        @param[in] incx stride of Vector_ x
        @param[in] beta double value by which y is scaled
        @param[in,out] y pointer to the y Vector_ values
        @param[in] incy stride of Vector_ y
    */ 
    void SymMatrixVectorProduct(const double alpha, const LowTriMatrix_<double>& a, 
                                const Vector_<double>& x, const size_t incx, const double beta, Vector_<double>& y, const size_t incy)
        {
            auto [nrow, ncol] = check_size_mv(a, x, y);
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;

            cblas_dspmv(major, tri, ncol, alpha, a.data(), x.data(), inx, beta, y.data(), iny);
        };

    /*! @brief Simple interface to DSYMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$ 
    for specified stride
        @param[in] a pointer to the A matrix in column-major ordering
        @param[in] x pointer to the Vector_ x
        @param[in] incx stride of Vector_ x
        @param[in,out] y pointer to the y Vector_ values
        @param[in] alpha (optional, default 1.0) double value by which A*x is scaled 
        @param[in] beta (optional, default 0.0) double value by which y is scaled
        @param[in] incy (optional, default 1) stride of Vector_ y
    */ 
    void SymMatrixVectorProduct(const LowTriMatrix_<double>& a, const Vector_<double>& x, Vector_<double>& y,
                                const double alpha, const double beta, const size_t incx, const size_t incy)
        {
            auto [nrow, ncol] = check_size_mv(a, x, y); 
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;

            cblas_dspmv(major, tri, ncol, alpha, a.data(), x.data(), inx, beta, y.data(), iny);
        };

    /*! @brief Simple interface to DSYMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$ 
    for specified stride
        @param[in] alpha float value by which A*x is scaled 
        @param[in] a pointer to the A matrix in column-major ordering
        @param[in] x pointer to the Vector_ x
        @param[in] incx stride of Vector_ x
        @param[in] beta float value by which y is scaled
        @param[in,out] y pointer to the y Vector_ values
        @param[in] incy stride of Vector_ y
    */ 
    void SymMatrixVectorProduct(const float alpha, const LowTriMatrix_<float>& a, 
                                const Vector_<float>& x, const size_t incx, const float beta, Vector_<float>& y, const size_t incy)
        {
            auto [nrow, ncol] = check_size_mv(a, x, y); 
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;

            cblas_sspmv(major, tri, ncol, alpha, a.data(), x.data(), inx, beta, y.data(), iny);
        };

    /*! @brief Simple interface to SSYMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$ 
    for specified stride
        @param[in] a pointer to the A matrix in column-major ordering
        @param[in] x pointer to the Vector_ x
        @param[in] incx stride of Vector_ x
        @param[in,out] y pointer to the y Vector_ values
        @param[in] alpha (optional, default 1.0) float value by which A*x is scaled 
        @param[in] beta (optional, default 0.0) float value by which y is scaled
        @param[in] incy (optional, default 1) stride of Vector_ y
    */ 
    void SymMatrixVectorProduct(const LowTriMatrix_<float>& a, const Vector_<float>& x, Vector_<float>& y,
                                const float alpha, const float beta, const size_t incx, const size_t incy)
        {
            auto [nrow, ncol] = check_size_mv(a, x, y);
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;

            cblas_sspmv(major, tri, ncol, alpha, a.data(), x.data(), inx, beta, y.data(), iny);
        };

    
    /*! @brief Simple interface to DTPMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$ 
    for specified stride
        @param[in] T character representing the equation execute "N", "T", or "C"
        @param[in] unit is the given traingular matrix also a unit matrix
        @param[in] a pointer to the lower triangular matrix A in column-major ordering
        @param[in,out] x pointer to the Vector_ x
        @param[in] incx stride of Vector_ x
    */  
    void LowTriMatrixVectorProduct(const char* T, const CBLAS_DIAG unit, const LowTriMatrix_<double>& a, Vector_<double>& x, const size_t incx) 
        {   
            BLAS_INT inx = incx;
            CBLAS_TRANSPOSE trans = get_trans(T);

            auto [row, col] = check_size_mv(a, x);

            cblas_dtpmv(major, tri, trans, unit, col, a.data(), x.data(), inx);
        };

    /*! @brief Simple interface to DTPMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$ 
    for specified stride
        
        @param[in] a pointer to the lower triangular matrix A in column-major ordering
        @param[in,out] x pointer to the Vector_ x
        @param[in] T (optional, default "N") character representing the equation execute "N", "T", or "C"
        @param[in] unit (optional, default CblasNonUnit) is the given traingular matrix also a unit matrix
        @param[in] incx (optional, defaul 1) stride of Vector_ x
    */  
    void LowTriMatrixVectorProduct(const LowTriMatrix_<double>& a, Vector_<double>& x, const char* T, const CBLAS_DIAG unit, const size_t incx)
    {
        BLAS_INT inx = incx;
        CBLAS_TRANSPOSE trans = get_trans(T);

        auto [row, col] = check_size_mv(a, x);

        cblas_dtpmv(major, tri, trans, unit, col, a.data(), x.data(), inx);
    };

    /*! @brief Simple interface to STPMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$ 
    for specified stride
        @param[in] T character representing the equation execute "N", "T", or "C"
        @param[in] unit is the given traingular matrix also a unit matrix
        @param[in] a pointer to the lower triangular matrix A in column-major ordering
        @param[in,out] x pointer to the Vector_ x
        @param[in] incx stride of Vector_ x
    */  
    void LowTriMatrixVectorProduct(const char* T, const CBLAS_DIAG unit, const LowTriMatrix_<float>& a, Vector_<float>& x, const size_t incx)
    {
        BLAS_INT inx = incx;
        CBLAS_TRANSPOSE trans = get_trans(T);

        auto [row, col] = check_size_mv(a, x);

        cblas_stpmv(major, tri, trans, unit, col, a.data(), x.data(), inx);
    };

    /*! @brief Simple interface to STPMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$ 
    for specified stride
        @param[in] a pointer to the lower triangular matrix A in column-major ordering
        @param[in,out] x pointer to the Vector_ x
        @param[in] T (optional, default "N") character representing the equation execute "N", "T", or "C"
        @param[in] unit (optional, default CblasNonUnit) is the given traingular matrix also a unit matrix
        @param[in] incx (optional, defaul 1) stride of Vector_ x
    */ 
    void LowTriMatrixVectorProduct(const LowTriMatrix_<float>& a, Vector_<float>& x, const char* T, CBLAS_DIAG unit, const size_t incx)
    {
        BLAS_INT inx = incx;
        CBLAS_TRANSPOSE trans = get_trans(T);

        auto [row, col] = check_size_mv(a, x);

        cblas_stpmv(major, tri, trans, unit, col, a.data(), x.data(), inx);
    };
    }
}