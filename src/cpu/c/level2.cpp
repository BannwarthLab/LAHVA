#include "impl/blas/cpu/level2.h"
#include "../utils/utils.hpp"

namespace lahva{
    namespace cpu{
    /// @brief Simple interface to DGER \f$\mathbf{A}=alpha*\vec{x}*\vec{y}**T + A\f$
    /// @param ndimX number of elements in x
    /// @param x pointer to x values
    /// @param incx stride of Vector x
    /// @param ndimY number of elements in y
    /// @param y pointer to y values
    /// @param incy stride of Vector y
    /// @param alpha value by which \f$\vec{x}*\vec{y}**T\f$ is scaled
    /// @param A pointer to output matrix A of size ndimX by ndimY
    void OuterVectorProduct(const size_t ndimX, const double* x, const size_t incx, const size_t ndimY, const double* y, const size_t incy, const double alpha, double* A) {
        BLAS_INT m = (BLAS_INT) ndimX;
        BLAS_INT n = (BLAS_INT) ndimY;
        BLAS_INT inx = incx;
        BLAS_INT iny = incy;
        BLAS_INT lda = get_leading(ndimX, ndimY);

        cblas_dger(major, m, n, alpha, x, inx, y, iny, A, lda);
    }

    /// @brief Simple interface to SGER \f$\mathbf{A}=alpha*\vec{x}*\vec{y}**T + A\f$
    /// @param ndimX number of elements in x
    /// @param x pointer to x values
    /// @param incx stride of Vector x
    /// @param ndimY number of elements in y
    /// @param y pointer to y values
    /// @param incy stride of Vector y
    /// @param alpha value by which \f$\vec{x}*\vec{y}**T\f$ is scaled
    /// @param A pointer to output matrix A of size ndimX by ndimY
    void OuterVectorProduct(const size_t ndimX, const float* x, const size_t incx, const size_t ndimY, const float* y, const size_t incy, const float alpha, float* A) {
        BLAS_INT m = (BLAS_INT) ndimX;
        BLAS_INT n = (BLAS_INT) ndimY;
        BLAS_INT inx = incx;
        BLAS_INT iny = incy;
        BLAS_INT lda = get_leading(ndimX, ndimY);

        cblas_sger(major, m, n, alpha, x, inx, y, iny, A, lda);
    }
    
    
    
    
    /*! @brief Simple interface to DGEMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
    or \f$\vec{y}=alpha*\mathbf{A}^\intercal*\vec{x}+beta*\vec{y}\f$ or 
    \f$\vec{y}=alpha*conj(\mathbf{A})*\vec{x}+beta*\vec{y}\f$  for specified stride
        @param[in] T character representing the equation execute "N", "T", or "C"
        @param[in] m row size of the matrix A
        @param[in] n column size of the matrix A
        @param[in] alpha double value by which A*x is scaled 
        @param[in] a pointer to the A matrix in column-major ordering
        @param[in] x pointer to the Vector x
        @param[in] incx stride of Vector x
        @param[in] beta double value by which y is scaled
        @param[in,out] y pointer to the y Vector values
        @param[in] incy stride of Vector y
    */
    void MatrixVectorProduct(const char* T, const size_t m, const size_t n, const double alpha, const double* a, 
                             const double* x, const size_t incx, const double beta, double* y, const size_t incy)
        {
            BLAS_INT nrow = m;
            BLAS_INT ncol = n; 
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;
            CBLAS_TRANSPOSE trans = get_trans(T);
            BLAS_INT lda = get_leading(nrow, ncol);

            cblas_dgemv(major, trans, nrow, ncol, alpha, a, lda, x, inx, beta, y, iny);
        };

     /*! @brief Simple interface to DGEMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
    or \f$\vec{y}=alpha*\mathbf{A}^\intercal*\vec{x}+beta*\vec{y}\f$ or 
    \f$\vec{y}=alpha*conj(\mathbf{A})*\vec{x}+beta*\vec{y}\f$ for specified stride
        @param[in] m row size of the matrix A
        @param[in] n column size of the matrix A
        @param[in] a pointer to the A matrix in column-major ordering
        @param[in] x pointer to the Vector x
        @param[in,out] y pointer to the y Vector values
        @param[in] T (optional, default value "N") character representing the equation execute "N", "T", or "C"
        @param[in] alpha (optional, default 1.0) double value by which A*x is scaled
        @param[in] beta (optional, default 0.0) double value by which y is scaled
        @param[in] incx (optional, default 1) stride of Vector x
        @param[in] incy (optional, default 1) stride of Vector y
    */
    void MatrixVectorProduct(const size_t m, const size_t n,  const double* a, const double* x, double* y,
                             const char* T, const double alpha, const double beta, const size_t incx, const size_t incy)
        {
            BLAS_INT nrow = m;
            BLAS_INT ncol = n; 
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;
            CBLAS_TRANSPOSE trans = get_trans(T);
            BLAS_INT lda = get_leading(nrow, ncol);

            cblas_dgemv(major, trans, nrow, ncol, alpha, a, lda, x, inx, beta, y, iny);
        };

    /*! @brief Simple interface to SGEMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$ 
    or \f$\vec{y}=alpha*\mathbf{A}^\intercal*\vec{x}+beta*\vec{y}\f$ or 
    \f$\vec{y}=alpha*conj(\mathbf{A})*\vec{x}+beta*\vec{y}\f$  for specified stride
        @param[in] T character representing the equation execute "N", "T", or "C"
        @param[in] m row size of the matrix A
        @param[in] n column size of the matrix A
        @param[in] alpha float value by which A*x is scaled 
        @param[in] a pointer to the A matrix in column-major ordering
        @param[in] x pointer to the Vector x
        @param[in] incx stride of Vector x
        @param[in] beta float value by which y is scaled
        @param[in,out] y pointer to the y Vector values
        @param[in] incy stride of Vector y
    */
    void MatrixVectorProduct(const char* T, const size_t m, const size_t n, const float alpha, const float* a, 
                             const float* x, const size_t incx, const float beta, float* y, const size_t incy)
        {
            BLAS_INT nrow = m;
            BLAS_INT ncol = n; 
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;
            CBLAS_TRANSPOSE trans = get_trans(T);
            BLAS_INT lda = get_leading(nrow, ncol);

            cblas_sgemv(major, trans, nrow, ncol, alpha, a, lda, x, inx, beta, y, iny);
        };

    /*! @brief Simple interface to SGEMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$ 
    or \f$\vec{y}=alpha*\mathbf{A}^\intercal*\vec{x}+beta*\vec{y}\f$ or 
    \f$\vec{y}=alpha*conj(\mathbf{A})*\vec{x}+beta*\vec{y}\f$ for specified stride
        @param[in] m row size of the matrix A
        @param[in] n column size of the matrix A
        @param[in] a pointer to the A matrix in column-major ordering
        @param[in] x pointer to the Vector x
        @param[in,out] y pointer to the y Vector values
        @param[in] T (optional, default value "N") character representing the equation execute "N", "T", or "C"
        @param[in] alpha (optional, default 1.0) float value by which A*x is scaled
        @param[in] beta (optional, default 0.0) float value by which y is scaled
        @param[in] incx (optional, default 1) stride of Vector x
        @param[in] incy (optional, default 1) stride of Vector y
    */ 
    void MatrixVectorProduct(const size_t m, const size_t n,  const float* a, const float* x, float* y,
                             const char* T, const float alpha, const float beta, const size_t incx, const size_t incy)
        {
            BLAS_INT nrow = m;
            BLAS_INT ncol = n; 
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;
            CBLAS_TRANSPOSE trans = get_trans(T);
            BLAS_INT lda = get_leading(nrow, ncol);

            cblas_sgemv(major, trans, nrow, ncol, alpha, a, lda, x, inx, beta, y, iny);
        };                    


    /*! @brief Simple interface to DSYMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$ 
    for specified stride
        @param[in] n column size of the matrix A
        @param[in] alpha double value by which A*x is scaled 
        @param[in] a pointer to the A matrix in column-major ordering
        @param[in] x pointer to the Vector x
        @param[in] incx stride of Vector x
        @param[in] beta double value by which y is scaled
        @param[in,out] y pointer to the y Vector values
        @param[in] incy stride of Vector y
    */ 
    void SymMatrixVectorProduct(const size_t n, const double alpha, const double* a, 
                                const double* x, const size_t incx, const double beta, double* y, const size_t incy)
        {
            BLAS_INT ncol = n; 
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;
            BLAS_INT lda = get_leading(ncol, ncol);

            cblas_dsymv(major, tri, ncol, alpha, a, lda, x, inx, beta, y, iny);
        };

    /*! @brief Simple interface to DSYMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$ 
    for specified stride
        @param[in] n column size of the matrix A
        @param[in] a pointer to the A matrix in column-major ordering
        @param[in] x pointer to the Vector x
        @param[in] incx stride of Vector x
        @param[in,out] y pointer to the y Vector values
        @param[in] alpha (optional, default 1.0) double value by which A*x is scaled 
        @param[in] beta (optional, default 0.0) double value by which y is scaled
        @param[in] incy (optional, default 1) stride of Vector y
    */ 
    void SymMatrixVectorProduct(const size_t n,  const double* a, const double* x, double* y,
                                const double alpha, const double beta, const size_t incx, const size_t incy)
        {
            BLAS_INT ncol = n; 
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;
            BLAS_INT lda = get_leading(ncol, ncol);

            cblas_dsymv(major, tri, ncol, alpha, a, lda, x, inx, beta, y, iny);
        };

    /*! @brief Simple interface to DSYMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$ 
    for specified stride
        @param[in] n column size of the matrix A
        @param[in] alpha float value by which A*x is scaled 
        @param[in] a pointer to the A matrix in column-major ordering
        @param[in] x pointer to the Vector x
        @param[in] incx stride of Vector x
        @param[in] beta float value by which y is scaled
        @param[in,out] y pointer to the y Vector values
        @param[in] incy stride of Vector y
    */ 
    void SymMatrixVectorProduct(const size_t n, const float alpha, const float* a, 
                                const float* x, const size_t incx, const float beta, float* y, const size_t incy)
        {
            BLAS_INT ncol = n; 
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;
            BLAS_INT lda = get_leading(ncol, ncol);

            cblas_ssymv(major, tri, ncol, alpha, a, lda, x, inx, beta, y, iny);
        };

    /*! @brief Simple interface to SSYMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$ 
    for specified stride
        @param[in] n column size of the matrix A
        @param[in] a pointer to the A matrix in column-major ordering
        @param[in] x pointer to the Vector x
        @param[in] incx stride of Vector x
        @param[in,out] y pointer to the y Vector values
        @param[in] alpha (optional, default 1.0) float value by which A*x is scaled 
        @param[in] beta (optional, default 0.0) float value by which y is scaled
        @param[in] incy (optional, default 1) stride of Vector y
    */ 
    void SymMatrixVectorProduct(const size_t n,  const float* a, const float* x, float* y,
                                const float alpha, const float beta, const size_t incx, const size_t incy)
        {
            BLAS_INT ncol = n; 
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;
            BLAS_INT lda = get_leading(ncol, ncol);

            cblas_ssymv(major, tri, ncol, alpha, a, lda, x, inx, beta, y, iny);
        };
    }
}