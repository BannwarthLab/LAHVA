#include "impl/blas/cpu/level2.h"
#include "../utils/utils.hpp"

namespace lahva
{
    namespace cpu
    {
        /// @brief Outer product of two double-precision vectors, wrapper to BLAS function dger.
        ///
        /// Performs A = alpha * x * y^T where x is a vector of size ndimX and y is a vector of size ndimY.
        ///
        /// @param ndimX Size of vector x (number of rows in output matrix A).
        /// @param x Pointer to the first element of vector x.
        /// @param incx Stride between elements of vector x.
        /// @param ndimY Size of vector y (number of columns in output matrix A).
        /// @param y Pointer to the first element of vector y.
        /// @param incy Stride between elements of vector y.
        /// @param alpha Scalar multiplier for the outer product.
        /// @param A Pointer to the first element of output matrix A (ndimX x ndimY, column-major).
        void OuterVectorProduct(const size_t ndimX, const double *x, const size_t incx, const size_t ndimY, const double *y, const size_t incy, const double alpha, double *A)
        {
            BLAS_INT m = (BLAS_INT)ndimX;
            BLAS_INT n = (BLAS_INT)ndimY;
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;
            BLAS_INT lda = get_leading(ndimX, ndimY);

            cblas_dger(major, m, n, alpha, x, inx, y, iny, A, lda);
        }

        /// @brief Outer product of two single-precision vectors, wrapper to BLAS function sger.
        ///
        /// Performs A = alpha * x * y^T where x is a vector of size ndimX and y is a vector of size ndimY.
        ///
        /// @param ndimX Size of vector x (number of rows in output matrix A).
        /// @param x Pointer to the first element of vector x.
        /// @param incx Stride between elements of vector x.
        /// @param ndimY Size of vector y (number of columns in output matrix A).
        /// @param y Pointer to the first element of vector y.
        /// @param incy Stride between elements of vector y.
        /// @param alpha Scalar multiplier for the outer product.
        /// @param A Pointer to the first element of output matrix A (ndimX x ndimY, column-major).
        void OuterVectorProduct(const size_t ndimX, const float *x, const size_t incx, const size_t ndimY, const float *y, const size_t incy, const float alpha, float *A)
        {
            BLAS_INT m = (BLAS_INT)ndimX;
            BLAS_INT n = (BLAS_INT)ndimY;
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;
            BLAS_INT lda = get_leading(ndimX, ndimY);

            cblas_sger(major, m, n, alpha, x, inx, y, iny, A, lda);
        }

        /// @brief Matrix-vector multiply (GEMV-like), wrapper to BLAS function dgemv (double).
        ///
        /// Performs y := alpha * op(A) * x + beta * y where op(A) is A, A^T, or A^H depending on `T`.
        /// This routine computes general matrix-vector products with optional transpose and conjugate-transpose.
        ///
        /// @param T Transpose option for A: "N" (no transpose, A), "T" (transpose, A^T), "C" (conjugate-transpose, A^H).
        /// @param m Number of rows in matrix A.
        /// @param n Number of columns in matrix A.
        /// @param alpha Scaling factor applied to op(A)*x.
        /// @param a Pointer to matrix A in column-major order (m x n).
        /// @param x Pointer to input vector x.
        /// @param incx Stride between consecutive elements in vector x.
        /// @param beta Scaling factor applied to existing contents of y.
        /// @param y Pointer to output vector y (length m if T="N", length n if T="T" or "C").
        /// @param incy Stride between consecutive elements in vector y.
        void MatrixVectorProduct(const char *T, const size_t m, const size_t n, const double alpha, const double *a,
                                 const double *x, const size_t incx, const double beta, double *y, const size_t incy)
        {
            BLAS_INT nrow = m;
            BLAS_INT ncol = n;
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;
            CBLAS_TRANSPOSE trans = get_trans(T);
            BLAS_INT lda = get_leading(nrow, ncol);

            cblas_dgemv(major, trans, nrow, ncol, alpha, a, lda, x, inx, beta, y, iny);
        };

        /// @brief Matrix-vector multiply convenience overload with default parameters (double precision).
        ///
        /// Convenience overload where matrix A and vector x are provided first, with optional parameters
        /// defaulting to no transpose, alpha=1.0, beta=0.0, and unit strides.
        ///
        /// @param m Number of rows in matrix A.
        /// @param n Number of columns in matrix A.
        /// @param a Pointer to matrix A in column-major order (m x n).
        /// @param x Pointer to input vector x.
        /// @param y Pointer to output vector y (length m if T="N", length n if T="T" or "C").
        /// @param T Transpose option for A: "N" (default, no transpose), "T" (transpose, A^T), "C" (conjugate-transpose, A^H).
        /// @param alpha Scaling factor applied to op(A)*x. Default: 1.0.
        /// @param beta Scaling factor applied to existing contents of y. Default: 0.0.
        /// @param incx Stride between consecutive elements in vector x. Default: 1.
        /// @param incy Stride between consecutive elements in vector y. Default: 1.
        void MatrixVectorProduct(const size_t m, const size_t n, const double *a, const double *x, double *y,
                                 const char *T, const double alpha, const double beta, const size_t incx, const size_t incy)
        {
            BLAS_INT nrow = m;
            BLAS_INT ncol = n;
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;
            CBLAS_TRANSPOSE trans = get_trans(T);
            BLAS_INT lda = get_leading(nrow, ncol);

            cblas_dgemv(major, trans, nrow, ncol, alpha, a, lda, x, inx, beta, y, iny);
        };

        /// @brief Matrix-vector multiply (GEMV-like), wrapper to BLAS function sgemv (float).
        ///
        /// Performs y := alpha * op(A) * x + beta * y where op(A) is A, A^T, or A^H depending on `T`.
        /// This routine computes general matrix-vector products with optional transpose and conjugate-transpose.
        ///
        /// @param T Transpose option for A: "N" (no transpose, A), "T" (transpose, A^T), "C" (conjugate-transpose, A^H).
        /// @param m Number of rows in matrix A.
        /// @param n Number of columns in matrix A.
        /// @param alpha Scaling factor applied to op(A)*x.
        /// @param a Pointer to matrix A in column-major order (m x n).
        /// @param x Pointer to input vector x.
        /// @param incx Stride between consecutive elements in vector x.
        /// @param beta Scaling factor applied to existing contents of y.
        /// @param y Pointer to output vector y (length m if T="N", length n if T="T" or "C").
        /// @param incy Stride between consecutive elements in vector y.
        void MatrixVectorProduct(const char *T, const size_t m, const size_t n, const float alpha, const float *a,
                                 const float *x, const size_t incx, const float beta, float *y, const size_t incy)
        {
            BLAS_INT nrow = m;
            BLAS_INT ncol = n;
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;
            CBLAS_TRANSPOSE trans = get_trans(T);
            BLAS_INT lda = get_leading(nrow, ncol);

            cblas_sgemv(major, trans, nrow, ncol, alpha, a, lda, x, inx, beta, y, iny);
        };

        /// @brief Matrix-vector multiply convenience overload with default parameters (float precision).
        ///
        /// Convenience overload where matrix A and vector x are provided first, with optional parameters
        /// defaulting to no transpose, alpha=1.0, beta=0.0, and unit strides.
        ///
        /// @param m Number of rows in matrix A.
        /// @param n Number of columns in matrix A.
        /// @param a Pointer to matrix A in column-major order (m x n).
        /// @param x Pointer to input vector x.
        /// @param y Pointer to output vector y (length m if T="N", length n if T="T" or "C").
        /// @param T Transpose option for A: "N" (default, no transpose), "T" (transpose, A^T), "C" (conjugate-transpose, A^H).
        /// @param alpha Scaling factor applied to op(A)*x. Default: 1.0.
        /// @param beta Scaling factor applied to existing contents of y. Default: 0.0.
        /// @param incx Stride between consecutive elements in vector x. Default: 1.
        /// @param incy Stride between consecutive elements in vector y. Default: 1.
        void MatrixVectorProduct(const size_t m, const size_t n, const float *a, const float *x, float *y,
                                 const char *T, const float alpha, const float beta, const size_t incx, const size_t incy)
        {
            BLAS_INT nrow = m;
            BLAS_INT ncol = n;
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;
            CBLAS_TRANSPOSE trans = get_trans(T);
            BLAS_INT lda = get_leading(nrow, ncol);

            cblas_sgemv(major, trans, nrow, ncol, alpha, a, lda, x, inx, beta, y, iny);
        };

        /// @brief Matrix-vector multiply convenience overload with default parameters (complex float precision).
        ///
        /// Convenience overload where matrix A and vector x are provided first, with optional parameters
        /// defaulting to no transpose, alpha=1.0, beta=0.0, and unit strides.
        ///
        /// @param m Number of rows in matrix A.
        /// @param n Number of columns in matrix A.
        /// @param a Pointer to matrix A in column-major order (m x n).
        /// @param x Pointer to input vector x.
        /// @param y Pointer to output vector y (length m if T="N", length n if T="T" or "C").
        /// @param T Transpose option for A: "N" (default, no transpose), "T" (transpose, A^T), "C" (conjugate-transpose, A^H).
        /// @param alpha Scaling factor applied to op(A)*x. Default: 1.0.
        /// @param beta Scaling factor applied to existing contents of y. Default: 0.0.
        /// @param incx Stride between consecutive elements in vector x. Default: 1.
        /// @param incy Stride between consecutive elements in vector y. Default: 1.
        void MatrixVectorProduct(const size_t m, const size_t n, const complex_float *a, const complex_float *x, complex_float *y,
                                 const char *T, const complex_float alpha, const complex_float beta, const size_t incx, const size_t incy)
        {
            BLAS_INT nrow = m;
            BLAS_INT ncol = n;
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;
            BLAS_INT lda = get_leading(nrow, ncol);
            CBLAS_TRANSPOSE trans = get_trans(T);

            cblas_cgemv(major, trans, nrow, ncol, &alpha, a, lda, x, inx, &beta, y, iny);
        };

        /// @brief Matrix-vector multiply (GEMV-like), wrapper to BLAS function cgemv (complex float).
        ///
        /// Performs y := alpha * op(A) * x + beta * y where op(A) is A, A^T, or A^H depending on `T`.
        /// This routine computes general matrix-vector products with optional transpose and conjugate-transpose.
        ///
        /// @param T Transpose option for A: "N" (no transpose, A), "T" (transpose, A^T), "C" (conjugate-transpose, A^H).
        /// @param m Number of rows in matrix A.
        /// @param n Number of columns in matrix A.
        /// @param alpha Scaling factor applied to op(A)*x.
        /// @param a Pointer to matrix A in column-major order (m x n).
        /// @param x Pointer to input vector x.
        /// @param incx Stride between consecutive elements in vector x.
        /// @param beta Scaling factor applied to existing contents of y.
        /// @param y Pointer to output vector y (length m if T="N", length n if T="T" or "C").
        /// @param incy Stride between consecutive elements in vector y.
        void MatrixVectorProduct(const char *T, const size_t m, const size_t n, const complex_float alpha, const complex_float *a,
                                 const complex_float *x, const size_t incx, const complex_float beta, complex_float *y, const size_t incy)
        {
            BLAS_INT nrow = m;
            BLAS_INT ncol = n;
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;
            CBLAS_TRANSPOSE trans = get_trans(T);
            BLAS_INT lda = get_leading(nrow, ncol);

            cblas_cgemv(major, trans, nrow, ncol, &alpha, a, lda, x, inx, &beta, y, iny);
        };

        /// @brief Matrix-vector multiply (GEMV-like), wrapper to BLAS function zgemv (complex double).
        ///
        /// Performs y := alpha * op(A) * x + beta * y where op(A) is A, A^T, or A^H depending on `T`.
        /// This routine computes general matrix-vector products with optional transpose and conjugate-transpose.
        ///
        /// @param T Transpose option for A: "N" (no transpose, A), "T" (transpose, A^T), "C" (conjugate-transpose, A^H).
        /// @param m Number of rows in matrix A.
        /// @param n Number of columns in matrix A.
        /// @param alpha Scaling factor applied to op(A)*x.
        /// @param a Pointer to matrix A in column-major order (m x n).
        /// @param x Pointer to input vector x.
        /// @param incx Stride between consecutive elements in vector x.
        /// @param beta Scaling factor applied to existing contents of y.
        /// @param y Pointer to output vector y (length m if T="N", length n if T="T" or "C").
        /// @param incy Stride between consecutive elements in vector y.
        void MatrixVectorProduct(const char *T, const size_t m, const size_t n, const complex_double alpha, const complex_double *a,
                                 const complex_double *x, const size_t incx, const complex_double beta, complex_double *y, const size_t incy)
        {
            BLAS_INT nrow = m;
            BLAS_INT ncol = n;
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;
            CBLAS_TRANSPOSE trans = get_trans(T);
            BLAS_INT lda = get_leading(nrow, ncol);

            cblas_zgemv(major, trans, nrow, ncol, &alpha, a, lda, x, inx, &beta, y, iny);
        };

        /// @brief Matrix-vector multiply convenience overload with default parameters (complex double precision).
        ///
        /// Convenience overload where matrix A and vector x are provided first, with optional parameters
        /// defaulting to no transpose, alpha=1.0, beta=0.0, and unit strides.
        ///
        /// @param m Number of rows in matrix A.
        /// @param n Number of columns in matrix A.
        /// @param a Pointer to matrix A in column-major order (m x n).
        /// @param x Pointer to input vector x.
        /// @param y Pointer to output vector y (length m if T="N", length n if T="T" or "C").
        /// @param T Transpose option for A: "N" (default, no transpose), "T" (transpose, A^T), "C" (conjugate-transpose, A^H).
        /// @param alpha Scaling factor applied to op(A)*x. Default: 1.0.
        /// @param beta Scaling factor applied to existing contents of y. Default: 0.0.
        /// @param incx Stride between consecutive elements in vector x. Default: 1.
        /// @param incy Stride between consecutive elements in vector y. Default: 1.
        void MatrixVectorProduct(const size_t m, const size_t n, const complex_double *a, const complex_double *x, complex_double *y,
                                 const char *T, const complex_double alpha, const complex_double beta, const size_t incx, const size_t incy)
        {
            BLAS_INT nrow = m;
            BLAS_INT ncol = n;
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;
            BLAS_INT lda = get_leading(nrow, ncol);
            CBLAS_TRANSPOSE trans = get_trans(T);

            cblas_zgemv(major, trans, nrow, ncol, &alpha, a, lda, x, inx, &beta, y, iny);
        };

        /// @brief Symmetric matrix-vector multiply (SYMV-like), wrapper to BLAS function dsymv (double).
        ///
        /// Performs y := alpha * A * x + beta * y where A is a symmetric matrix.
        /// Only the lower or upper triangular part of A is used (determined by project configuration).
        ///
        /// @param n Size of the square matrix A (number of rows and columns).
        /// @param alpha Scaling factor applied to A*x.
        /// @param a Pointer to matrix A in column-major order (n x n).
        /// @param x Pointer to input vector x (length n).
        /// @param incx Stride between consecutive elements in vector x.
        /// @param beta Scaling factor applied to existing contents of y.
        /// @param y Pointer to output vector y (length n).
        /// @param incy Stride between consecutive elements in vector y.
        void SymMatrixVectorProduct(const size_t n, const double alpha, const double *a,
                                    const double *x, const size_t incx, const double beta, double *y, const size_t incy)
        {
            BLAS_INT ncol = n;
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;
            BLAS_INT lda = get_leading(ncol, ncol);

            cblas_dsymv(major, tri, ncol, alpha, a, lda, x, inx, beta, y, iny);
        };

        /// @brief Symmetric matrix-vector multiply convenience overload with default parameters (double precision).
        ///
        /// Convenience overload where matrix A and vector x are provided first, with optional parameters
        /// defaulting to alpha=1.0, beta=0.0, and unit strides.
        ///
        /// @param n Size of the square matrix A (number of rows and columns).
        /// @param a Pointer to matrix A in column-major order (n x n).
        /// @param x Pointer to input vector x (length n).
        /// @param y Pointer to output vector y (length n).
        /// @param alpha Scaling factor applied to A*x. Default: 1.0.
        /// @param beta Scaling factor applied to existing contents of y. Default: 0.0.
        /// @param incx Stride between consecutive elements in vector x. Default: 1.
        /// @param incy Stride between consecutive elements in vector y. Default: 1.
        void SymMatrixVectorProduct(const size_t n, const double *a, const double *x, double *y,
                                    const double alpha, const double beta, const size_t incx, const size_t incy)
        {
            BLAS_INT ncol = n;
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;
            BLAS_INT lda = get_leading(ncol, ncol);

            cblas_dsymv(major, tri, ncol, alpha, a, lda, x, inx, beta, y, iny);
        };

        /// @brief Symmetric matrix-vector multiply (SYMV-like), wrapper to BLAS function ssymv (float).
        ///
        /// Performs y := alpha * A * x + beta * y where A is a symmetric matrix.
        /// Only the lower or upper triangular part of A is used (determined by project configuration).
        ///
        /// @param n Size of the square matrix A (number of rows and columns).
        /// @param alpha Scaling factor applied to A*x.
        /// @param a Pointer to matrix A in column-major order (n x n).
        /// @param x Pointer to input vector x (length n).
        /// @param incx Stride between consecutive elements in vector x.
        /// @param beta Scaling factor applied to existing contents of y.
        /// @param y Pointer to output vector y (length n).
        /// @param incy Stride between consecutive elements in vector y.
        void SymMatrixVectorProduct(const size_t n, const float alpha, const float *a,
                                    const float *x, const size_t incx, const float beta, float *y, const size_t incy)
        {
            BLAS_INT ncol = n;
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;
            BLAS_INT lda = get_leading(ncol, ncol);

            cblas_ssymv(major, tri, ncol, alpha, a, lda, x, inx, beta, y, iny);
        };

        /// @brief Symmetric matrix-vector multiply convenience overload with default parameters (float precision).
        ///
        /// Convenience overload where matrix A and vector x are provided first, with optional parameters
        /// defaulting to alpha=1.0, beta=0.0, and unit strides.
        ///
        /// @param n Size of the square matrix A (number of rows and columns).
        /// @param a Pointer to matrix A in column-major order (n x n).
        /// @param x Pointer to input vector x (length n).
        /// @param y Pointer to output vector y (length n).
        /// @param alpha Scaling factor applied to A*x. Default: 1.0.
        /// @param beta Scaling factor applied to existing contents of y. Default: 0.0.
        /// @param incx Stride between consecutive elements in vector x. Default: 1.
        /// @param incy Stride between consecutive elements in vector y. Default: 1.
        void SymMatrixVectorProduct(const size_t n, const float *a, const float *x, float *y,
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