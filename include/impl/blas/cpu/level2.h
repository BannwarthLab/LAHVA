#ifndef LAHVA_C_LEVEL2_H
#define LAHVA_C_LEVEL2_H
#include "const.h"

namespace lahva
{
    namespace cpu
    {
        /// @brief Outer product of two vectors, wrapper to BLAS function dger.
        ///
        /// Performs A = alpha * x * y^T where x is a vector of size ndimX and y is a vector of size ndimY.
        ///
        /// @param ndimX size of vector x
        /// @param x pointer to the first element of vector x
        /// @param incx stride between elements of vector x
        /// @param ndimY size of vector y
        /// @param y pointer to the first element of vector y
        /// @param incy stride between elements of vector y
        /// @param alpha scalar multiplier
        /// @param A pointer to the first element of matrix A (ndimX x ndimY), stored in column-major order
        void OuterVectorProduct(const size_t ndimX, const double *x, const size_t incx, const size_t ndimY, const double *y, const size_t incy, const double alpha, double *A);

        /// @brief Outer product of two vectors, wrapper to BLAS function sger.
        ///
        /// Performs A = alpha * x * y^T where x is a vector of size ndimX and y is a vector of size ndimY.
        ///
        /// @param ndimX size of vector x
        /// @param x pointer to the first element of vector x
        /// @param incx stride between elements of vector x
        /// @param ndimY size of vector y
        /// @param y pointer to the first element of vector y
        /// @param incy stride between elements of vector y
        /// @param alpha scalar multiplier
        /// @param A pointer to the first element of matrix A (ndimX x ndimY), stored in column-major order
        void OuterVectorProduct(const size_t ndimX, const float *x, const size_t incx, const size_t ndimY, const float *y, const size_t incy, const float alpha, float *A);

        /// @brief Product of scaled matrix a and vector x, added to scaled vector y, wrapper to BLAS function dgemv.
        ///
        /// Performs y = alpha * op(a) * x + beta * y where op(a) is a, a^T, or a^H depending on Ta.
        ///
        /// @param Ta Transpose option for matrix a: "N" (no transpose, a), "T" (transpose, a^T), "C" (conjugate-transpose, a^H).
        /// @param m leading dimension of matrix a, i.e. number of rows in a
        /// @param n number of columns in matrix a
        /// @param alpha scalar multiplier for matrix-vector product
        /// @param a pointer to the first element of matrix a (m x n), stored in column-major order
        /// @param x pointer to the first element of vector x (size n when Ta="N", size m otherwise)
        /// @param incx stride between elements of vector x
        /// @param beta scalar multiplier for vector y
        /// @param y pointer to the first element of vector y (size m when Ta="N", size n otherwise)
        /// @param incy stride between elements of vector y
        void MatrixVectorProduct(const char *Ta, const size_t m, const size_t n, const double alpha, const double *a,
                                 const double *x, const size_t incx, const double beta, double *y, const size_t incy);

        /// @brief Product of scaled matrix a and vector x, added to scaled vector y, wrapper to BLAS function dgemv (with defaults).
        ///
        /// Performs y = alpha * op(a) * x + beta * y where op(a) is a, a^T, or a^H depending on Ta.
        ///
        /// @param m leading dimension of matrix a, i.e. number of rows in a
        /// @param n number of columns in matrix a
        /// @param a pointer to the first element of matrix a (m x n), stored in column-major order
        /// @param x pointer to the first element of vector x (size n when Ta="N", size m otherwise)
        /// @param y pointer to the first element of vector y (size m when Ta="N", size n otherwise)
        /// @param Ta Transpose option for matrix a: "N" (no transpose, a), "T" (transpose, a^T), "C" (conjugate-transpose, a^H). Default: "N"
        /// @param alpha scalar multiplier for matrix-vector product. Default: 1.0
        /// @param beta scalar multiplier for vector y. Default: 0.0
        /// @param incx stride between elements of vector x. Default: 1
        /// @param incy stride between elements of vector y. Default: 1
        void MatrixVectorProduct(const size_t m, const size_t n, const double *a, const double *x, double *y,
                                 const char *Ta = "N", const double alpha = 1.0, const double beta = 0.0, const size_t incx = 1, const size_t incy = 1);

        /// @brief Product of scaled matrix a and vector x, added to scaled vector y, wrapper to BLAS function sgemv.
        ///
        /// Performs y = alpha * op(a) * x + beta * y where op(a) is a, a^T, or a^H depending on Ta.
        ///
        /// @param Ta Transpose option for matrix a: "N" (no transpose, a), "T" (transpose, a^T), "C" (conjugate-transpose, a^H).
        /// @param m leading dimension of matrix a, i.e. number of rows in a
        /// @param n number of columns in matrix a
        /// @param alpha scalar multiplier for matrix-vector product
        /// @param a pointer to the first element of matrix a (m x n), stored in column-major order
        /// @param x pointer to the first element of vector x (size n when Ta="N", size m otherwise)
        /// @param incx stride between elements of vector x
        /// @param beta scalar multiplier for vector y
        /// @param y pointer to the first element of vector y (size m when Ta="N", size n otherwise)
        /// @param incy stride between elements of vector y
        void MatrixVectorProduct(const char *Ta, const size_t m, const size_t n, const float alpha, const float *a,
                                 const float *x, const size_t incx, const float beta, float *y, const size_t incy);

        /// @brief Product of scaled matrix a and vector x, added to scaled vector y, wrapper to BLAS function sgemv (with defaults).
        ///
        /// Performs y = alpha * op(a) * x + beta * y where op(a) is a, a^T, or a^H depending on Ta.
        ///
        /// @param m leading dimension of matrix a, i.e. number of rows in a
        /// @param n number of columns in matrix a
        /// @param a pointer to the first element of matrix a (m x n), stored in column-major order
        /// @param x pointer to the first element of vector x (size n when Ta="N", size m otherwise)
        /// @param y pointer to the first element of vector y (size m when Ta="N", size n otherwise)
        /// @param Ta Transpose option for matrix a: "N" (no transpose, a), "T" (transpose, a^T), "C" (conjugate-transpose, a^H). Default: "N"
        /// @param alpha scalar multiplier for matrix-vector product. Default: 1.0
        /// @param beta scalar multiplier for vector y. Default: 0.0
        /// @param incx stride between elements of vector x. Default: 1
        /// @param incy stride between elements of vector y. Default: 1
        void MatrixVectorProduct(const size_t m, const size_t n, const float *a, const float *x, float *y,
                                 const char *Ta = "N", const float alpha = 1.0, const float beta = 0.0, const size_t incx = 1, const size_t incy = 1);

        /// @brief Product of scaled matrix a and vector x, added to scaled vector y, wrapper to BLAS function zgemv.
        ///
        /// Performs y = alpha * op(a) * x + beta * y where op(a) is a, a^T, or a^H depending on Ta.
        ///
        /// @param Ta Transpose option for matrix a: "N" (no transpose, a), "T" (transpose, a^T), "C" (conjugate-transpose, a^H).
        /// @param m leading dimension of matrix a, i.e. number of rows in a
        /// @param n number of columns in matrix a
        /// @param alpha scalar multiplier for matrix-vector product
        /// @param a pointer to the first element of matrix a (m x n), stored in column-major order
        /// @param x pointer to the first element of vector x (size n when Ta="N", size m otherwise)
        /// @param incx stride between elements of vector x
        /// @param beta scalar multiplier for vector y
        /// @param y pointer to the first element of vector y (size m when Ta="N", size n otherwise)
        /// @param incy stride between elements of vector y
        void MatrixVectorProduct(const char *Ta, const size_t m, const size_t n, const complex_double alpha, const complex_double *a,
                                 const complex_double *x, const size_t incx, const complex_double beta, complex_double *y, const size_t incy);

        /// @brief Product of scaled matrix a and vector x, added to scaled vector y, wrapper to BLAS function zgemv (with defaults).
        ///
        /// Performs y = alpha * op(a) * x + beta * y where op(a) is a, a^T, or a^H depending on Ta.
        ///
        /// @param m leading dimension of matrix a, i.e. number of rows in a
        /// @param n number of columns in matrix a
        /// @param a pointer to the first element of matrix a (m x n), stored in column-major order
        /// @param x pointer to the first element of vector x (size n when Ta="N", size m otherwise)
        /// @param y pointer to the first element of vector y (size m when Ta="N", size n otherwise)
        /// @param Ta Transpose option for matrix a: "N" (no transpose, a), "T" (transpose, a^T), "C" (conjugate-transpose, a^H). Default: "N"
        /// @param alpha scalar multiplier for matrix-vector product. Default: 1.0
        /// @param beta scalar multiplier for vector y. Default: 0.0
        /// @param incx stride between elements of vector x. Default: 1
        /// @param incy stride between elements of vector y. Default: 1
        void MatrixVectorProduct(const size_t m, const size_t n, const complex_double *a, const complex_double *x, complex_double *y,
                                 const char *Ta = "N", const complex_double alpha = 1.0, const complex_double beta = 0.0, const size_t incx = 1, const size_t incy = 1);

        /// @brief Product of scaled matrix a and vector x, added to scaled vector y, wrapper to BLAS function cgemv.
        ///
        /// Performs y = alpha * op(a) * x + beta * y where op(a) is a, a^T, or a^H depending on Ta.
        ///
        /// @param Ta Transpose option for matrix a: "N" (no transpose, a), "T" (transpose, a^T), "C" (conjugate-transpose, a^H).
        /// @param m leading dimension of matrix a, i.e. number of rows in a
        /// @param n number of columns in matrix a
        /// @param alpha scalar multiplier for matrix-vector product
        /// @param a pointer to the first element of matrix a (m x n), stored in column-major order
        /// @param x pointer to the first element of vector x (size n when Ta="N", size m otherwise)
        /// @param incx stride between elements of vector x
        /// @param beta scalar multiplier for vector y
        /// @param y pointer to the first element of vector y (size m when Ta="N", size n otherwise)
        /// @param incy stride between elements of vector y
        void MatrixVectorProduct(const char *Ta, const size_t m, const size_t n, const complex_float alpha, const complex_float *a,
                                 const complex_float *x, const size_t incx, const complex_float beta, complex_float *y, const size_t incy);

        /// @brief Product of scaled matrix a and vector x, added to scaled vector y, wrapper to BLAS function cgemv (with defaults).
        ///
        /// Performs y = alpha * op(a) * x + beta * y where op(a) is a, a^T, or a^H depending on Ta.
        ///
        /// @param m leading dimension of matrix a, i.e. number of rows in a
        /// @param n number of columns in matrix a
        /// @param a pointer to the first element of matrix a (m x n), stored in column-major order
        /// @param x pointer to the first element of vector x (size n when Ta="N", size m otherwise)
        /// @param y pointer to the first element of vector y (size m when Ta="N", size n otherwise)
        /// @param Ta Transpose option for matrix a: "N" (no transpose, a), "T" (transpose, a^T), "C" (conjugate-transpose, a^H). Default: "N"
        /// @param alpha scalar multiplier for matrix-vector product. Default: 1.0
        /// @param beta scalar multiplier for vector y. Default: 0.0
        /// @param incx stride between elements of vector x. Default: 1
        /// @param incy stride between elements of vector y. Default: 1
        void MatrixVectorProduct(const size_t m, const size_t n, const complex_float *a, const complex_float *x, complex_float *y,
                                 const char *Ta = "N", const complex_float alpha = 1.0, const complex_float beta = 0.0, const size_t incx = 1, const size_t incy = 1);

        /// @brief Product of scaled symmetric matrix a and vector x, added to scaled vector y, wrapper to BLAS function dsymv.
        ///
        /// Performs y = alpha * A * x + beta * y where A is a symmetric n x n matrix stored in lower triangular packed format.
        ///
        /// @param n size of the symmetric matrix a and vectors x and y
        /// @param alpha scalar multiplier for matrix-vector product
        /// @param a pointer to the first element of matrix a (n x n), stored in lower triangular packed format
        /// @param x pointer to the first element of vector x (size n)
        /// @param incx stride between elements of vector x
        /// @param beta scalar multiplier for vector y
        /// @param y pointer to the first element of vector y (size n)
        /// @param incy stride between elements of vector y
        void SymMatrixVectorProduct(const size_t n, const double alpha, const double *a,
                                    const double *x, const size_t incx, const double beta, double *y, const size_t incy);

        /// @brief Product of scaled symmetric matrix a and vector x, added to scaled vector y, wrapper to BLAS function dsymv (with defaults).
        ///
        /// Performs y = alpha * A * x + beta * y where A is a symmetric n x n matrix stored in lower triangular packed format.
        ///
        /// @param n size of the symmetric matrix a and vectors x and y
        /// @param a pointer to the first element of matrix a (n x n), stored in lower triangular packed format
        /// @param x pointer to the first element of vector x (size n)
        /// @param y pointer to the first element of vector y (size n)
        /// @param alpha scalar multiplier for matrix-vector product. Default: 1.0
        /// @param beta scalar multiplier for vector y. Default: 0.0
        /// @param incx stride between elements of vector x. Default: 1
        /// @param incy stride between elements of vector y. Default: 1
        void SymMatrixVectorProduct(const size_t n, const double *a, const double *x, double *y,
                                    const double alpha = 1.0, const double beta = 0.0, const size_t incx = 1, const size_t incy = 1);

        /// @brief Product of scaled symmetric matrix a and vector x, added to scaled vector y, wrapper to BLAS function ssymv.
        ///
        /// Performs y = alpha * A * x + beta * y where A is a symmetric n x n matrix stored in lower triangular packed format.
        ///
        /// @param n size of the symmetric matrix a and vectors x and y
        /// @param alpha scalar multiplier for matrix-vector product
        /// @param a pointer to the first element of matrix a (n x n), stored in lower triangular packed format
        /// @param x pointer to the first element of vector x (size n)
        /// @param incx stride between elements of vector x
        /// @param beta scalar multiplier for vector y
        /// @param y pointer to the first element of vector y (size n)
        /// @param incy stride between elements of vector y
        void SymMatrixVectorProduct(const size_t n, const float alpha, const float *a,
                                    const float *x, const size_t incx, const float beta, float *y, const size_t incy);

        /// @brief Product of scaled symmetric matrix a and vector x, added to scaled vector y, wrapper to BLAS function ssymv (with defaults).
        ///
        /// Performs y = alpha * A * x + beta * y where A is a symmetric n x n matrix stored in lower triangular packed format.
        ///
        /// @param n size of the symmetric matrix a and vectors x and y
        /// @param a pointer to the first element of matrix a (n x n), stored in lower triangular packed format
        /// @param x pointer to the first element of vector x (size n)
        /// @param y pointer to the first element of vector y (size n)
        /// @param alpha scalar multiplier for matrix-vector product. Default: 1.0
        /// @param beta scalar multiplier for vector y. Default: 0.0
        /// @param incx stride between elements of vector x. Default: 1
        /// @param incy stride between elements of vector y. Default: 1
        void SymMatrixVectorProduct(const size_t n, const float *a, const float *x, float *y,
                                    const float alpha = 1.0, const float beta = 0.0, const size_t incx = 1, const size_t incy = 1);
    
    } // namespace cpu
    
} // namespace lahva

#endif // LAHVA_C_LEVEL2_H