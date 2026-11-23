#pragma once
#include "const.h"

namespace lahva{
    namespace cpu{ 
        void OuterVectorProduct(const size_t ndimX, const double *x, const size_t incx, const size_t ndimY, const double *y, const size_t incy, const double alpha, double *A);
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
        void MatrixVectorProduct(const size_t m, const size_t n, const float *a, const float *x, float *y,
                                 const char *Ta = "N", const float alpha = 1.0, const float beta = 0.0, const size_t incx = 1, const size_t incy = 1);

        void SymMatrixVectorProduct(const size_t n, const double alpha, const double *a,
                                    const double *x, const size_t incx, const double beta, double *y, const size_t incy);
        void SymMatrixVectorProduct(const size_t n, const double *a, const double *x, double *y,
                                    const double alpha = 1.0, const double beta = 0.0, const size_t incx = 1, const size_t incy = 1);
        void SymMatrixVectorProduct(const size_t n, const float alpha, const float *a,
                                    const float *x, const size_t incx, const float beta, float *y, const size_t incy);
        void SymMatrixVectorProduct(const size_t n, const float *a, const float *x, float *y,
                                    const float alpha = 1.0, const float beta = 0.0, const size_t incx = 1, const size_t incy = 1);
    }
}
