/// @file level2.cpp
/// @brief CPU C++ template implementations of Level-2 BLAS operations.
///
/// Provides C++ template functions for Level-2 BLAS matrix-vector operations including
/// outer products and matrix-vector multiplication. Functions accept Matrix<T> and Vector<T>
/// types and forward to lower-level pointer-based CBLAS routines.

#include "../utils/utils.hpp"
#include "impl/blas/cpu/level2.h"
#include "impl/blas/cpu/level2.hpp"
#include "linalg.hpp"

namespace lahva
{
    namespace cpu
    {
        /// @brief Outer product explicit specialization for double precision.
        ///
        /// Performs A = alpha * x * y^T where x is a vector of size M and y is a vector of size N,
        /// producing an M x N matrix. This specialization wrapper accepts Vector<double> and Matrix<double> typed storage
        /// and forwards to the lower-level pointer-based implementation.
        ///
        /// @param x First input vector (Vector<double>).
        /// @param y Second input vector (Vector<double>).
        /// @param A Output matrix to store the outer product (Matrix<double>).
        /// @param incx Stride between elements of vector x.
        /// @param incy Stride between elements of vector y.
        /// @param alpha Scalar multiplier for the outer product.
        template <>
        void OuterVectorProduct<double>(const Vector<double>& x, const Vector<double>& y, Matrix<double>& A, size_t incx, size_t incy, const double alpha) {
            if (A.shape().first != x.size() || A.shape().second != y.size()) {
                throw std::invalid_argument("OuterVectorProduct: Output matrix A has incorrect shape.");
            }
            OuterVectorProduct(x.size(), x.data(), incx, y.size(), y.data(), incy, alpha, A.data());
        }

        /// @brief Outer product explicit specialization for float precision.
        ///
        /// Performs A = alpha * x * y^T where x is a vector of size M and y is a vector of size N,
        /// producing an M x N matrix. This specialization wrapper accepts Vector<float> and Matrix<float> typed storage
        /// and forwards to the lower-level pointer-based implementation.
        ///
        /// @param x First input vector (Vector<float>).
        /// @param y Second input vector (Vector<float>).
        /// @param A Output matrix to store the outer product (Matrix<float>).
        /// @param incx Stride between elements of vector x.
        /// @param incy Stride between elements of vector y.
        /// @param alpha Scalar multiplier for the outer product.
        template <>
        void OuterVectorProduct<float>(const Vector<float>& x, const Vector<float>& y, Matrix<float>& A, size_t incx, size_t incy, const float alpha) {
            if (A.shape().first != x.size() || A.shape().second != y.size()) {
                throw std::invalid_argument("OuterVectorProduct: Output matrix A has incorrect shape.");
            }
            OuterVectorProduct(x.size(), x.data(), incx, y.size(), y.data(), incy, alpha, A.data());
        }
       
       
        /// @brief Matrix-vector multiply (GEMV-like), explicit specialization for double precision.
        ///
        /// Performs y := alpha * op(A) * x + beta * y where op(A) is A, A^T, or A^H depending on `Ta`.
        /// This routine computes general matrix-vector products with optional transpose and conjugate-transpose.
        /// Explicit specialization wrapping BLAS function dgemv.
        ///
        /// @param T Transpose option for A: "N" (no transpose, A), "T" (transpose, A^T), "C" (conjugate-transpose, A^H).
        /// @param alpha Scaling factor applied to op(A)*x.
        /// @param a Left-hand input matrix (Matrix_<double>).
        /// @param x Input vector (Vector_<double>).
        /// @param incx Stride between consecutive elements in vector x.
        /// @param beta Scaling factor applied to existing contents of y.
        /// @param y Output vector (destination / input-output, Vector_<double>).
        /// @param incy Stride between consecutive elements in vector y.
        template <>
        void MatrixVectorProduct<double>(const char *T, const double alpha, const Matrix_<double> &a,
                                 const Vector_<double> &x, const size_t incx, const double beta, Vector_<double> &y, const size_t incy)
        {
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;
            CBLAS_TRANSPOSE trans = get_trans(T);
            int nrow, ncol;
            std::tie(nrow, ncol) = check_size_mv(a, x, y, trans);
            BLAS_INT lda = get_leading(nrow, ncol);

            cblas_dgemv(major, trans, nrow, ncol, alpha, a.data(), lda, x.data(), inx, beta, y.data(), iny);
        };

        /// @brief Matrix-vector multiply convenience overload with default parameters (double precision).
        ///
        /// Convenience overload where matrix A and vector x are provided first, with optional parameters
        /// defaulting to no transpose, alpha=1.0, beta=0.0, and unit strides.
        ///
        /// @param a Left-hand input matrix (Matrix_<double>).
        /// @param x Input vector (Vector_<double>).
        /// @param y Output vector (destination / input-output, Vector_<double>).
        /// @param Ta Transpose option for A: "N" (default, no transpose), "T" (transpose, A^T), "C" (conjugate-transpose, A^H).
        /// @param alpha Scaling factor applied to op(A)*x. Default: 1.0.
        /// @param beta Scaling factor applied to existing contents of y. Default: 0.0.
        /// @param incx Stride between consecutive elements in vector x. Default: 1.
        /// @param incy Stride between consecutive elements in vector y. Default: 1.
        void MatrixVectorProduct(const Matrix_<double> &a, const Vector_<double> &x, Vector_<double> &y, const char *T,
                                 const double alpha, const double beta, const size_t incx, const size_t incy)
        {
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;
            CBLAS_TRANSPOSE trans = get_trans(T);
            int nrow, ncol;
            std::tie(nrow, ncol) = check_size_mv(a, x, y, trans);
            BLAS_INT lda = get_leading(nrow, ncol);
            cblas_dgemv(major, trans, nrow, ncol, alpha, a.data(), lda, x.data(), inx, beta, y.data(), iny);
        };

        /// @brief Matrix-vector multiply (GEMV-like), explicit specialization for single precision.
        ///
        /// Performs y := alpha * op(A) * x + beta * y where op(A) is A, A^T, or A^H depending on `Ta`.
        /// This routine computes general matrix-vector products with optional transpose and conjugate-transpose.
        /// Explicit specialization wrapping BLAS function sgemv.
        ///
        /// @param T Transpose option for A: "N" (no transpose, A), "T" (transpose, A^T), "C" (conjugate-transpose, A^H).
        /// @param alpha Scaling factor applied to op(A)*x.
        /// @param a Left-hand input matrix (Matrix_<float>).
        /// @param x Input vector (Vector_<float>).
        /// @param incx Stride between consecutive elements in vector x.
        /// @param beta Scaling factor applied to existing contents of y.
        /// @param y Output vector (destination / input-output, Vector_<float>).
        /// @param incy Stride between consecutive elements in vector y.
        template <>
        void MatrixVectorProduct<float>(const char *T, const float alpha, const Matrix_<float> &a,
                                 const Vector_<float> &x, const size_t incx, const float beta, Vector_<float> &y, const size_t incy)
        {
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;
            CBLAS_TRANSPOSE trans = get_trans(T);
            int nrow, ncol;
            std::tie(nrow, ncol) = check_size_mv(a, x, y, trans);
            BLAS_INT lda = get_leading(nrow, ncol);
            cblas_sgemv(major, trans, nrow, ncol, alpha, a.data(), lda, x.data(), inx, beta, y.data(), iny);
        };

        /// @brief Matrix-vector multiply convenience overload with default parameters (float precision).
        ///
        /// Convenience overload where matrix A and vector x are provided first, with optional parameters
        /// defaulting to no transpose, alpha=1.0, beta=0.0, and unit strides.
        ///
        /// @param a Left-hand input matrix (Matrix_<float>).
        /// @param x Input vector (Vector_<float>).
        /// @param y Output vector (destination / input-output, Vector_<float>).
        /// @param Ta Transpose option for A: "N" (default, no transpose), "T" (transpose, A^T), "C" (conjugate-transpose, A^H).
        /// @param alpha Scaling factor applied to op(A)*x. Default: 1.0.
        /// @param beta Scaling factor applied to existing contents of y. Default: 0.0.
        /// @param incx Stride between consecutive elements in vector x. Default: 1.
        /// @param incy Stride between consecutive elements in vector y. Default: 1.
        void MatrixVectorProduct(const Matrix_<float> &a, const Vector_<float> &x, Vector_<float> &y, const char *T,
                                 const float alpha, const float beta, const size_t incx, const size_t incy)
        {
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;
            CBLAS_TRANSPOSE trans = get_trans(T);
            int nrow, ncol;
            std::tie(nrow, ncol) = check_size_mv(a, x, y, trans);
            BLAS_INT lda = get_leading(nrow, ncol);
            cblas_sgemv(major, trans, nrow, ncol, alpha, a.data(), lda, x.data(), inx, beta, y.data(), iny);
        };

        /// @brief Matrix-vector multiply (GEMV-like), explicit specialization for complex double precision.
        ///
        /// Performs y := alpha * op(A) * x + beta * y where op(A) is A, A^T, or A^H depending on `Ta`.
        /// This routine computes general matrix-vector products with optional transpose and conjugate-transpose.
        /// Explicit specialization wrapping BLAS function zgemv.
        ///
        /// @param T Transpose option for A: "N" (no transpose, A), "T" (transpose, A^T), "C" (conjugate-transpose, A^H).
        /// @param alpha Scaling factor applied to op(A)*x.
        /// @param a Left-hand input matrix (Matrix_<complex_double>).
        /// @param x Input vector (Vector_<complex_double>).
        /// @param incx Stride between consecutive elements in vector x.
        /// @param beta Scaling factor applied to existing contents of y.
        /// @param y Output vector (destination / input-output, Vector_<complex_double>).
        /// @param incy Stride between consecutive elements in vector y.
        template <>
        void MatrixVectorProduct<complex_double>(const char *T, const complex_double alpha, const Matrix_<complex_double> &a, const Vector_<complex_double> &x, 
                                 const size_t incx, const complex_double beta, Vector_<complex_double> &y, const size_t incy)
        {
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;
            CBLAS_TRANSPOSE trans = get_trans(T);
            int nrow, ncol;
            std::tie(nrow, ncol) = check_size_mv(a, x, y, trans);
            BLAS_INT lda = get_leading(nrow, ncol);

            cblas_zgemv(major, trans, nrow, ncol, &alpha, a.data(), lda, x.data(), inx, &beta, y.data(), iny);
        };

        /// @brief Matrix-vector multiply convenience overload with default parameters (complex_double precision).
        ///
        /// Convenience overload where matrix A and vector x are provided first, with optional parameters
        /// defaulting to no transpose, alpha=1.0, beta=0.0, and unit strides.
        ///
        /// @param a Left-hand input matrix (Matrix_<complex_double>).
        /// @param x Input vector (Vector_<complex_double>).
        /// @param y Output vector (destination / input-output, Vector_<complex_double>).
        /// @param Ta Transpose option for A: "N" (default, no transpose), "T" (transpose, A^T), "C" (conjugate-transpose, A^H).
        /// @param alpha Scaling factor applied to op(A)*x. Default: 1.0.
        /// @param beta Scaling factor applied to existing contents of y. Default: 0.0.
        /// @param incx Stride between consecutive elements in vector x. Default: 1.
        /// @param incy Stride between consecutive elements in vector y. Default: 1.
        void MatrixVectorProduct(const Matrix_<complex_double> &a, const Vector_<complex_double> &x, Vector_<complex_double> &y, const char *T,
                                 const complex_double alpha, const complex_double beta, const size_t incx, const size_t incy)
        {
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;
            CBLAS_TRANSPOSE trans = get_trans(T);
            int nrow, ncol;
            std::tie(nrow, ncol) = check_size_mv(a, x, y, trans);
            BLAS_INT lda = get_leading(nrow, ncol);
            cblas_zgemv(major, trans, nrow, ncol, &alpha, a.data(), lda, x.data(), inx, &beta, y.data(), iny);
        };

        /// @brief Matrix-vector multiply (GEMV-like), explicit specialization for complex float precision.
        ///
        /// Performs y := alpha * op(A) * x + beta * y where op(A) is A, A^T, or A^H depending on `Ta`.
        /// This routine computes general matrix-vector products with optional transpose and conjugate-transpose.
        /// Explicit specialization wrapping BLAS function cgemv.
        ///
        /// @param T Transpose option for A: "N" (no transpose, A), "T" (transpose, A^T), "C" (conjugate-transpose, A^H).
        /// @param alpha Scaling factor applied to op(A)*x.
        /// @param a Left-hand input matrix (Matrix_<complex_float>).
        /// @param x Input vector (Vector_<complex_float>).
        /// @param incx Stride between consecutive elements in vector x.
        /// @param beta Scaling factor applied to existing contents of y.
        /// @param y Output vector (destination / input-output, Vector_<complex_float>).
        /// @param incy Stride between consecutive elements in vector y.
        template <>
        void MatrixVectorProduct<complex_float>(const char *T, const complex_float alpha, const Matrix_<complex_float> &a, const Vector_<complex_float> &x, 
                                 const size_t incx, const complex_float beta, Vector_<complex_float> &y, const size_t incy)
        {
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;
            CBLAS_TRANSPOSE trans = get_trans(T);
            int nrow, ncol;
            std::tie(nrow, ncol) = check_size_mv(a, x, y, trans);
            BLAS_INT lda = get_leading(nrow, ncol);
            cblas_cgemv(major, trans, nrow, ncol, &alpha, a.data(), lda, x.data(), inx, &beta, y.data(), iny);
        };

        /// @brief Matrix-vector multiply convenience overload with default parameters (complex_float precision).
        ///
        /// Convenience overload where matrix A and vector x are provided first, with optional parameters
        /// defaulting to no transpose, alpha=1.0, beta=0.0, and unit strides.
        ///
        /// @param a Left-hand input matrix (Matrix_<complex_float>).
        /// @param x Input vector (Vector_<complex_float>).
        /// @param y Output vector (destination / input-output, Vector_<complex_float>).
        /// @param Ta Transpose option for A: "N" (default, no transpose), "T" (transpose, A^T), "C" (conjugate-transpose, A^H).
        /// @param alpha Scaling factor applied to op(A)*x. Default: 1.0.
        /// @param beta Scaling factor applied to existing contents of y. Default: 0.0.
        /// @param incx Stride between consecutive elements in vector x. Default: 1.
        /// @param incy Stride between consecutive elements in vector y. Default: 1.
        void MatrixVectorProduct(const Matrix_<complex_float> &a, const Vector_<complex_float> &x, Vector_<complex_float> &y, const char *T,
                                 const complex_float alpha, const complex_float beta, const size_t incx, const size_t incy)
        {
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;
            CBLAS_TRANSPOSE trans = get_trans(T);
            int nrow, ncol;
            std::tie(nrow, ncol) = check_size_mv(a, x, y, trans);
            BLAS_INT lda = get_leading(nrow, ncol);
            cblas_cgemv(major, trans, nrow, ncol, &alpha, a.data(), lda, x.data(), inx, &beta, y.data(), iny);
        };

        /// @brief Symmetric matrix-vector multiply (SYMV-like), wrapper to BLAS function dsymv (double).
        ///
        /// Performs y := alpha * A * x + beta * y where A is a symmetric matrix.
        /// Only the lower or upper triangular part of A is used (determined by project configuration).
        ///
        /// @param alpha Scaling factor applied to A*x.
        /// @param a Symmetric input matrix (Matrix_<double>).
        /// @param x Input vector (Vector_<double>).
        /// @param incx Stride between consecutive elements in vector x.
        /// @param beta Scaling factor applied to existing contents of y.
        /// @param y Output vector (destination / input-output, Vector_<double>).
        /// @param incy Stride between consecutive elements in vector y.
        void SymMatrixVectorProduct(const double alpha, const Matrix_<double> &a,
                                    const Vector_<double> &x, const size_t incx, const double beta, Vector_<double> &y, const size_t incy)
        {
            int nrow, ncol;
            std::tie(nrow, ncol) = check_size_mv(a, x, y);
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;

            BLAS_INT lda = get_leading(nrow, ncol);

            cblas_dsymv(major, tri, ncol, alpha, a.data(), lda, x.data(), inx, beta, y.data(), iny);
        };



        /// @brief Symmetric matrix-vector multiply (SYMV-like), wrapper to BLAS function ssymv (float).
        ///
        /// Performs y := alpha * A * x + beta * y where A is a symmetric matrix.
        /// Only the lower or upper triangular part of A is used (determined by project configuration).
        ///
        /// @param alpha Scaling factor applied to A*x.
        /// @param a Symmetric input matrix (Matrix_<float>).
        /// @param x Input vector (Vector_<float>).
        /// @param incx Stride between consecutive elements in vector x.
        /// @param beta Scaling factor applied to existing contents of y.
        /// @param y Output vector (destination / input-output, Vector_<float>).
        /// @param incy Stride between consecutive elements in vector y.
        void SymMatrixVectorProduct(const float alpha, const Matrix_<float> &a,
                                    const Vector_<float> &x, const size_t incx, const float beta, Vector_<float> &y, const size_t incy)
        {
            int nrow, ncol;
            std::tie(nrow, ncol) = check_size_mv(a, x, y);
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;

            BLAS_INT lda = get_leading(nrow, ncol);

            cblas_ssymv(major, tri, ncol, alpha, a.data(), lda, x.data(), inx, beta, y.data(), iny);
        };



        /// @brief Symmetric matrix-vector multiply using packed storage (SPMV-like), wrapper to BLAS function dspmv (double).
        ///
        /// Performs y := alpha * A * x + beta * y where A is a symmetric matrix stored in packed format.
        /// Only the lower or upper triangular part of A is used (determined by project configuration).
        ///
        /// @param alpha Scaling factor applied to A*x.
        /// @param a Symmetric input matrix in packed storage format (LowTriMatrix_<double>).
        /// @param x Input vector (Vector_<double>).
        /// @param incx Stride between consecutive elements in vector x.
        /// @param beta Scaling factor applied to existing contents of y.
        /// @param y Output vector (destination / input-output, Vector_<double>).
        /// @param incy Stride between consecutive elements in vector y.
        void SymMatrixVectorProduct(const double alpha, const LowTriMatrix_<double> &a,
                                    const Vector_<double> &x, const size_t incx, const double beta, Vector_<double> &y, const size_t incy)
        {
            int nrow, ncol;
            std::tie(nrow, ncol) = check_size_mv(a, x, y);
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;

            cblas_dspmv(major, tri, ncol, alpha, a.data(), x.data(), inx, beta, y.data(), iny);
        };



        /// @brief Symmetric matrix-vector multiply using packed storage (SPMV-like), wrapper to BLAS function sspmv (float).
        ///
        /// Performs y := alpha * A * x + beta * y where A is a symmetric matrix stored in packed format.
        /// Only the lower or upper triangular part of A is used (determined by project configuration).
        ///
        /// @param alpha Scaling factor applied to A*x.
        /// @param a Symmetric input matrix in packed storage format (LowTriMatrix_<float>).
        /// @param x Input vector (Vector_<float>).
        /// @param incx Stride between consecutive elements in vector x.
        /// @param beta Scaling factor applied to existing contents of y.
        /// @param y Output vector (destination / input-output, Vector_<float>).
        /// @param incy Stride between consecutive elements in vector y.
        void SymMatrixVectorProduct(const float alpha, const LowTriMatrix_<float> &a,
                                    const Vector_<float> &x, const size_t incx, const float beta, Vector_<float> &y, const size_t incy)
        {
            int nrow, ncol;
            std::tie(nrow, ncol) = check_size_mv(a, x, y);
            BLAS_INT inx = incx;
            BLAS_INT iny = incy;

            cblas_sspmv(major, tri, ncol, alpha, a.data(), x.data(), inx, beta, y.data(), iny);
        };



        /// @brief Triangular matrix-vector multiply (TPMV-like), wrapper to BLAS function dtpmv (double).
        ///
        /// Performs x := op(A) * x where A is a triangular matrix stored in packed format and op(A) is A, A^T, or A^H.
        /// Overwrites the input vector x with the result.
        ///
        /// @param Ta Transpose option for A: "N" (no transpose, A), "T" (transpose, A^T), "C" (conjugate-transpose, A^H).
        /// @param unit Specifies whether A is a unit triangular matrix (CblasUnit) or not (CblasNonUnit).
        /// @param a Input triangular matrix in packed storage format (LowTriMatrix_<double>).
        /// @param x Input-output vector (overwritten with result).
        /// @param incx Stride between consecutive elements in vector x.
        void LowTriMatrixVectorProduct(const char *T, const CBLAS_DIAG unit, const LowTriMatrix_<double> &a, Vector_<double> &x, const size_t incx)
        {
            BLAS_INT inx = incx;
            CBLAS_TRANSPOSE trans = get_trans(T);

            int row, col;
            std::tie(row, col) = check_size_mv(a, x);

            cblas_dtpmv(major, tri, trans, unit, col, a.data(), x.data(), inx);
        };



        /// @brief Triangular matrix-vector multiply (TPMV-like), wrapper to BLAS function stpmv (float).
        ///
        /// Performs x := op(A) * x where A is a triangular matrix stored in packed format and op(A) is A, A^T, or A^H.
        /// Overwrites the input vector x with the result.
        ///
        /// @param Ta Transpose option for A: "N" (no transpose, A), "T" (transpose, A^T), "C" (conjugate-transpose, A^H).
        /// @param unit Specifies whether A is a unit triangular matrix (CblasUnit) or not (CblasNonUnit).
        /// @param a Input triangular matrix in packed storage format (LowTriMatrix_<float>).
        /// @param x Input-output vector (overwritten with result).
        /// @param incx Stride between consecutive elements in vector x.
        void LowTriMatrixVectorProduct(const char *T, const CBLAS_DIAG unit, const LowTriMatrix_<float> &a, Vector_<float> &x, const size_t incx)
        {
            BLAS_INT inx = incx;
            CBLAS_TRANSPOSE trans = get_trans(T);

            int row, col;
            std::tie(row, col) = check_size_mv(a, x);

            cblas_stpmv(major, tri, trans, unit, col, a.data(), x.data(), inx);
        };

    } // namespace cpu
} // namespace lahva