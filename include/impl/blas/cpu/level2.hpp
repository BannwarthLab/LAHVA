/// @file level2.hpp
/// @brief C++-style BLAS Level-2 operations (matrix-vector operations).
///
// Level-2 BLAS-like operations (matrix * vector) - C++ template declarations
// These functions are C++-style wrappers operating on Matrix_<>, Vector_<>, and LowTriMatrix_<>
// objects. They provide convenient overloads for different scalar types and dispatch to optimized
// CPU BLAS kernels. Each overload is provided for double and float precision; both precisions
// are documented explicitly.

#pragma once
#include "const.h"
#include "linalg.hpp"

namespace lahva
{
    namespace cpu
    {
    void OuterVectorProduct(const Vector<double>& x, const Vector<double>& y, Matrix<double>& A, size_t incx = 1, size_t incy = 1, const double alpha = 1.0);
    void OuterVectorProduct(const Vector<float>& x, const Vector<float>& y, Matrix<float>& A, size_t incx = 1, size_t incy = 1, const float alpha = 1.0);

    void MatrixVectorProduct(const char* T, const double alpha, const Matrix_<double>& a, const Vector_<double>& x, 
                             const size_t incx, const double beta, Vector_<double>& y, const size_t incy);
    void MatrixVectorProduct(const Matrix_<double>& a, const Vector_<double>& x, Vector_<double>& y,
                             const char* T = "N", const double alpha = 1.0, const double beta = 0.0, const size_t incx = 1, const size_t incy = 1);
    void MatrixVectorProduct(const char* T, const float alpha, const Matrix_<float>& a, const Vector_<float>& x, 
                             const size_t incx, const float beta, Vector_<float>& y, const size_t incy);
    void MatrixVectorProduct(const Matrix_<float>& a, const Vector_<float>& x, Vector_<float>& y,
                             const char* T = "N", const float alpha = 1.0, const float beta = 0.0, const size_t incx = 1, const size_t incy = 1);

    void MatrixVectorProduct(const char* T, const complex_double alpha, const Matrix_<complex_double>& a, const Vector_<complex_double>& x, 
                             const size_t incx, const complex_double beta, Vector_<complex_double>& y, const size_t incy);
    void MatrixVectorProduct(const Matrix_<complex_double>& a, const Vector_<complex_double>& x, Vector_<complex_double>& y,
                             const char* T = "N", const complex_double alpha = 1.0, const complex_double beta = 0.0, const size_t incx = 1, const size_t incy = 1);
    void MatrixVectorProduct(const char* T, const complex_float alpha, const Matrix_<complex_float>& a, const Vector_<complex_float>& x, 
                             const size_t incx, const complex_float beta, Vector_<complex_float>& y, const size_t incy);
    void MatrixVectorProduct(const Matrix_<complex_float>& a, const Vector_<complex_float>& x, Vector_<complex_float>& y,
                             const char* T = "N", const complex_float alpha = 1.0, const complex_float beta = 0.0, const size_t incx = 1, const size_t incy = 1);


        /// @brief Symmetric matrix-vector multiply (double precision), wrapper to BLAS function dsymv.
        ///
        /// Computes y := alpha * A * x + beta * y where A is a symmetric matrix stored in
        /// lower triangular packed format. This C++ template wrapper accepts Matrix_<double> and
        /// Vector_<double> typed storage and forwards to the lower-level pointer-based implementation.
        ///
        /// @param alpha Scaling factor applied to A*x.
        /// @param a     Symmetric matrix (Matrix_<double>), n x n.
        /// @param x     Input vector (Vector_<double>), size n.
        /// @param incx  Stride for x.
        /// @param beta  Scaling factor applied to existing contents of y.
        /// @param y     Output vector (destination / input-output, Vector_<double>), size n.
        /// @param incy  Stride for y.
        void SymMatrixVectorProduct(const double alpha, const Matrix_<double> &a, const Vector_<double> &x,
                                    const size_t incx, const double beta, Vector_<double> &y, const size_t incy);
    void SymMatrixVectorProduct(const Matrix_<double>& a, const Vector_<double>& x, Vector_<double>& y,
                             const double alpha = 1.0, const double beta = 0.0, const size_t incx = 1, const size_t incy = 1);

        /// @brief Symmetric matrix-vector multiply (single precision), wrapper to BLAS function ssymv.
        ///
        /// Computes y := alpha * A * x + beta * y where A is a symmetric matrix stored in
        /// lower triangular packed format. Float variant of the double-precision version.
        ///
        /// @param alpha Scaling factor applied to A*x.
        /// @param a     Symmetric matrix (Matrix_<float>), n x n.
        /// @param x     Input vector (Vector_<float>), size n.
        /// @param incx  Stride for x.
        /// @param beta  Scaling factor applied to existing contents of y.
        /// @param y     Output vector (destination / input-output, Vector_<float>), size n.
        /// @param incy  Stride for y.
        void SymMatrixVectorProduct(const float alpha, const Matrix_<float> &a, const Vector_<float> &x,
                                    const size_t incx, const float beta, Vector_<float> &y, const size_t incy);
    void SymMatrixVectorProduct(const Matrix_<float>& a, const Vector_<float>& x, Vector_<float>& y,
                             const float alpha = 1.0, const float beta = 0.0, const size_t incx = 1, const size_t incy = 1);

        /// @brief Symmetric matrix-vector multiply using lower-triangular storage (double), wrapper to BLAS function dsymv.
        ///
        /// Variant that accepts a `LowTriMatrix_<double>` representation (lower-triangular
        /// packed or compact storage). Semantically equivalent to SymMatrixVectorProduct
        /// but specialized for the project's LowTriMatrix_ layout.
        ///
        /// @param alpha Scaling factor applied to A*x.
        /// @param a     Lower-triangular matrix (LowTriMatrix_<double>), n x n.
        /// @param x     Input vector (Vector_<double>), size n.
        /// @param incx  Stride for x.
        /// @param beta  Scaling factor applied to existing contents of y.
        /// @param y     Output vector (destination / input-output, Vector_<double>), size n.
        /// @param incy  Stride for y.
        void SymMatrixVectorProduct(const double alpha, const LowTriMatrix_<double> &a, const Vector_<double> &x,
                                    const size_t incx, const double beta, Vector_<double> &y, const size_t incy);
    void SymMatrixVectorProduct(const LowTriMatrix_<double>& a, const Vector_<double>& x, Vector_<double>& y,
                             const double alpha = 1.0, const double beta = 0.0, const size_t incx = 1, const size_t incy = 1);

        /// @brief Symmetric matrix-vector multiply using lower-triangular storage (float), wrapper to BLAS function ssymv.
        ///
        /// Float variant specialized for `LowTriMatrix_<float>` storage.
        ///
        /// @param alpha Scaling factor applied to A*x.
        /// @param a     Lower-triangular matrix (LowTriMatrix_<float>), n x n.
        /// @param x     Input vector (Vector_<float>), size n.
        /// @param incx  Stride for x.
        /// @param beta  Scaling factor applied to existing contents of y.
        /// @param y     Output vector (destination / input-output, Vector_<float>), size n.
        /// @param incy  Stride for y.
        void SymMatrixVectorProduct(const float alpha, const LowTriMatrix_<float> &a, const Vector_<float> &x,
                                    const size_t incx, const float beta, Vector_<float> &y, const size_t incy);
    void SymMatrixVectorProduct(const LowTriMatrix_<float>& a, const Vector_<float>& x, Vector_<float>& y,
                             const float alpha = 1.0, const float beta = 0.0, const size_t incx = 1, const size_t incy = 1);

        /// @brief Lower-triangular matrix-vector product / triangular matrix-vector multiply (double), wrapper to BLAS function dtrmv.
        ///
        /// Performs x := op(T) * x where T is a lower-triangular matrix stored in `LowTriMatrix_<double>`.
        /// The parameter `T` controls transpose ("N", "T", "C") and `unit` indicates whether the diagonal
        /// is assumed unit.
        ///
        /// @param T    Transpose option for the triangular matrix: "N" (no transpose), "T" (transpose), "C" (conjugate-transpose).
        /// @param unit Whether the diagonal is unit (CblasUnit/CblasNonUnit).
        /// @param a    Lower-triangular matrix (LowTriMatrix_<double>), n x n.
        /// @param x    Vector to be multiplied in-place (Vector_<double>), size n.
        /// @param incx Stride for x.
        void LowTriMatrixVectorProduct(const char *T, const CBLAS_DIAG unit, const LowTriMatrix_<double> &a, Vector_<double> &x, const size_t incx);
    void LowTriMatrixVectorProduct(const LowTriMatrix_<double>& a, Vector_<double>& x, const char* T = "N", const CBLAS_DIAG unit = CblasNonUnit, const size_t incx = 1);
        /// @brief Lower-triangular matrix-vector product / triangular matrix-vector multiply (float), wrapper to BLAS function strmv.
        ///
        /// Float variant of the lower-triangular multiply.
        ///
        /// @param T    Transpose option for the triangular matrix: "N" (no transpose, A), "T" (transpose, A^T), "C" (conjugate-transpose, A^H).
        /// @param unit Whether the diagonal is unit (CblasUnit/CblasNonUnit).
        /// @param a    Lower-triangular matrix (LowTriMatrix_<float>), n x n.
        /// @param x    Vector to be multiplied in-place (Vector_<float>), size n.
        /// @param incx Stride for x.
        void LowTriMatrixVectorProduct(const char *T, const CBLAS_DIAG unit, const LowTriMatrix_<float> &a, Vector_<float> &x, const size_t incx);
    void LowTriMatrixVectorProduct(const LowTriMatrix_<float>& a, Vector_<float>& x, const char* T = "N", const CBLAS_DIAG unit = CblasNonUnit, const size_t incx = 1);

        /// @brief Runtime-dispatching overload that allows passing a CPURuntime first.
        ///
        /// The runtime parameter is ignored for CPU implementations but this overload
        /// mirrors the signatures used for GPU/CUDA dispatching.
        template <typename... Args>
        void OuterVectorProduct(const CPURuntime &rt_, Args &&...args)
        {
            (OuterVectorProduct(args...));
        }

        /// @brief Runtime-dispatching overload that allows passing a CPURuntime first.
        ///
        /// The runtime parameter is ignored for CPU implementations but this overload
        /// mirrors the signatures used for GPU/CUDA dispatching.
        template <typename... Args>
        void LowTriMatrixVectorProduct(const CPURuntime &rt_, Args &&...args)
        {
            (LowTriMatrixVectorProduct(args...));
        }

        /// @brief Runtime-dispatching overload that allows passing a CPURuntime first.
        ///
        /// The runtime parameter is ignored for CPU implementations but this overload
        /// mirrors the signatures used for GPU/CUDA dispatching.
        template <typename... Args>
        void MatrixVectorProduct(const CPURuntime &rt_, Args &&...args)
        {
            (MatrixVectorProduct(args...));
        }

        /// @brief Runtime-dispatching overload that allows passing a CPURuntime first.
        ///
        /// The runtime parameter is ignored for CPU implementations but this overload
        /// mirrors the signatures used for GPU/CUDA dispatching.
        template <typename... Args>
        void SymMatrixVectorProduct(const CPURuntime &rt_, Args &&...args)
        {
            (SymMatrixVectorProduct(args...));
        }

    }
}
