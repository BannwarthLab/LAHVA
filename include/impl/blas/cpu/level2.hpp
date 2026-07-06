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
        /// @brief Outer product of two vectors, wrapper to BLAS functions dger (double), sger (float), zger (complex_double), cger (complex_float).
        ///
        /// Performs A = alpha * x * y^T where x is a vector of size M and y is a vector of size N,
        /// producing an M x N matrix. This C++ template wrapper accepts Vector_<T> typed storage
        /// and forwards to the lower-level pointer-based implementation that calls the appropriate
        /// BLAS routine.
        ///
        /// @tparam T Element type (double, float, complex_double, complex_float).
        /// @param x     First input vector (Vector_<T>).
        /// @param y     Second input vector (Vector_<T>).
        /// @param A     Output matrix to store the outer product (Matrix_<T>).
        /// @param incx  Stride between elements of vector x. Default: 1
        /// @param incy  Stride between elements of vector y. Default: 1
        /// @param alpha Scalar multiplier for the outer product. Default: 1.0
        template <typename T>
        void OuterVectorProduct(const Vector<T> &x, const Vector<T> &y, Matrix<T> &A, size_t incx = 1, size_t incy = 1, const T alpha = (T)1.0);

        /// @brief Outer product convenience overloads accepting different scalar types.
        ///
        /// Allows calling the outer product wrapper with scalar types that differ from the vector
        /// element type (for example, passing double literals when T==float). Scalars are converted
        /// to `T` before forwarding. Dispatches to the appropriate BLAS routine (dger, sger, zger, cger)
        /// based on `T`.
        ///
        /// @tparam Scalar Scalar type for `alpha`.
        /// @tparam T      Element type for vectors and matrix.
        /// @param x     First input vector (Vector_<T>).
        /// @param y     Second input vector (Vector_<T>).
        /// @param A     Output matrix to store the outer product (Matrix_<T>).
        /// @param incx  Stride between elements of vector x. Default: 1.
        /// @param incy  Stride between elements of vector y. Default: 1.
        /// @param alpha Scalar multiplier for the outer product (will be cast to T). Default: 1.0.
        template <typename Scalar, typename T>
        void OuterVectorProduct(const Vector<T> &x, const Vector<T> &y, Matrix<T> &A, size_t incx = 1, size_t incy = 1, const Scalar alpha = (Scalar)1.0)
        {
            OuterVectorProduct(x, y, A, incx, incy, static_cast<T>(alpha));
        };

        /// @brief Matrix-vector multiply (GEMV-like), wrapper to BLAS functions dgemv (double), sgemv (float), zgemv (complex_double), cgemv (complex_float).
        ///
        /// Performs y := alpha * op(A) * x + beta * y where op(A) is A, A^T, or A^H depending on `Ta`.
        /// This C++ template wrapper accepts Matrix_/Vector_ typed storage and forwards to the lower-level
        /// pointer-based implementation that calls the appropriate BLAS routine.
        ///
        /// @tparam T Element type (double, float, complex_double, complex_float).
        /// @param Ta    Transpose option for A: "N" (no transpose, A), "T" (transpose, A^T), "C" (conjugate-transpose, A^H).
        /// @param alpha Scaling factor applied to op(A)*x.
        /// @param a     Left-hand input matrix (Matrix_<T>).
        /// @param x     Input vector (Vector_<T>).
        /// @param incx  Stride for x (in elements).
        /// @param beta  Scaling factor applied to existing contents of y.
        /// @param y     Output vector (destination / input-output, Vector_<T>).
        /// @param incy  Stride for y (in elements).
        template <typename T>
        void MatrixVectorProduct(const char *Ta, const T alpha, const Matrix_<T> &a, const Vector_<T> &x,
                                 const size_t incx, const T beta, Vector_<T> &y, const size_t incy);

        /// @brief Matrix-vector multiply convenience overloads accepting different scalar types.
        ///
        /// Allows calling the matrix-vector wrapper with scalar types that differ from the matrix
        /// element type (for example, passing double literals when T==float). Scalars are converted
        /// to `T` before forwarding. Dispatches to the appropriate BLAS routine based on `T`.
        ///
        /// @tparam Scale  Scalar type for `alpha`.
        /// @tparam Scale2 Scalar type for `beta`.
        /// @tparam T      Element type for matrix and vectors.
        /// @param Ta    Transpose option for A: "N" (no transpose, A), "T" (transpose, A^T), "C" (conjugate-transpose, A^H).
        /// @param alpha Scaling factor (will be cast to T).
        /// @param a     Left-hand input matrix (Matrix_<T>).
        /// @param x     Input vector (Vector_<T>).
        /// @param incx  Stride for x (in elements).
        /// @param beta  Scaling factor (will be cast to T).
        /// @param y     Output vector (destination / input-output, Vector_<T>).
        /// @param incy  Stride for y (in elements).
        template <typename Scale, typename Scale2, typename T>
        void MatrixVectorProduct(const char *Ta, const Scale alpha, const Matrix_<T> &a, const Vector_<T> &x,
                                 const size_t incx, const Scale2 beta, Vector_<T> &y, const size_t incy)
        {
            MatrixVectorProduct(Ta, static_cast<T>(alpha), a, x, incx, static_cast<T>(beta), y, incy);
        };

        /// @brief Matrix-vector multiply convenience overload with defaults.
        ///
        /// Provides a simplified interface for the common case of y := A*x by defaulting
        /// transpose to "N", alpha to 1, and beta to 0. This allows calling with just
        /// the matrix and vectors for the most common usage pattern. Dispatches to the appropriate
        /// BLAS routine (dgemv, sgemv, zgemv, cgemv) based on `T`.
        ///
        /// @tparam T Element type for matrix and vectors.
        /// @param a     Left-hand input matrix (Matrix_<T>).
        /// @param x     Input vector (Vector_<T>).
        /// @param y     Output vector (destination / input-output, Vector_<T>).
        /// @param Ta    Transpose option for A: "N" (no transpose, A), "T" (transpose, A^T), "C" (conjugate-transpose, A^H). Default: "N".
        /// @param alpha Scaling factor applied to op(A)*x. Default: 1.0.
        /// @param beta  Scaling factor applied to existing contents of y. Default: 0.0.
        /// @param incx  Stride for x (in elements). Default: 1.
        /// @param incy  Stride for y (in elements). Default: 1.
        template <typename T>
        inline void MatrixVectorProduct(const Matrix_<T> &a, const Vector_<T> &x, Vector_<T> &y,
                                        const char *Ta = "N", const T alpha = static_cast<T>(1.0), const T beta = static_cast<T>(0.0), const size_t incx = 1, const size_t incy = 1)
        {
            MatrixVectorProduct(Ta, alpha, a, x, incx, beta, y, incy);
        };

        /// @brief Simplified convenience wrapper: y := alpha*A*x + beta*y with defaults.
        ///
        /// This template overload uses default transpose="N", alpha=1 and beta=0 to
        /// support the common case y := A*x, with convenience scalar type conversion.
        ///
        /// @tparam Scalar Scalar type for `alpha`.
        /// @tparam Scalar2 Scalar type for `beta`.
        /// @tparam T      Element type for matrix and vectors.
        /// @param a     Left-hand input matrix (Matrix_<T>).
        /// @param x     Input vector (Vector_<T>).
        /// @param y     Output vector (destination / input-output, Vector_<T>).
        /// @param Ta    Transpose option for A: "N" (no transpose, A), "T" (transpose, A^T), "C" (conjugate-transpose, A^H). Default: "N".
        /// @param alpha Scaling factor applied to op(A)*x (will be cast to T). Default: 1.0.
        /// @param beta  Scaling factor applied to existing contents of y (will be cast to T). Default: 0.0.
        /// @param incx  Stride for x (in elements). Default: 1.
        /// @param incy  Stride for y (in elements). Default: 1.
        template <typename Scalar, typename Scalar2, typename T>
        void MatrixVectorProduct(const Matrix_<T> &a, const Vector_<T> &x, Vector_<T> &y,
                                 const char *Ta = "N", const Scalar alpha = static_cast<Scalar>(1.0), const Scalar2 beta = static_cast<Scalar2>(0.0), const size_t incx = 1, const size_t incy = 1)
        {
            MatrixVectorProduct(Ta, static_cast<T>(alpha), a, x, incx, static_cast<T>(beta), y, incy);
        };

        /// @brief Block-diagonal matrix-vector multiply with full parameters: y = alpha * op(A) * x + beta * y
        ///
        /// Performs block-wise matrix-vector multiplication where A is a block-diagonal matrix.
        /// Supports transpose operations on each diagonal block.
        ///
        /// @tparam T Numeric element type (double, float).
        /// @param Ta   Transpose option for A: "N" (no transpose, A), "T" (transpose, A^T), "C" (conjugate-transpose, A^H).
        /// @param alpha Scaling factor for op(A)*x.
        /// @param a     Block-diagonal input matrix (BlockDiagMatrix_<T>).
        /// @param x     Input vector (Vector_<T>).
        /// @param incx  Stride for x (in elements).
        /// @param beta  Scaling factor applied to existing contents of y.
        /// @param y     Output vector (Vector_<T>).
        /// @param incy  Stride for y (in elements).
        template <typename T>
        void MatrixVectorProduct(const char *Ta, const T alpha, const BlockDiagMatrix_<T> &a, const Vector_<T> &x,
                                 const size_t incx, const T beta, Vector_<T> &y, const size_t incy);

        /// @brief Block-diagonal matrix-vector multiply: y = alpha * A * x + beta * y
        ///
        /// Performs block-wise matrix-vector multiplication where A is a block-diagonal matrix.
        /// Each block of A multiplies the corresponding portion of x using optimized BLAS
        ///
        /// @tparam T Numeric element type (double, float).
        /// @param alpha Scaling factor for A*x.
        /// @param a     Block-diagonal input matrix (BlockDiagMatrix_<T>).
        /// @param x     Input vector (Vector_<T>).
        /// @param beta  Scaling factor applied to existing contents of y.
        /// @param y     Output vector (Vector_<T>).
        template <typename T>
        void MatrixVectorProduct(const T alpha, const BlockDiagMatrix_<T> &a, const Vector_<T> &x,
                                 const T beta, Vector_<T> &y)
        {
            MatrixVectorProduct("N", alpha, a, x, 1, beta, y, 1);
        };

        /// @brief Convenience wrapper matching standard Matrix signature: y := op(A)*x + beta*y
        ///
        /// This template overload matches the standard Matrix interface with support for
        /// transpose, strides, and scalar factors. Supports the common case y := A*x with defaults.
        ///
        /// @tparam T Numeric element type (double, float).
        /// @param a     Block-diagonal input matrix (BlockDiagMatrix_<T>).
        /// @param x     Input vector (Vector_<T>).
        /// @param y     Output vector (Vector_<T>).
        /// @param Ta    Transpose option for A: "N" (default, no transpose), "T" (transpose, A^T), "C" (conjugate-transpose, A^H). Default: "N".
        /// @param alpha Scaling factor for op(A)*x. Default: 1.0.
        /// @param beta  Scaling factor for existing y. Default: 0.0.
        /// @param incx  Stride for x (in elements). Default: 1.
        /// @param incy  Stride for y (in elements). Default: 1.
        template <typename T>
        inline void MatrixVectorProduct(const BlockDiagMatrix_<T> &a, const Vector_<T> &x, Vector_<T> &y,
                                        const char *Ta = "N", const T alpha = static_cast<T>(1.0), const T beta = static_cast<T>(0.0), const size_t incx = 1, const size_t incy = 1)
        {
            MatrixVectorProduct(Ta, alpha, a, x, incx, beta, y, incy);
        };

        /// @brief Simplified convenience wrapper: y := alpha*A*x + beta*y with defaults.
        ///
        /// This template overload uses default transpose="N", alpha=1 and beta=0 to
        /// support the common case y := A*x, with convenience scalar type conversion.
        ///
        /// @tparam T Numeric element type (double, float).
        /// @param a     Block-diagonal input matrix (BlockDiagMatrix_<T>).
        /// @param x     Input vector (Vector_<T>).
        /// @param y     Output vector (Vector_<T>).
        /// @param alpha Scaling factor for A*x. Default: 1.0.
        /// @param beta  Scaling factor for existing y. Default: 0.0.
        template <typename T, typename Scalar = T, typename Scalar2 = T>
        void MatrixVectorProduct(const BlockDiagMatrix_<T> &a, const Vector_<T> &x, Vector_<T> &y,
                                 const Scalar alpha = 1.0, const Scalar2 beta = 0.0)
        {
            MatrixVectorProduct("N", (T)alpha, a, x, 1, (T)beta, y, 1);
        };

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

        /// @brief Symmetric matrix-vector multiply convenience overload with defaults (double precision), wrapper to BLAS function dsymv.
        ///
        /// Convenience overload using defaults: alpha=1.0, beta=0.0, incx=1, incy=1.
        ///
        /// @param a     Symmetric matrix (Matrix_<double>), n x n.
        /// @param x     Input vector (Vector_<double>), size n.
        /// @param y     Output vector (destination / input-output, Vector_<double>), size n.
        /// @param alpha Scaling factor applied to A*x. Default: 1.0
        /// @param beta  Scaling factor applied to existing contents of y. Default: 0.0
        /// @param incx  Stride for x (in elements). Default: 1
        /// @param incy  Stride for y (in elements). Default: 1
        inline void SymMatrixVectorProduct(const Matrix_<double> &a, const Vector_<double> &x, Vector_<double> &y,
                                           const double alpha = 1.0, const double beta = 0.0, const size_t incx = 1, const size_t incy = 1)
        {
            SymMatrixVectorProduct(alpha, a, x, incx, beta, y, incy);
        };

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

        /// @brief Symmetric matrix-vector multiply convenience overload with defaults (single precision), wrapper to BLAS function ssymv.
        ///
        /// Convenience overload using defaults: alpha=1.0, beta=0.0, incx=1, incy=1.
        ///
        /// @param a     Symmetric matrix (Matrix_<float>), n x n.
        /// @param x     Input vector (Vector_<float>), size n.
        /// @param y     Output vector (destination / input-output, Vector_<float>), size n.
        /// @param alpha Scaling factor applied to A*x. Default: 1.0
        /// @param beta  Scaling factor applied to existing contents of y. Default: 0.0
        /// @param incx  Stride for x (in elements). Default: 1
        /// @param incy  Stride for y (in elements). Default: 1
        inline void SymMatrixVectorProduct(const Matrix_<float> &a, const Vector_<float> &x, Vector_<float> &y,
                                           const float alpha = 1.0, const float beta = 0.0, const size_t incx = 1, const size_t incy = 1)
        {
            SymMatrixVectorProduct(alpha, a, x, incx, beta, y, incy);
        };

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

        /// @brief Symmetric matrix-vector multiply convenience overload with defaults (double precision) for LowTriMatrix_ storage, wrapper to BLAS function dsymv.
        ///
        /// Convenience overload using defaults: alpha=1.0, beta=0.0, incx=1, incy=1.
        ///
        /// @param a     Lower-triangular matrix (LowTriMatrix_<double>), n x n.
        /// @param x     Input vector (Vector_<double>), size n.
        /// @param y     Output vector (destination / input-output, Vector_<double>), size n.
        /// @param alpha Scaling factor applied to A*x. Default: 1.0
        /// @param beta  Scaling factor applied to existing contents of y. Default: 0.0
        /// @param incx  Stride for x (in elements). Default: 1
        /// @param incy  Stride for y (in elements). Default: 1
        inline void SymMatrixVectorProduct(const LowTriMatrix_<double> &a, const Vector_<double> &x, Vector_<double> &y,
                                           const double alpha = 1.0, const double beta = 0.0, const size_t incx = 1, const size_t incy = 1)
        {
            SymMatrixVectorProduct(alpha, a, x, incx, beta, y, incy);
        };

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

        /// @brief Symmetric matrix-vector multiply convenience overload with defaults (single precision) for LowTriMatrix_ storage, wrapper to BLAS function ssymv.
        ///
        /// Convenience overload using defaults: alpha=1.0, beta=0.0, incx=1, incy=1.
        ///
        /// @param a     Lower-triangular matrix (LowTriMatrix_<float>), n x n.
        /// @param x     Input vector (Vector_<float>), size n.
        /// @param y     Output vector (destination / input-output, Vector_<float>), size n.
        /// @param alpha Scaling factor applied to A*x. Default: 1.0
        /// @param beta  Scaling factor applied to existing contents of y. Default: 0.0
        /// @param incx  Stride for x (in elements). Default: 1
        /// @param incy  Stride for y (in elements). Default: 1
        inline void SymMatrixVectorProduct(const LowTriMatrix_<float> &a, const Vector_<float> &x, Vector_<float> &y,
                                           const float alpha = 1.0, const float beta = 0.0, const size_t incx = 1, const size_t incy = 1)
        {
            SymMatrixVectorProduct(alpha, a, x, incx, beta, y, incy);
        };

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

        /// @brief Lower-triangular matrix-vector product convenience overload with defaults (double precision), wrapper to BLAS function dtrmv.
        ///
        /// Convenience overload using defaults: T="N", unit=CblasNonUnit, incx=1.
        ///
        /// @param a     Lower-triangular matrix (LowTriMatrix_<double>), n x n.
        /// @param x     Vector to be multiplied in-place (Vector_<double>), size n.
        /// @param T     Transpose option for the triangular matrix: "N" (no transpose), "T" (transpose), "C" (conjugate-transpose). Default: "N"
        /// @param unit  Whether the diagonal is unit (CblasUnit/CblasNonUnit). Default: CblasNonUnit
        /// @param incx  Stride for x. Default: 1
        inline void LowTriMatrixVectorProduct(const LowTriMatrix_<double> &a, Vector_<double> &x, const char *T = "N", const CBLAS_DIAG unit = CblasNonUnit, const size_t incx = 1)
        {
            LowTriMatrixVectorProduct(T, unit, a, x, incx);
        };
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

        /// @brief Lower-triangular matrix-vector product convenience overload with defaults (single precision), wrapper to BLAS function strmv.
        ///
        /// Convenience overload using defaults: T="N", unit=CblasNonUnit, incx=1.
        ///
        /// @param a     Lower-triangular matrix (LowTriMatrix_<float>), n x n.
        /// @param x     Vector to be multiplied in-place (Vector_<float>), size n.
        /// @param T     Transpose option for the triangular matrix: "N" (no transpose), "T" (transpose), "C" (conjugate-transpose). Default: "N"
        /// @param unit  Whether the diagonal is unit (CblasUnit/CblasNonUnit). Default: CblasNonUnit
        /// @param incx  Stride for x. Default: 1
        inline void LowTriMatrixVectorProduct(const LowTriMatrix_<float> &a, Vector_<float> &x, const char *T = "N", const CBLAS_DIAG unit = CblasNonUnit, const size_t incx = 1)
        {
            LowTriMatrixVectorProduct(T, unit, a, x, incx);
        };

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
