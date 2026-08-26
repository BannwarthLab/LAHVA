/// @file level2.hpp
/// @brief GPU-accelerated BLAS Level-2 operations (matrix-vector operations).
///
/// Provides GPU BLAS Level-2 kernel declarations for CUDA acceleration.
/// Level-2 operations include matrix-vector multiplication, triangular solves, and rank-1/2 updates.
/// Each function is provided for double and float precision with GPU runtime management.

#pragma once
#include "const.h"
#include "linalg.hpp"
#include "runtime.hpp"

namespace lahva
{
    namespace gpu
    {

    /// @brief Computes outer product of two double-precision GPU vectors, wrapper to cuBLAS function cublasDger.
    ///
    /// Performs A = alpha * x * y^T + A where x and y are vectors and A is a matrix.
    ///
    /// @param cudart CUDA runtime instance
    /// @param x First input GPU vector (Vector<double>).
    /// @param y Second input GPU vector (Vector<double>).
    /// @param A Output GPU matrix to accumulate result (Matrix<double>).
    /// @param incx Stride between consecutive elements in vector x (default: 1).
    /// @param incy Stride between consecutive elements in vector y (default: 1).
    /// @param alpha Scalar multiplier for the outer product (default: 1.0).
    void OuterVectorProduct(const CudaRuntime& cudart, const Vector<double>& x, const Vector<double>& y, Matrix<double>& A, size_t incx = 1, size_t incy = 1, const double alpha = 1.0);

    /// @brief Computes outer product of two single-precision GPU vectors, wrapper to cuBLAS function cublasSger.
    ///
    /// Performs A = alpha * x * y^T + A where x and y are vectors and A is a matrix.
    ///
    /// @param cudart CUDA runtime instance
    /// @param x First input GPU vector (Vector<float>).
    /// @param y Second input GPU vector (Vector<float>).
    /// @param A Output GPU matrix to accumulate result (Matrix<float>).
    /// @param incx Stride between consecutive elements in vector x (default: 1).
    /// @param incy Stride between consecutive elements in vector y (default: 1).
    /// @param alpha Scalar multiplier for the outer product (default: 1.0).
    void OuterVectorProduct(const CudaRuntime& cudart, const Vector<float>& x, const Vector<float>& y, Matrix<float>& A, size_t incx = 1, size_t incy = 1, const float alpha = 1.0);

    /// @brief Computes matrix-vector product with double-precision GPU tensors, wrapper to cuBLAS function cublasDgemv.
    ///
    /// Performs y = alpha * op(A) * x + beta * y where op(A) is specified by T ("N" for no transpose, "T" for transpose).
    ///
    /// @param cudart CUDA runtime instance
    /// @param T Operation on matrix A ("N" for no transpose, "T" for transpose).
    /// @param alpha Scalar multiplier for the matrix-vector product.
    /// @param a Input GPU matrix (Matrix_<double>).
    /// @param x Input GPU vector (Vector_<double>).
    /// @param incx Stride between consecutive elements in vector x.
    /// @param beta Scalar multiplier for vector y.
    /// @param y Output GPU vector (destination / input-output, Vector_<double>).
    /// @param incy Stride between consecutive elements in vector y.
    void MatrixVectorProduct(const CudaRuntime& cudart, const char* T, const double alpha, const Matrix_<double>& a, const Vector_<double>& x,
                             const size_t incx, const double beta, Vector_<double>& y, const size_t incy);

    /// @brief Computes matrix-vector product with double-precision GPU tensors (convenience overload).
    ///
    /// Performs y = alpha * op(A) * x + beta * y with default parameters.
    ///
    /// @param cudart CUDA runtime instance
    /// @param a Input GPU matrix (Matrix_<double>).
    /// @param x Input GPU vector (Vector_<double>).
    /// @param y Output GPU vector (destination / input-output, Vector_<double>).
    /// @param T Operation on matrix A (default: "N" for no transpose).
    /// @param alpha Scalar multiplier (default: 1.0).
    /// @param beta Scalar multiplier for y (default: 0.0).
    /// @param incx Stride in vector x (default: 1).
    /// @param incy Stride in vector y (default: 1).
    void MatrixVectorProduct(const CudaRuntime& cudart, const Matrix_<double>& a, const Vector_<double>& x, Vector_<double>& y,
                             const char* T = "N", const double alpha = 1.0, const double beta = 0.0, const size_t incx = 1, const size_t incy = 1);

    /// @brief Computes matrix-vector product with single-precision GPU tensors, wrapper to cuBLAS function cublasSgemv.
    ///
    /// Performs y = alpha * op(A) * x + beta * y where op(A) is specified by T.
    ///
    /// @param cudart CUDA runtime instance
    /// @param T Operation on matrix A ("N" for no transpose, "T" for transpose).
    /// @param alpha Scalar multiplier for the matrix-vector product.
    /// @param a Input GPU matrix (Matrix_<float>).
    /// @param x Input GPU vector (Vector_<float>).
    /// @param incx Stride between consecutive elements in vector x.
    /// @param beta Scalar multiplier for vector y.
    /// @param y Output GPU vector (destination / input-output, Vector_<float>).
    /// @param incy Stride between consecutive elements in vector y.
    void MatrixVectorProduct(const CudaRuntime& cudart, const char* T, const float alpha, const Matrix_<float>& a, const Vector_<float>& x,
                             const size_t incx, const float beta, Vector_<float>& y, const size_t incy);

    /// @brief Computes matrix-vector product with single-precision GPU tensors (convenience overload).
    ///
    /// Performs y = alpha * op(A) * x + beta * y with default parameters.
    ///
    /// @param cudart CUDA runtime instance
    /// @param a Input GPU matrix (Matrix_<float>).
    /// @param x Input GPU vector (Vector_<float>).
    /// @param y Output GPU vector (destination / input-output, Vector_<float>).
    /// @param T Operation on matrix A (default: "N" for no transpose).
    /// @param alpha Scalar multiplier (default: 1.0).
    /// @param beta Scalar multiplier for y (default: 0.0).
    /// @param incx Stride in vector x (default: 1).
    /// @param incy Stride in vector y (default: 1).
    void MatrixVectorProduct(const CudaRuntime& cudart, const Matrix_<float>& a, const Vector_<float>& x, Vector_<float>& y,
                             const char* T = "N", const float alpha = 1.0, const float beta = 0.0, const size_t incx = 1, const size_t incy = 1);

    /// @brief Computes matrix-vector product with complex double-precision GPU tensors, wrapper to cuBLAS function cublasZgemv.
    ///
    /// Performs y = alpha * op(A) * x + beta * y where op(A) is specified by T.
    ///
    /// @param cudart CUDA runtime instance
    /// @param T Operation on matrix A ("N" for no transpose, "T" for transpose).
    /// @param alpha Scalar multiplier for the matrix-vector product (complex_double).
    /// @param a Input GPU matrix (Matrix_<complex_double>).
    /// @param x Input GPU vector (Vector_<complex_double>).
    /// @param incx Stride between consecutive elements in vector x.
    /// @param beta Scalar multiplier for vector y (complex_double).
    /// @param y Output GPU vector (destination / input-output, Vector_<complex_double>).
    /// @param incy Stride between consecutive elements in vector y.
    void MatrixVectorProduct(const CudaRuntime& cudart, const char* T, const complex_double alpha, const Matrix_<complex_double>& a,
                            const Vector_<complex_double>& x, const size_t incx, const complex_double beta, Vector_<complex_double>& y, const size_t incy);

    /// @brief Computes matrix-vector product with complex double-precision GPU tensors (convenience overload).
    ///
    /// Performs y = alpha * op(A) * x + beta * y with default parameters.
    ///
    /// @param cudart CUDA runtime instance
    /// @param a Input GPU matrix (Matrix_<complex_double>).
    /// @param x Input GPU vector (Vector_<complex_double>).
    /// @param y Output GPU vector (destination / input-output, Vector_<complex_double>).
    /// @param T Operation on matrix A (default: "N" for no transpose).
    /// @param alpha Scalar multiplier (default: 1.0).
    /// @param beta Scalar multiplier for y (default: 0.0).
    /// @param incx Stride in vector x (default: 1).
    /// @param incy Stride in vector y (default: 1).
    void MatrixVectorProduct(const CudaRuntime& cudart, const Matrix_<complex_double>& a, const Vector_<complex_double>& x, Vector_<complex_double>& y,
                            const char* T = "N", const complex_double alpha = 1.0, const complex_double beta = 0.0, const size_t incx = 1, const size_t incy = 1);

    /// @brief Computes matrix-vector product with complex single-precision GPU tensors, wrapper to cuBLAS function cublasCgemv.
    ///
    /// Performs y = alpha * op(A) * x + beta * y where op(A) is specified by T.
    ///
    /// @param cudart CUDA runtime instance
    /// @param T Operation on matrix A ("N" for no transpose, "T" for transpose).
    /// @param alpha Scalar multiplier for the matrix-vector product (complex_float).
    /// @param a Input GPU matrix (Matrix_<complex_float>).
    /// @param x Input GPU vector (Vector_<complex_float>).
    /// @param incx Stride between consecutive elements in vector x.
    /// @param beta Scalar multiplier for vector y (complex_float).
    /// @param y Output GPU vector (destination / input-output, Vector_<complex_float>).
    /// @param incy Stride between consecutive elements in vector y.
    void MatrixVectorProduct(const CudaRuntime& cudart, const char* T, const complex_float alpha, const Matrix_<complex_float>& a, const Vector_<complex_float>& x,
                            const size_t incx, const complex_float beta, Vector_<complex_float>& y, const size_t incy);

    /// @brief Computes matrix-vector product with complex single-precision GPU tensors (convenience overload).
    ///
    /// Performs y = alpha * op(A) * x + beta * y with default parameters.
    ///
    /// @param cudart CUDA runtime instance
    /// @param a Input GPU matrix (Matrix_<complex_float>).
    /// @param x Input GPU vector (Vector_<complex_float>).
    /// @param y Output GPU vector (destination / input-output, Vector_<complex_float>).
    /// @param T Operation on matrix A (default: "N" for no transpose).
    /// @param alpha Scalar multiplier (default: 1.0).
    /// @param beta Scalar multiplier for y (default: 0.0).
    /// @param incx Stride in vector x (default: 1).
    /// @param incy Stride in vector y (default: 1).
    void MatrixVectorProduct(const CudaRuntime& cudart, const Matrix_<complex_float>& a, const Vector_<complex_float>& x, Vector_<complex_float>& y,
                            const char* T = "N", const complex_float alpha = 1.0, const complex_float beta = 0.0, const size_t incx = 1, const size_t incy = 1);

    /// @brief Computes block-diagonal matrix-vector product with double-precision GPU tensors using batched GEMV.
    ///
    /// Performs y = alpha * op(A) * x + beta * y where A is a block-diagonal matrix
    /// and op(A) is specified by ta ("N" for no transpose, "T" for transpose).
    /// Uses cuBLAS batched operations for efficient GPU computation.
    ///
    /// @param cudart CUDA runtime instance
    /// @param ta Transposition character: 'N' (no transpose), 'T' (transpose)
    /// @param alpha Scalar multiplier for the matrix-vector product (double)
    /// @param a Input block-diagonal GPU matrix (BlockDiagMatrix<double>)
    /// @param x Input GPU vector (Vector<double>)
    /// @param beta Scalar multiplier for vector y (double)
    /// @param y Input/output GPU vector (Vector<double>), replaced with result
    void MatrixVectorProduct(CudaRuntime& cudart, const char* ta, const double alpha, const BlockDiagMatrix<double>& a,
                            const Vector<double>& x, const double beta, Vector<double>& y);

    /// @brief Computes block-diagonal matrix-vector product with double-precision GPU tensors (convenience overload).
    ///
    /// Performs y = alpha * op(A) * x + beta * y with default parameters.
    ///
    /// @param cudart CUDA runtime instance
    /// @param a Input block-diagonal GPU matrix (BlockDiagMatrix<double>)
    /// @param x Input GPU vector (Vector<double>)
    /// @param y Input/output GPU vector (Vector<double>)
    /// @param ta Transposition character (default: "N" for no transpose)
    /// @param alpha Scalar multiplier (default: 1.0)
    /// @param beta Scalar multiplier for y (default: 0.0)
    inline void MatrixVectorProduct(CudaRuntime& cudart, const BlockDiagMatrix<double>& a, const Vector<double>& x, Vector<double>& y,
                            const char* ta = "N", const double alpha = 1.0, const double beta = 0.0)
    {
        MatrixVectorProduct(cudart, ta, alpha, a, x, beta, y);
    }

    /// @brief Computes block-diagonal matrix-vector product with single-precision GPU tensors using batched GEMV.
    ///
    /// Performs y = alpha * op(A) * x + beta * y where A is a block-diagonal matrix
    /// and op(A) is specified by ta ("N" for no transpose, "T" for transpose).
    /// Uses cuBLAS batched operations for efficient GPU computation.
    ///
    /// @param cudart CUDA runtime instance
    /// @param ta Transposition character: 'N' (no transpose), 'T' (transpose)
    /// @param alpha Scalar multiplier for the matrix-vector product (float)
    /// @param a Input block-diagonal GPU matrix (BlockDiagMatrix<float>)
    /// @param x Input GPU vector (Vector<float>)
    /// @param beta Scalar multiplier for vector y (float)
    /// @param y Input/output GPU vector (Vector<float>), replaced with result
    void MatrixVectorProduct(CudaRuntime& cudart, const char* ta, const float alpha, const BlockDiagMatrix<float>& a,
                            const Vector<float>& x, const float beta, Vector<float>& y);

    /// @brief Computes block-diagonal matrix-vector product with single-precision GPU tensors (convenience overload).
    ///
    /// Performs y = alpha * op(A) * x + beta * y with default parameters.
    ///
    /// @param cudart CUDA runtime instance
    /// @param a Input block-diagonal GPU matrix (BlockDiagMatrix<float>)
    /// @param x Input GPU vector (Vector<float>)
    /// @param y Input/output GPU vector (Vector<float>)
    /// @param ta Transposition character (default: "N" for no transpose)
    /// @param alpha Scalar multiplier (default: 1.0)
    /// @param beta Scalar multiplier for y (default: 0.0)
    inline void MatrixVectorProduct(CudaRuntime& cudart, const BlockDiagMatrix<float>& a, const Vector<float>& x, Vector<float>& y,
                            const char* ta = "N", const float alpha = 1.0, const float beta = 0.0)
    {
        MatrixVectorProduct(cudart, ta, alpha, a, x, beta, y);
    }

    /// @brief Implementation of sparse matrix-vector product with BlockMatrix using cuSPARSE.
    /// Should be called via the public MatrixVectorProduct wrapper functions for double and float precision.
    template <typename T>
    void MatrixVectorProduct_sparse(CudaRuntime& cudart, const char* ta, const T alpha, const BlockMatrix_<T>& a,
                            const Vector_<T>& x, const T beta, Vector_<T>& y);


    /// @brief Sparse matrix-vector product with double-precision BlockMatrix via cuSPARSE.
    ///
    /// Performs sparse matrix-vector multiplication \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
    /// with double-precision (double) elements using cuSPARSE operations.
    /// The BlockMatrix is converted to BSR format for uniform blocks or CSR format for non-uniform blocks.
    ///
    /// @param cudart CUDA runtime instance.
    /// @param ta Transposition character: 'N' (no transpose), 'T' (transpose).
    /// @param alpha Scalar factor for the matrix-vector product.
    /// @param a Input double-precision block-structured sparse matrix.
    /// @param x Input double-precision vector.
    /// @param beta Scalar factor for vector y.
    /// @param y Input/output double-precision vector, replaced with result.
    inline void MatrixVectorProduct(CudaRuntime& cudart, const char* ta, const double alpha, const BlockMatrix_<double>& a,
                            const Vector_<double>& x, const double beta, Vector_<double>& y)
    {
        MatrixVectorProduct_sparse<double>(cudart, ta, alpha, a, x, beta, y);
    }

    /// @brief Computes sparse matrix-vector product with BlockMatrix (convenience overload, double precision).
    ///
    /// Performs y = alpha * A * x + beta * y with default parameters.
    ///
    /// @param cudart CUDA runtime instance
    /// @param a Input block-sparse GPU matrix (BlockMatrix_<double>).
    /// @param x Input GPU vector (Vector_<double>).
    /// @param y Output GPU vector (destination / input-output, Vector_<double>).
    /// @param ta Operation on matrix A (default: "N").
    /// @param alpha Scalar multiplier (default: 1.0).
    /// @param beta Scalar multiplier for y (default: 0.0).
    inline void MatrixVectorProduct(CudaRuntime& cudart, const BlockMatrix_<double>& a, const Vector_<double>& x, Vector_<double>& y,
                            const char* ta = "N", const double alpha = 1.0, const double beta = 0.0)
    {
        MatrixVectorProduct(cudart, ta, alpha, a, x, beta, y);
    }

    /// @brief Sparse matrix-vector product with single-precision BlockMatrix via cuSPARSE.
    ///
    /// Performs sparse matrix-vector multiplication \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
    /// with single-precision (float) elements using cuSPARSE operations.
    /// The BlockMatrix is converted to BSR format for uniform blocks or CSR format for non-uniform blocks.
    ///
    /// @param cudart CUDA runtime instance.
    /// @param ta Transposition character: 'N' (no transpose), 'T' (transpose).
    /// @param alpha Scalar factor for the matrix-vector product.
    /// @param a Input single-precision block-structured sparse matrix.
    /// @param x Input single-precision vector.
    /// @param beta Scalar factor for vector y.
    /// @param y Input/output single-precision vector, replaced with result.
    inline void MatrixVectorProduct(CudaRuntime& cudart, const char* ta, const float alpha, const BlockMatrix_<float>& a,
                            const Vector_<float>& x, const float beta, Vector_<float>& y)
    {
        MatrixVectorProduct_sparse<float>(cudart, ta, alpha, a, x, beta, y);
    }

    /// @brief Computes sparse matrix-vector product with BlockMatrix (convenience overload, single precision).
    ///
    /// Performs y = alpha * A * x + beta * y with default parameters.
    ///
    /// @param cudart CUDA runtime instance
    /// @param a Input block-sparse GPU matrix (BlockMatrix_<float>).
    /// @param x Input GPU vector (Vector_<float>).
    /// @param y Output GPU vector (destination / input-output, Vector_<float>).
    /// @param ta Operation on matrix A (default: "N").
    /// @param alpha Scalar multiplier (default: 1.0).
    /// @param beta Scalar multiplier for y (default: 0.0).
    inline void MatrixVectorProduct(CudaRuntime& cudart, const BlockMatrix_<float>& a, const Vector_<float>& x, Vector_<float>& y,
                            const char* ta = "N", const float alpha = 1.0, const float beta = 0.0)
    {
        MatrixVectorProduct(cudart, ta, alpha, a, x, beta, y);
    }

    /// @brief Computes symmetric matrix-vector product with double-precision GPU tensors, wrapper to cuBLAS function cublasDsymv.
    ///
    /// Performs y = alpha * A * x + beta * y where A is a symmetric matrix.
    ///
    /// @param cudart CUDA runtime instance
    /// @param alpha Scalar multiplier for the matrix-vector product.
    /// @param a Input symmetric GPU matrix (Matrix_<double>).
    /// @param x Input GPU vector (Vector_<double>).
    /// @param incx Stride between consecutive elements in vector x.
    /// @param beta Scalar multiplier for vector y.
    /// @param y Output GPU vector (destination / input-output, Vector_<double>).
    /// @param incy Stride between consecutive elements in vector y.
    void SymMatrixVectorProduct(const CudaRuntime& cudart, const double alpha, const Matrix_<double>& a, const Vector_<double>& x,
                             const size_t incx, const double beta, Vector_<double>& y, const size_t incy);

    /// @brief Computes symmetric matrix-vector product with double-precision GPU tensors (convenience overload).
    ///
    /// Performs y = alpha * A * x + beta * y with default parameters.
    ///
    /// @param cudart CUDA runtime instance
    /// @param a Input symmetric GPU matrix (Matrix_<double>).
    /// @param x Input GPU vector (Vector_<double>).
    /// @param y Output GPU vector (destination / input-output, Vector_<double>).
    /// @param alpha Scalar multiplier (default: 1.0).
    /// @param beta Scalar multiplier for y (default: 0.0).
    /// @param incx Stride in vector x (default: 1).
    /// @param incy Stride in vector y (default: 1).
    void SymMatrixVectorProduct(const CudaRuntime& cudart, const Matrix_<double>& a, const Vector_<double>& x, Vector_<double>& y,
                             const double alpha = 1.0, const double beta = 0.0, const size_t incx = 1, const size_t incy = 1);

    /// @brief Computes symmetric matrix-vector product with single-precision GPU tensors, wrapper to cuBLAS function cublasSsymv.
    ///
    /// Performs y = alpha * A * x + beta * y where A is a symmetric matrix.
    ///
    /// @param cudart CUDA runtime instance
    /// @param alpha Scalar multiplier for the matrix-vector product.
    /// @param a Input symmetric GPU matrix (Matrix_<float>).
    /// @param x Input GPU vector (Vector_<float>).
    /// @param incx Stride between consecutive elements in vector x.
    /// @param beta Scalar multiplier for vector y.
    /// @param y Output GPU vector (destination / input-output, Vector_<float>).
    /// @param incy Stride between consecutive elements in vector y.
    void SymMatrixVectorProduct(const CudaRuntime& cudart, const float alpha, const Matrix_<float>& a, const Vector_<float>& x,
                             const size_t incx, const float beta, Vector_<float>& y, const size_t incy);

    /// @brief Computes symmetric matrix-vector product with single-precision GPU tensors (convenience overload).
    ///
    /// Performs y = alpha * A * x + beta * y with default parameters.
    ///
    /// @param cudart CUDA runtime instance
    /// @param a Input symmetric GPU matrix (Matrix_<float>).
    /// @param x Input GPU vector (Vector_<float>).
    /// @param y Output GPU vector (destination / input-output, Vector_<float>).
    /// @param alpha Scalar multiplier (default: 1.0).
    /// @param beta Scalar multiplier for y (default: 0.0).
    /// @param incx Stride in vector x (default: 1).
    /// @param incy Stride in vector y (default: 1).
    void SymMatrixVectorProduct(const CudaRuntime& cudart, const Matrix_<float>& a, const Vector_<float>& x, Vector_<float>& y,
                             const float alpha = 1.0, const float beta = 0.0, const size_t incx = 1, const size_t incy = 1);

    /// @brief Computes symmetric matrix-vector product with double-precision lower triangular GPU matrix.
    ///
    /// Performs y = alpha * A * x + beta * y where A is lower triangular.
    ///
    /// @param cudart CUDA runtime instance
    /// @param alpha Scalar multiplier for the matrix-vector product.
    /// @param a Input lower triangular GPU matrix (LowTriMatrix_<double>).
    /// @param x Input GPU vector (Vector_<double>).
    /// @param incx Stride between consecutive elements in vector x.
    /// @param beta Scalar multiplier for vector y.
    /// @param y Output GPU vector (destination / input-output, Vector_<double>).
    /// @param incy Stride between consecutive elements in vector y.
    void SymMatrixVectorProduct(const CudaRuntime& cudart, const double alpha, const LowTriMatrix_<double>& a, const Vector_<double>& x,
                             const size_t incx, const double beta, Vector_<double>& y, const size_t incy);

    /// @brief Computes symmetric matrix-vector product with double-precision lower triangular GPU matrix (convenience overload).
    ///
    /// Performs y = alpha * A * x + beta * y with default parameters.
    ///
    /// @param cudart CUDA runtime instance
    /// @param a Input lower triangular GPU matrix (LowTriMatrix_<double>).
    /// @param x Input GPU vector (Vector_<double>).
    /// @param y Output GPU vector (destination / input-output, Vector_<double>).
    /// @param alpha Scalar multiplier (default: 1.0).
    /// @param beta Scalar multiplier for y (default: 0.0).
    /// @param incx Stride in vector x (default: 1).
    /// @param incy Stride in vector y (default: 1).
    void SymMatrixVectorProduct(const CudaRuntime& cudart, const LowTriMatrix_<double>& a, const Vector_<double>& x, Vector_<double>& y,
                             const double alpha = 1.0, const double beta = 0.0, const size_t incx = 1, const size_t incy = 1);

    /// @brief Computes symmetric matrix-vector product with single-precision lower triangular GPU matrix.
    ///
    /// Performs y = alpha * A * x + beta * y where A is lower triangular.
    ///
    /// @param cudart CUDA runtime instance
    /// @param alpha Scalar multiplier for the matrix-vector product.
    /// @param a Input lower triangular GPU matrix (LowTriMatrix_<float>).
    /// @param x Input GPU vector (Vector_<float>).
    /// @param incx Stride between consecutive elements in vector x.
    /// @param beta Scalar multiplier for vector y.
    /// @param y Output GPU vector (destination / input-output, Vector_<float>).
    /// @param incy Stride between consecutive elements in vector y.
    void SymMatrixVectorProduct(const CudaRuntime& cudart, const float alpha, const LowTriMatrix_<float>& a, const Vector_<float>& x,
                             const size_t incx, const float beta, Vector_<float>& y, const size_t incy);

    /// @brief Computes symmetric matrix-vector product with single-precision lower triangular GPU matrix (convenience overload).
    ///
    /// Performs y = alpha * A * x + beta * y with default parameters.
    ///
    /// @param cudart CUDA runtime instance
    /// @param a Input lower triangular GPU matrix (LowTriMatrix_<float>).
    /// @param x Input GPU vector (Vector_<float>).
    /// @param y Output GPU vector (destination / input-output, Vector_<float>).
    /// @param alpha Scalar multiplier (default: 1.0).
    /// @param beta Scalar multiplier for y (default: 0.0).
    /// @param incx Stride in vector x (default: 1).
    /// @param incy Stride in vector y (default: 1).
    void SymMatrixVectorProduct(const CudaRuntime& cudart, const LowTriMatrix_<float>& a, const Vector_<float>& x, Vector_<float>& y,
                             const float alpha = 1.0, const float beta = 0.0, const size_t incx = 1, const size_t incy = 1);

    } // namespace gpu
} // namespace lahva
