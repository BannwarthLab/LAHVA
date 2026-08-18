#ifndef LAHVA_GPU_LEVEL2_HPP
#define LAHVA_GPU_LEVEL2_HPP
#include "const.h"
#include "linalg.hpp"
#include "runtime.hpp"

namespace lahva{
    namespace gpu{ 

    void OuterVectorProduct(const CudaRuntime& cudart, const Vector<double>& x, const Vector<double>& y, Matrix<double>& A, size_t incx = 1, size_t incy = 1, const double alpha = 1.0);
    void OuterVectorProduct(const CudaRuntime& cudart, const Vector<float>& x, const Vector<float>& y, Matrix<float>& A, size_t incx = 1, size_t incy = 1, const float alpha = 1.0);


    void MatrixVectorProduct(const CudaRuntime& cudart, const char* T, const double alpha, const Matrix_<double>& a, const Vector_<double>& x, 
                             const size_t incx, const double beta, Vector_<double>& y, const size_t incy);
    void MatrixVectorProduct(const CudaRuntime& cudart, const Matrix_<double>& a, const Vector_<double>& x, Vector_<double>& y,
                             const char* T = "N", const double alpha = 1.0, const double beta = 0.0, const size_t incx = 1, const size_t incy = 1);
    void MatrixVectorProduct(const CudaRuntime& cudart, const char* T, const float alpha, const Matrix_<float>& a, const Vector_<float>& x, 
                             const size_t incx, const float beta, Vector_<float>& y, const size_t incy);
    void MatrixVectorProduct(const CudaRuntime& cudart, const Matrix_<float>& a, const Vector_<float>& x, Vector_<float>& y,
                             const char* T = "N", const float alpha = 1.0, const float beta = 0.0, const size_t incx = 1, const size_t incy = 1);

    void MatrixVectorProduct(const CudaRuntime& cudart, const char* T, const complex_double alpha, const Matrix_<complex_double>& a, 
                            const Vector_<complex_double>& x, const size_t incx, const complex_double beta, Vector_<complex_double>& y, const size_t incy);
    void MatrixVectorProduct(const CudaRuntime& cudart, const Matrix_<complex_double>& a, const Vector_<complex_double>& x, Vector_<complex_double>& y,
                            const char* T = "N", const complex_double alpha = 1.0, const complex_double beta = 0.0, const size_t incx = 1, const size_t incy = 1);
    void MatrixVectorProduct(const CudaRuntime& cudart, const char* T, const complex_float alpha, const Matrix_<complex_float>& a, const Vector_<complex_float>& x, 
                            const size_t incx, const complex_float beta, Vector_<complex_float>& y, const size_t incy);
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
    void MatrixVectorProduct(const CudaRuntime& cudart, const char* ta, const double alpha, const BlockDiagMatrix<double>& a,
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
    inline void MatrixVectorProduct(const CudaRuntime& cudart, const BlockDiagMatrix<double>& a, const Vector<double>& x, Vector<double>& y,
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
    void MatrixVectorProduct(const CudaRuntime& cudart, const char* ta, const float alpha, const BlockDiagMatrix<float>& a,
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
    inline void MatrixVectorProduct(const CudaRuntime& cudart, const BlockDiagMatrix<float>& a, const Vector<float>& x, Vector<float>& y,
                            const char* ta = "N", const float alpha = 1.0, const float beta = 0.0)
    {
        MatrixVectorProduct(cudart, ta, alpha, a, x, beta, y);
    }


    void SymMatrixVectorProduct(const CudaRuntime& cudart, const double alpha, const Matrix_<double>& a, const Vector_<double>& x, 
                             const size_t incx, const double beta, Vector_<double>& y, const size_t incy);
    void SymMatrixVectorProduct(const CudaRuntime& cudart, const Matrix_<double>& a, const Vector_<double>& x, Vector_<double>& y,
                             const double alpha = 1.0, const double beta = 0.0, const size_t incx = 1, const size_t incy = 1);
    void SymMatrixVectorProduct(const CudaRuntime& cudart, const float alpha, const Matrix_<float>& a, const Vector_<float>& x, 
                             const size_t incx, const float beta, Vector_<float>& y, const size_t incy);
    void SymMatrixVectorProduct(const CudaRuntime& cudart, const Matrix_<float>& a, const Vector_<float>& x, Vector_<float>& y,
                             const float alpha = 1.0, const float beta = 0.0, const size_t incx = 1, const size_t incy = 1);

        void SymMatrixVectorProduct(const CudaRuntime& cudart, const double alpha, const LowTriMatrix_<double>& a, const Vector_<double>& x, 
                             const size_t incx, const double beta, Vector_<double>& y, const size_t incy);
    void SymMatrixVectorProduct(const CudaRuntime& cudart, const LowTriMatrix_<double>& a, const Vector_<double>& x, Vector_<double>& y,
                             const double alpha = 1.0, const double beta = 0.0, const size_t incx = 1, const size_t incy = 1);
    void SymMatrixVectorProduct(const CudaRuntime& cudart, const float alpha, const LowTriMatrix_<float>& a, const Vector_<float>& x, 
                             const size_t incx, const float beta, Vector_<float>& y, const size_t incy);
    void SymMatrixVectorProduct(const CudaRuntime& cudart, const LowTriMatrix_<float>& a, const Vector_<float>& x, Vector_<float>& y,
                             const float alpha = 1.0, const float beta = 0.0, const size_t incx = 1, const size_t incy = 1);

    //void LowTriMatrixVectorProduct(const CudaRuntime, const char* T, const CBLAS_DIAG unit,const LowTriMatrix_<double>& a, Vector_<double>& x, const size_t incx);
    //void LowTriMatrixVectorProduct(const CudaRuntime, const LowTriMatrix_<double>& a, Vector_<double>& x, const char* T = "N", const CBLAS_DIAG unit = CblasNonUnit, const size_t incx = 1);
    //void LowTriMatrixVectorProduct(const CudaRuntime, const char* T, const CBLAS_DIAG unit, const LowTriMatrix_<float>& a, Vector_<float>& x, const size_t incx);
    //void LowTriMatrixVectorProduct(const CudaRuntime, const LowTriMatrix_<float>& a, Vector_<float>& x, const char* T = "N", const CBLAS_DIAG unit = CblasNonUnit, const size_t incx = 1);
    }
} 

#endif