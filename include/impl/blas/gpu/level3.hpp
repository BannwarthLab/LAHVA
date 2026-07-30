/// @file level3.hpp
/// @brief GPU-accelerated BLAS Level-3 operations (matrix-matrix operations).
///
/// Provides GPU BLAS Level-3 kernel declarations for CUDA acceleration.
/// Level-3 operations include matrix-matrix multiplication, triangular solves, and rank-k updates.
/// Each function is provided for double and float precision with GPU runtime management.

#pragma once
#include "const.h"
#include "linalg.hpp"
#include "runtime.hpp"

namespace lahva
{
    namespace gpu 
    {

    /// @brief Computes matrix-matrix product with double-precision GPU matrices, wrapper to cuBLAS function cublasDgemm.
    ///
    /// Performs C = alpha * op(A) * op(B) + beta * C where op is specified by Ta and Tb.
    ///
    /// @param cudart CUDA runtime instance
    /// @param Ta Operation on matrix A ("N" for no transpose, "T" for transpose).
    /// @param Tb Operation on matrix B ("N" for no transpose, "T" for transpose).
    /// @param alpha Scalar multiplier for the matrix product.
    /// @param a Input GPU matrix A (Matrix_<double>).
    /// @param b Input GPU matrix B (Matrix_<double>).
    /// @param beta Scalar multiplier for matrix C.
    /// @param c Output GPU matrix C (destination / input-output, Matrix_<double>).
    void MatrixMatrixProduct(const CudaRuntime& cudart, const char* Ta, const char* Tb, const double alpha, const Matrix_<double>& a, const Matrix_<double>& b,
                             const double beta, Matrix_<double>& c);

    /// @brief Computes matrix-matrix product with double-precision GPU matrices (convenience overload).
    ///
    /// Performs C = alpha * op(A) * op(B) + beta * C with default parameters.
    ///
    /// @param cudart CUDA runtime instance
    /// @param a Input GPU matrix A (Matrix_<double>).
    /// @param b Input GPU matrix B (Matrix_<double>).
    /// @param c Output GPU matrix C (destination / input-output, Matrix_<double>).
    /// @param alpha Scalar multiplier (default: 1.0).
    /// @param beta Scalar multiplier for C (default: 0.0).
    /// @param Ta Operation on A (default: "N" for no transpose).
    /// @param Tb Operation on B (default: "N" for no transpose).
    void MatrixMatrixProduct(const CudaRuntime& cudart, const Matrix_<double>& a, const Matrix_<double>& b, Matrix_<double>& c,
                             const double alpha = 1.0 , const double beta = 0.0, const char* Ta = "N", const char* Tb = "N");

    /// @brief Computes matrix-matrix product with single-precision GPU matrices, wrapper to cuBLAS function cublasSgemm.
    ///
    /// Performs C = alpha * op(A) * op(B) + beta * C where op is specified by Ta and Tb.
    ///
    /// @param cudart CUDA runtime instance
    /// @param Ta Operation on matrix A ("N" for no transpose, "T" for transpose).
    /// @param Tb Operation on matrix B ("N" for no transpose, "T" for transpose).
    /// @param alpha Scalar multiplier for the matrix product.
    /// @param a Input GPU matrix A (Matrix_<float>).
    /// @param b Input GPU matrix B (Matrix_<float>).
    /// @param beta Scalar multiplier for matrix C.
    /// @param c Output GPU matrix C (destination / input-output, Matrix_<float>).
    void MatrixMatrixProduct(const CudaRuntime& cudart, const char* Ta, const char* Tb, const float alpha, const Matrix_<float>& a, const Matrix_<float>& b,
                             const float beta, Matrix_<float>& c);

    /// @brief Computes matrix-matrix product with single-precision GPU matrices (convenience overload).
    ///
    /// Performs C = alpha * op(A) * op(B) + beta * C with default parameters.
    ///
    /// @param cudart CUDA runtime instance
    /// @param a Input GPU matrix A (Matrix_<float>).
    /// @param b Input GPU matrix B (Matrix_<float>).
    /// @param c Output GPU matrix C (destination / input-output, Matrix_<float>).
    /// @param alpha Scalar multiplier (default: 1.0).
    /// @param beta Scalar multiplier for C (default: 0.0).
    /// @param Ta Operation on A (default: "N" for no transpose).
    /// @param Tb Operation on B (default: "N" for no transpose).
    void MatrixMatrixProduct(const CudaRuntime& cudart, const Matrix_<float>& a, const Matrix_<float>& b, Matrix_<float>& c,
                             const float alpha = 1.0 , const float beta = 0.0, const char* Ta = "N", const char* Tb = "N");

    /// @brief Computes matrix-matrix product with complex double-precision GPU matrices, wrapper to cuBLAS function cublasZgemm.
    ///
    /// Performs C = alpha * op(A) * op(B) + beta * C where op is specified by Ta and Tb.
    ///
    /// @param cudart CUDA runtime instance
    /// @param Ta Operation on matrix A ("N" for no transpose, "T" for transpose).
    /// @param Tb Operation on matrix B ("N" for no transpose, "T" for transpose).
    /// @param alpha Scalar multiplier for the matrix product (complex_double).
    /// @param a Input GPU matrix A (Matrix_<complex_double>).
    /// @param b Input GPU matrix B (Matrix_<complex_double>).
    /// @param beta Scalar multiplier for matrix C (complex_double).
    /// @param c Output GPU matrix C (destination / input-output, Matrix_<complex_double>).
    void MatrixMatrixProduct(const CudaRuntime& cudart, const char* Ta, const char* Tb, const complex_double alpha, const Matrix_<complex_double>& a,
                             const Matrix_<complex_double>& b, const complex_double beta, Matrix_<complex_double>& c);

    /// @brief Computes matrix-matrix product with complex double-precision GPU matrices (convenience overload).
    ///
    /// Performs C = alpha * op(A) * op(B) + beta * C with default parameters.
    ///
    /// @param cudart CUDA runtime instance
    /// @param a Input GPU matrix A (Matrix_<complex_double>).
    /// @param b Input GPU matrix B (Matrix_<complex_double>).
    /// @param c Output GPU matrix C (destination / input-output, Matrix_<complex_double>).
    /// @param alpha Scalar multiplier (default: 1.0).
    /// @param beta Scalar multiplier for C (default: 0.0).
    /// @param Ta Operation on A (default: "N" for no transpose).
    /// @param Tb Operation on B (default: "N" for no transpose).
    void MatrixMatrixProduct(const CudaRuntime& cudart, const Matrix_<complex_double>& a, const Matrix_<complex_double>& b, Matrix_<complex_double>& c,
                             const complex_double alpha = 1.0 , const complex_double beta = 0.0, const char* Ta = "N", const char* Tb = "N");

    /// @brief Computes matrix-matrix product with complex single-precision GPU matrices, wrapper to cuBLAS function cublasCgemm.
    ///
    /// Performs C = alpha * op(A) * op(B) + beta * C where op is specified by Ta and Tb.
    ///
    /// @param cudart CUDA runtime instance
    /// @param Ta Operation on matrix A ("N" for no transpose, "T" for transpose).
    /// @param Tb Operation on matrix B ("N" for no transpose, "T" for transpose).
    /// @param alpha Scalar multiplier for the matrix product (complex_float).
    /// @param a Input GPU matrix A (Matrix_<complex_float>).
    /// @param b Input GPU matrix B (Matrix_<complex_float>).
    /// @param beta Scalar multiplier for matrix C (complex_float).
    /// @param c Output GPU matrix C (destination / input-output, Matrix_<complex_float>).
    void MatrixMatrixProduct(const CudaRuntime& cudart, const char* Ta, const char* Tb, const complex_float alpha, const Matrix_<complex_float>& a,
                             const Matrix_<complex_float>& b, const complex_float beta, Matrix_<complex_float>& c);

    /// @brief Computes matrix-matrix product with complex single-precision GPU matrices (convenience overload).
    ///
    /// Performs C = alpha * op(A) * op(B) + beta * C with default parameters.
    ///
    /// @param cudart CUDA runtime instance
    /// @param a Input GPU matrix A (Matrix_<complex_float>).
    /// @param b Input GPU matrix B (Matrix_<complex_float>).
    /// @param c Output GPU matrix C (destination / input-output, Matrix_<complex_float>).
    /// @param alpha Scalar multiplier (default: 1.0).
    /// @param beta Scalar multiplier for C (default: 0.0).
    /// @param Ta Operation on A (default: "N" for no transpose).
    /// @param Tb Operation on B (default: "N" for no transpose).
    void MatrixMatrixProduct(const CudaRuntime& cudart, const Matrix_<complex_float>& a, const Matrix_<complex_float>& b, Matrix_<complex_float>& c,
                             const complex_float alpha = 1.0 , const complex_float beta = 0.0, const char* Ta = "N", const char* Tb = "N");

    /// @brief Computes symmetric matrix-matrix product with double-precision GPU matrices, wrapper to cuBLAS function cublasDsymm.
    ///
    /// Performs C = alpha * A * B + beta * C where A is symmetric (if side is left) 
    /// or C = alpha * B * A + beta * C (if side is right)
    ///
    /// @param cudart CUDA runtime instance
    /// @param side Side of matrix A (CUBLAS_SIDE_LEFT for A on left, CUBLAS_SIDE_RIGHT for A on right).
    /// @param alpha Scalar multiplier for the matrix product.
    /// @param a Input symmetric GPU matrix A (Matrix_<double>).
    /// @param b Input GPU matrix B (Matrix_<double>).
    /// @param beta Scalar multiplier for matrix C.
    /// @param c Output GPU matrix C (destination / input-output, Matrix_<double>).
    void SymMatrixMatrixProduct(const CudaRuntime& cudart, const cublasSideMode_t side, const double alpha, const Matrix_<double>& a, const Matrix_<double>& b,
                                const double beta, Matrix_<double>& c);

    /// @brief Computes symmetric matrix-matrix product with double-precision GPU matrices (convenience overload).
    ///
    /// Performs C = A * B where A is symmetric (if side is left) 
    /// or C = B * A (if side is right)
    ///
    /// @param cudart CUDA runtime instance
    /// @param a Input symmetric GPU matrix A (Matrix_<double>).
    /// @param b Input GPU matrix B (Matrix_<double>).
    /// @param c Output GPU matrix C (destination / input-output, Matrix_<double>).
    /// @param alpha Scalar multiplier (default: 1.0).
    /// @param beta Scalar multiplier for C (default: 0.0).
    /// @param side Side of matrix A (default: CUBLAS_SIDE_LEFT).
    void SymMatrixMatrixProduct(const CudaRuntime& cudart, const Matrix_<double>& a, const Matrix_<double>& b, Matrix_<double>& c,
                                const double alpha = 1.0 , const double beta = 0.0, const cublasSideMode_t side = CUBLAS_SIDE_LEFT);

    /// @brief Computes symmetric matrix-matrix product with single-precision GPU matrices, wrapper to cuBLAS function cublasSsymm.
    ///
    /// Performs C = alpha * A * B + beta * C where A is symmetric (if side is left) 
    /// or C = alpha * B * A + beta * C (if side is right)
    ///
    /// @param cudart CUDA runtime instance
    /// @param side Side of matrix A (CUBLAS_SIDE_LEFT for A on left, CUBLAS_SIDE_RIGHT for A on right).
    /// @param alpha Scalar multiplier for the matrix product.
    /// @param a Input symmetric GPU matrix A (Matrix_<float>).
    /// @param b Input GPU matrix B (Matrix_<float>).
    /// @param beta Scalar multiplier for matrix C.
    /// @param c Output GPU matrix C (destination / input-output, Matrix_<float>).
    void SymMatrixMatrixProduct(const CudaRuntime& cudart, const cublasSideMode_t side, const float alpha, const Matrix_<float>& a, const Matrix_<float>& b,
                                const float beta, Matrix_<float>& c);

    /// @brief Computes symmetric matrix-matrix product with single-precision GPU matrices (convenience overload).
    ///
    /// Performs C = A * B where A is symmetric (if side is left) 
    /// or C = alpha * B * A (if side is right)
    ///
    /// @param cudart CUDA runtime instance
    /// @param a Input symmetric GPU matrix A (Matrix_<float>).
    /// @param b Input GPU matrix B (Matrix_<float>).
    /// @param c Output GPU matrix C (destination / input-output, Matrix_<float>).
    /// @param alpha Scalar multiplier (default: 1.0).
    /// @param beta Scalar multiplier for C (default: 0.0).
    /// @param side Side of matrix A (default: CUBLAS_SIDE_LEFT).
    void SymMatrixMatrixProduct(const CudaRuntime& cudart, const Matrix_<float>& a, const Matrix_<float>& b, Matrix_<float>& c,
                                const float alpha = 1.0 , const float beta = 0.0, const cublasSideMode_t side = CUBLAS_SIDE_LEFT);

    /// @brief Computes matrix-matrix product with TF32 precision for accelerated single-precision computation.
    ///
    /// Performs C = alpha * op(A) * op(B) + beta * C using TF32 (tensor float 32) format.
    ///
    /// @param cudart CUDA runtime instance
    /// @param Ta Operation on matrix A ("N" for no transpose, "T" for transpose).
    /// @param Tb Operation on matrix B ("N" for no transpose, "T" for transpose).
    /// @param alpha Scalar multiplier for the matrix product.
    /// @param a Input GPU matrix A (Matrix_<float>).
    /// @param b Input GPU matrix B (Matrix_<float>).
    /// @param beta Scalar multiplier for matrix C.
    /// @param c Output GPU matrix C (destination / input-output, Matrix_<float>).
    void MatrixMatrixProductTF32(const CudaRuntime& cudart, const char* Ta, const char* Tb, const float alpha, const Matrix_<float>& a, const Matrix_<float>& b,
                             const float beta, Matrix_<float>& c);

    /// @brief Computes matrix-matrix product with TF32 precision (convenience overload).
    ///
    /// Performs C = alpha * op(A) * op(B) + beta * C using TF32 format with default parameters.
    ///
    /// @param cudart CUDA runtime instance
    /// @param a Input GPU matrix A (Matrix_<float>).
    /// @param b Input GPU matrix B (Matrix_<float>).
    /// @param c Output GPU matrix C (destination / input-output, Matrix_<float>).
    /// @param alpha Scalar multiplier (default: 1.0).
    /// @param beta Scalar multiplier for C (default: 0.0).
    /// @param Ta Operation on A (default: "N" for no transpose).
    /// @param Tb Operation on B (default: "N" for no transpose).
    void MatrixMatrixProductTF32(const CudaRuntime& cudart, const Matrix_<float>& a, const Matrix_<float>& b, Matrix_<float>& c,
                             const float alpha = 1.0 , const float beta = 0.0, const char* Ta = "N", const char* Tb = "N");


    /// @brief Computes matrix-matrix product with half-precision input and single-precision output (mixed precision).
    ///
    /// Performs C = alpha * op(A) * op(B) + beta * C with half-precision computation and float output.
    ///
    /// @param cudart CUDA runtime instance
    /// @param a Input GPU matrix A with half-precision elements (Matrix_<__half>).
    /// @param b Input GPU matrix B with half-precision elements (Matrix_<__half>).
    /// @param c Output GPU matrix C with single-precision elements (destination / input-output, Matrix_<float>).
    /// @param alpha Scalar multiplier (default: 1.0).
    /// @param beta Scalar multiplier for C (default: 0.0).
    /// @param Ta Operation on A (default: "N" for no transpose).
    /// @param Tb Operation on B (default: "N" for no transpose).
    void MatrixMatrixProductFP16(const CudaRuntime& cudart, const char* Ta, const char* Tb, const float alpha, const Matrix_<__half>& a, const Matrix_<__half>& b, 
        const float beta, Matrix_<float>& c);   

    /// @brief Computes matrix-matrix product with half-precision input and single-precision output (mixed precision).
    ///
    /// Performs C = alpha * op(A) * op(B) + beta * C with half-precision computation and float output 
    /// with the default parameters.
    ///
    /// @param cudart CUDA runtime instance
    /// @param a Input GPU matrix A with half-precision elements (Matrix_<__half>).
    /// @param b Input GPU matrix B with half-precision elements (Matrix_<__half>).
    /// @param c Output GPU matrix C with single-precision elements (destination / input-output, Matrix_<float>).
    /// @param alpha Scalar multiplier (default: 1.0).
    /// @param beta Scalar multiplier for C (default: 0.0).
    /// @param Ta Operation on A (default: "N" for no transpose).
    /// @param Tb Operation on B (default: "N" for no transpose).
    void MatrixMatrixProductFP16(const CudaRuntime& cudart, const Matrix_<__half>& a, const Matrix_<__half>& b, Matrix_<float>& c,
                             const float alpha = 1.0 , const float beta = 0.0, const char* Ta = "N", const char* Tb = "N");                         
    } // namespace gpu
} // namespace lahva