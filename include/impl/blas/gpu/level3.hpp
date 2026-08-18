#ifndef LAHVA_GPU_LEVEL_3_HPP
#define LAHVA_GPU_LEVEL_3_HPP
#include "linalg.hpp"
#include "const.h"
#include "runtime.hpp"
namespace lahva{
    namespace gpu {
    void MatrixMatrixProduct(const CudaRuntime& cudart, const char* Ta, const char* Tb, const double alpha, const Matrix_<double>& a, const Matrix_<double>& b,
                             const double beta, Matrix_<double>& c);
    void MatrixMatrixProduct(const CudaRuntime& cudart, const Matrix_<double>& a, const Matrix_<double>& b, Matrix_<double>& c,
                             const double alpha = 1.0 , const double beta = 0.0, const char* Ta = "N", const char* Tb = "N");
    void MatrixMatrixProduct(const CudaRuntime& cudart, const char* Ta, const char* Tb, const float alpha, const Matrix_<float>& a, const Matrix_<float>& b,
                             const float beta, Matrix_<float>& c);
    void MatrixMatrixProduct(const CudaRuntime& cudart, const Matrix_<float>& a, const Matrix_<float>& b, Matrix_<float>& c,
                             const float alpha = 1.0 , const float beta = 0.0, const char* Ta = "N", const char* Tb = "N");

    void MatrixMatrixProduct(const CudaRuntime& cudart, const char* Ta, const char* Tb, const complex_double alpha, const Matrix_<complex_double>& a, 
                             const Matrix_<complex_double>& b, const complex_double beta, Matrix_<complex_double>& c);
    void MatrixMatrixProduct(const CudaRuntime& cudart, const Matrix_<complex_double>& a, const Matrix_<complex_double>& b, Matrix_<complex_double>& c,
                             const complex_double alpha = 1.0 , const complex_double beta = 0.0, const char* Ta = "N", const char* Tb = "N");
    void MatrixMatrixProduct(const CudaRuntime& cudart, const char* Ta, const char* Tb, const complex_float alpha, const Matrix_<complex_float>& a, 
                             const Matrix_<complex_float>& b, const complex_float beta, Matrix_<complex_float>& c);
    void MatrixMatrixProduct(const CudaRuntime& cudart, const Matrix_<complex_float>& a, const Matrix_<complex_float>& b, Matrix_<complex_float>& c,
                             const complex_float alpha = 1.0 , const complex_float beta = 0.0, const char* Ta = "N", const char* Tb = "N");

    void SymMatrixMatrixProduct(const CudaRuntime& cudart, const cublasSideMode_t side, const double alpha, const Matrix_<double>& a, const Matrix_<double>& b,
                                const double beta, Matrix_<double>& c);
    void SymMatrixMatrixProduct(const CudaRuntime& cudart, const Matrix_<double>& a, const Matrix_<double>& b, Matrix_<double>& c,
                                const double alpha = 1.0 , const double beta = 0.0, const cublasSideMode_t side = CUBLAS_SIDE_LEFT);
    void SymMatrixMatrixProduct(const CudaRuntime& cudart, const cublasSideMode_t side, const float alpha, const Matrix_<float>& a, const Matrix_<float>& b,
                                const float beta, Matrix_<float>& c);
    void SymMatrixMatrixProduct(const CudaRuntime& cudart, const Matrix_<float>& a, const Matrix_<float>& b, Matrix_<float>& c,
                                const float alpha = 1.0 , const float beta = 0.0, const cublasSideMode_t side = CUBLAS_SIDE_LEFT);
    void MatrixMatrixProductTF32(const CudaRuntime& cudart, const char* Ta, const char* Tb, const float alpha, const Matrix_<float>& a, const Matrix_<float>& b,
                             const float beta, Matrix_<float>& c);
    void MatrixMatrixProductTF32(const CudaRuntime& cudart, const Matrix_<float>& a, const Matrix_<float>& b, Matrix_<float>& c,
                             const float alpha = 1.0 , const float beta = 0.0, const char* Ta = "N", const char* Tb = "N");

    void MatrixMatrixProductFP16(const CudaRuntime& cudart, const char* Ta, const char* Tb, const float alpha, const Matrix_<__half>& a, const Matrix_<__half>& b,
                             const float beta, Matrix_<float>& c);
    void MatrixMatrixProductFP16(const CudaRuntime& cudart, const Matrix_<__half>& a, const Matrix_<__half>& b, Matrix_<float>& c,
                             const float alpha = 1.0 , const float beta = 0.0, const char* Ta = "N", const char* Tb = "N");
    
    /// @brief Computes matrix-matrix product using sparse format for BlockMatrix.
    ///
    /// Performs C = alpha * A * B + beta * C where A is a BlockMatrix
    ///
    /// @param cudart CUDA runtime instance with cusparse handle
    /// @param Ta Operation on A ("N" for no transpose, "T" for transpose)
    /// @param Tb Operation on B ("N" for no transpose, "T" for transpose)
    /// @param alpha Scalar multiplier for the matrix product (double precision)
    /// @param a Input block matrix (sparse representation, BlockMatrix_<double>)
    /// @param b Input dense matrix B (Matrix_<double>)
    /// @param beta Scalar multiplier for result matrix C
    /// @param c Output/result matrix C (Matrix_<double>)
    void MatrixMatrixProduct(const CudaRuntime& cudart, const char* Ta, const char* Tb,
                            const double alpha, const BlockMatrix_<double>& a,
                            const Matrix_<double>& b, const double beta, Matrix_<double>& c);

    /// @brief Computes matrix-matrix product with double-precision GPU matrices (convenience overload).
    ///
    /// Performs C = alpha * op(A) * op(B) + beta * C with default parameters.
    ///
    /// @param cudart CUDA runtime instance with cusparse handle
    /// @param a Input block matrix using sparse format (BlockMatrix_<double>)
    /// @param b Input dense matrix B (Matrix_<double>)
    /// @param c Input/output result matrix C (Matrix_<double>)
    /// @param alpha Scalar multiplier for the matrix product (default: 1.0)
    /// @param beta Scalar multiplier for result matrix C (default: 0.0)
    /// @param Ta Operation on A: "N" for no transpose, "T" for transpose (default: "N")
    /// @param Tb Operation on B: "N" for no transpose, "T" for transpose (default: "N")
    inline void MatrixMatrixProduct(const CudaRuntime& cudart, const BlockMatrix_<double>& a,
                            const Matrix_<double>& b, Matrix_<double>& c,
                            const double alpha = 1.0, const double beta = 0.0, const char* Ta = "N", const char* Tb = "N")
    {
        MatrixMatrixProduct(cudart, Ta, Tb, alpha, a, b, beta, c);
    }

    /// @brief Computes matrix-matrix product using sparse format for BlockMatrix (single-precision).
    void MatrixMatrixProduct(const CudaRuntime& cudart, const char* Ta, const char* Tb,
                            const float alpha, const BlockMatrix_<float>& a,
                            const Matrix_<float>& b, const float beta, Matrix_<float>& c);

    /// @brief Computes matrix-matrix product with single-precision GPU matrices (convenience overload).
    ///
    /// Performs C = alpha * op(A) * op(B) + beta * C with default parameters.
    ///
    /// @param cudart CUDA runtime instance with cusparse handle
    /// @param a Input block matrix using sparse format (BlockMatrix_<float>)
    /// @param b Input dense matrix B (Matrix_<float>)
    /// @param c Input/output result matrix C (Matrix_<float>)
    /// @param alpha Scalar multiplier for the matrix product (default: 1.0)
    /// @param beta Scalar multiplier for result matrix C (default: 0.0)
    /// @param Ta Operation on A: "N" for no transpose, "T" for transpose (default: "N")
    /// @param Tb Operation on B: "N" for no transpose, "T" for transpose (default: "N")
    inline void MatrixMatrixProduct(const CudaRuntime& cudart, const BlockMatrix_<float>& a,
                            const Matrix_<float>& b, Matrix_<float>& c,
                            const float alpha = 1.0, const float beta = 0.0, const char* Ta = "N", const char* Tb = "N")
    {
        MatrixMatrixProduct(cudart, Ta, Tb, alpha, a, b, beta, c);
    }
    
    }
}
#endif