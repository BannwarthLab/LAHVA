#ifndef TCGMBLAS_GPU_LEVEL_3_HPP
#define TCGMBLAS_GPU_LEVEL_3_HPP
#include "linalg.hpp"
#include "const.h"
#include "runtime.hpp"
namespace tcgmtensor{
    namespace gpu {
    void MatrixMatrixProduct(const CudaRuntime& cudart, const char* Ta, const char* Tb, const double alpha, const Matrix<double>& a, const Matrix<double>& b,
                             const double beta, Matrix<double>& c);
    void MatrixMatrixProduct(const CudaRuntime& cudart, const Matrix<double>& a, const Matrix<double>& b, Matrix<double> c,
                             const double alpha = 1.0 , const double beta = 0.0, const char* Ta = "N", const char* Tb = "N");
    void MatrixMatrixProduct(const CudaRuntime& cudart, const char* Ta, const char* Tb, const float alpha, const Matrix<float>& a, const Matrix<float>& b,
                             const float beta, Matrix<float>& c);
    void MatrixMatrixProduct(const CudaRuntime& cudart, const Matrix<float>& a, const Matrix<float>& b, Matrix<float> c,
                             const float alpha = 1.0 , const float beta = 0.0, const char* Ta = "N", const char* Tb = "N");

    void SymMatrixMatrixProduct(const CudaRuntime& cudart, const cublasSideMode_t side, const double alpha, const Matrix<double>& a, const Matrix<double>& b,
                                const double beta, Matrix<double>& c);
    void SymMatrixMatrixProduct(const CudaRuntime& cudart, const Matrix<double>& a, const Matrix<double>& b, Matrix<double>& c,
                                const double alpha = 1.0 , const double beta = 0.0, const cublasSideMode_t side = CUBLAS_SIDE_LEFT);
    void SymMatrixMatrixProduct(const CudaRuntime& cudart, const cublasSideMode_t side, const float alpha, const Matrix<float>& a, const Matrix<float>& b,
                                const float beta, Matrix<float>& c);
    void SymMatrixMatrixProduct(const CudaRuntime& cudart, const Matrix<float>& a, const Matrix<float>& b, Matrix<float>& c,
                                const float alpha = 1.0 , const float beta = 0.0, const cublasSideMode_t side = CUBLAS_SIDE_LEFT);
    }
}
#endif