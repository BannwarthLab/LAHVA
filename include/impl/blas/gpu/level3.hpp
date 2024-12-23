#ifndef TCGMBLAS_GPU_LEVEL_3_HPP
#define TCGMBLAS_GPU_LEVEL_3_HPP
#include "linalg.hpp"
#include "const.h"
#include "runtime.hpp"
namespace tcgmtensor{
    namespace gpu {
    void MatrixMatrixProduct(const CudaRuntime& cudart, const char* Ta, const char* Tb, const double alpha, const Matrix_<double>& a, const Matrix_<double>& b,
                             const double beta, Matrix_<double>& c);
    void MatrixMatrixProduct(const CudaRuntime& cudart, const Matrix_<double>& a, const Matrix_<double>& b, Matrix_<double>& c,
                             const double alpha = 1.0 , const double beta = 0.0, const char* Ta = "N", const char* Tb = "N");
    void MatrixMatrixProduct(const CudaRuntime& cudart, const char* Ta, const char* Tb, const float alpha, const Matrix_<float>& a, const Matrix_<float>& b,
                             const float beta, Matrix_<float>& c);
    void MatrixMatrixProduct(const CudaRuntime& cudart, const Matrix_<float>& a, const Matrix_<float>& b, Matrix_<float>& c,
                             const float alpha = 1.0 , const float beta = 0.0, const char* Ta = "N", const char* Tb = "N");

    void SymMatrixMatrixProduct(const CudaRuntime& cudart, const cublasSideMode_t side, const double alpha, const Matrix_<double>& a, const Matrix_<double>& b,
                                const double beta, Matrix_<double>& c);
    void SymMatrixMatrixProduct(const CudaRuntime& cudart, const Matrix_<double>& a, const Matrix_<double>& b, Matrix_<double>& c,
                                const double alpha = 1.0 , const double beta = 0.0, const cublasSideMode_t side = CUBLAS_SIDE_LEFT);
    void SymMatrixMatrixProduct(const CudaRuntime& cudart, const cublasSideMode_t side, const float alpha, const Matrix_<float>& a, const Matrix_<float>& b,
                                const float beta, Matrix_<float>& c);
    void SymMatrixMatrixProduct(const CudaRuntime& cudart, const Matrix_<float>& a, const Matrix_<float>& b, Matrix_<float>& c,
                                const float alpha = 1.0 , const float beta = 0.0, const cublasSideMode_t side = CUBLAS_SIDE_LEFT);
    }
}
#endif