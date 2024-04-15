#ifndef TCGMBLAS_GPU_LEVEL2_HPP
#define TCGMBLAS_GPU_LEVEL2_HPP
#include "const.h"
#include "linalg.hpp"
#include "runtime.hpp"

namespace tcgmtensor{
    namespace gpu{ 
    void MatrixVectorProduct(const CudaRuntime& cudart, const char* T, const double alpha, const Matrix<double>& a, const Vector<double>& x, 
                             const size_t incx, const double beta, Vector<double>& y, const size_t incy);
    void MatrixVectorProduct(const CudaRuntime& cudart, const Matrix<double>& a, const Vector<double>& x, Vector<double>& y,
                             const char* T = "N", const double alpha = 1.0, const double beta = 0.0, const size_t incx = 1, const size_t incy = 1);
    void MatrixVectorProduct(const CudaRuntime& cudart, const char* T, const float alpha, const Matrix<float>& a, const Vector<float>& x, 
                             const size_t incx, const float beta, Vector<float>& y, const size_t incy);
    void MatrixVectorProduct(const CudaRuntime& cudart, const Matrix<float>& a, const Vector<float>& x, Vector<float>& y,
                             const char* T = "N", const float alpha = 1.0, const float beta = 0.0, const size_t incx = 1, const size_t incy = 1);

    void SymMatrixVectorProduct(const CudaRuntime& cudart, const double alpha, const Matrix<double>& a, const Vector<double>& x, 
                             const size_t incx, const double beta, Vector<double>& y, const size_t incy);
    void SymMatrixVectorProduct(const CudaRuntime& cudart, const Matrix<double>& a, const Vector<double>& x, Vector<double>& y,
                             const double alpha = 1.0, const double beta = 0.0, const size_t incx = 1, const size_t incy = 1);
    void SymMatrixVectorProduct(const CudaRuntime& cudart, const float alpha, const Matrix<float>& a, const Vector<float>& x, 
                             const size_t incx, const float beta, Vector<float>& y, const size_t incy);
    void SymMatrixVectorProduct(const CudaRuntime& cudart, const Matrix<float>& a, const Vector<float>& x, Vector<float>& y,
                             const float alpha = 1.0, const float beta = 0.0, const size_t incx = 1, const size_t incy = 1);

        void SymMatrixVectorProduct(const CudaRuntime& cudart, const double alpha, const LowTriMatrix<double>& a, const Vector<double>& x, 
                             const size_t incx, const double beta, Vector<double>& y, const size_t incy);
    void SymMatrixVectorProduct(const CudaRuntime& cudart, const LowTriMatrix<double>& a, const Vector<double>& x, Vector<double>& y,
                             const double alpha = 1.0, const double beta = 0.0, const size_t incx = 1, const size_t incy = 1);
    void SymMatrixVectorProduct(const CudaRuntime& cudart, const float alpha, const LowTriMatrix<float>& a, const Vector<float>& x, 
                             const size_t incx, const float beta, Vector<float>& y, const size_t incy);
    void SymMatrixVectorProduct(const CudaRuntime& cudart, const LowTriMatrix<float>& a, const Vector<float>& x, Vector<float>& y,
                             const float alpha = 1.0, const float beta = 0.0, const size_t incx = 1, const size_t incy = 1);

    //void LowTriMatrixVectorProduct(const CudaRuntime, const char* T, const CBLAS_DIAG unit,const LowTriMatrix<double>& a, Vector<double>& x, const size_t incx);
    //void LowTriMatrixVectorProduct(const CudaRuntime, const LowTriMatrix<double>& a, Vector<double>& x, const char* T = "N", const CBLAS_DIAG unit = CblasNonUnit, const size_t incx = 1);
    //void LowTriMatrixVectorProduct(const CudaRuntime, const char* T, const CBLAS_DIAG unit, const LowTriMatrix<float>& a, Vector<float>& x, const size_t incx);
    //void LowTriMatrixVectorProduct(const CudaRuntime, const LowTriMatrix<float>& a, Vector<float>& x, const char* T = "N", const CBLAS_DIAG unit = CblasNonUnit, const size_t incx = 1);
    }
} 

#endif