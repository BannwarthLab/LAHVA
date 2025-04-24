#ifndef LAHVA_GPU_LEVEL2_HPP
#define LAHVA_GPU_LEVEL2_HPP
#include "const.h"
#include "linalg.hpp"
#include "runtime.hpp"

namespace lahva{
    namespace gpu{ 
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