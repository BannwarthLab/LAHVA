#ifndef LAHVA_C_LEVEL2_H
#define LAHVA_C_LEVEL2_H
#include "const.h"

namespace lahva{
    namespace cpu{ 
    void OuterVectorProduct(const size_t ndimX, const double* x, const size_t incx, const size_t ndimY, const double* y, const size_t incy, const double alpha, double* A);
    void OuterVectorProduct(const size_t ndimX, const float* x, const size_t incx, const size_t ndimY, const float* y, const size_t incy, const float alpha, float* A);
    
    void MatrixVectorProduct(const char* T, const size_t m, const size_t n, const double alpha, const double* a, 
                             const double* x, const size_t incx, const double beta, double* y, const size_t incy);
    void MatrixVectorProduct(const size_t m, const size_t n,  const double* a, const double* x, double* y,
                             const char* T = "N", const double alpha = 1.0, const double beta = 0.0, const size_t incx = 1, const size_t incy = 1);
    void MatrixVectorProduct(const char* T, const size_t m, const size_t n, const float alpha, const float* a, 
                             const float* x, const size_t incx, const float beta, float* y, const size_t incy);
    void MatrixVectorProduct(const size_t m, const size_t n,  const float* a, const float* x, float* y,
                             const char* T = "N", const float alpha = 1.0, const float beta = 0.0, const size_t incx = 1, const size_t incy = 1);                    

    void SymMatrixVectorProduct(const size_t n, const double alpha, const double* a, 
                            const double* x, const size_t incx, const double beta, double* y, const size_t incy);
    void SymMatrixVectorProduct(const size_t n,  const double* a, const double* x, double* y,
                            const double alpha = 1.0, const double beta = 0.0, const size_t incx = 1, const size_t incy = 1);
    void SymMatrixVectorProduct(const size_t n, const float alpha, const float* a, 
                            const float* x, const size_t incx, const float beta, float* y, const size_t incy);
    void SymMatrixVectorProduct(const size_t n,  const float* a, const float* x, float* y,
                            const float alpha = 1.0, const float beta = 0.0, const size_t incx = 1, const size_t incy = 1);
    }
} 

#endif