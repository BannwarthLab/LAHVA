#ifndef LAHVA_C_LEVEL_3_H
#define LAHVA_C_LEVEL_3_H
#include "const.h"

namespace lahva{
    namespace cpu{
    void MatrixMatrixProduct(const char* Ta, const char* Tb, const size_t m, const size_t n, const size_t k, 
                             const double alpha, const double* a, const double* b, const double beta, double* c);
    void MatrixMatrixProduct(const size_t m, const size_t n, const size_t k, const double* a, const double* b, double* c,
                             const double alpha = 1.0 , const double beta = 0.0, const char* Ta = "N", const char* Tb = "N");
    void MatrixMatrixProduct(const char* Ta, const char* Tb, const size_t m, const size_t n, const size_t k, 
                             const float alpha, const float* a, const float* b, const float beta, float* c);
    void MatrixMatrixProduct(const size_t m, const size_t n, const size_t k, const float* a, const float* b, float* c,
                             const float alpha = 1.0 , const float beta = 0.0, const char* Ta = "N", const char* Tb = "N");

    void SymMatrixMatrixProduct(const CBLAS_SIDE side, const size_t m, const size_t n, const double alpha, const double* a, 
                                const double* b, const double beta, double* c);
    void SymMatrixMatrixProduct(const size_t m, const size_t n, const double* a, const double* b, double* c,
                                const double alpha = 1.0 , const double beta = 0.0, const CBLAS_SIDE side = CblasLeft);
    void SymMatrixMatrixProduct(const CBLAS_SIDE side, const size_t m, const size_t n, const float alpha, const float* a, 
                                const float* b, const float beta, float* c);
    void SymMatrixMatrixProduct(const size_t m, const size_t n, const float* a, const float* b, float* c,
                                const float alpha = 1.0 , const float beta = 0.0, const CBLAS_SIDE side = CblasLeft);
    }
}
#endif