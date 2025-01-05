#ifndef LAHVA_C_LEVEL_3_HPP
#define LAHVA_C_LEVEL_3_HPP
#include "linalg.hpp"
#include "const.h"

namespace lahva{
    namespace cpu {
    void MatrixMatrixProduct(const char* Ta, const char* Tb, const double alpha, const Matrix_<double>& a, const Matrix_<double>& b,
                             const double beta, Matrix_<double>& c);
    void MatrixMatrixProduct(const Matrix_<double>& a, const Matrix_<double>& b, Matrix_<double>& c,
                             const double alpha = 1.0 , const double beta = 0.0, const char* Ta = "N", const char* Tb = "N");
    void MatrixMatrixProduct(const char* Ta, const char* Tb, const float alpha, const Matrix_<float>& a, const Matrix_<float>& b,
                             const float beta, Matrix_<float>& c);
    void MatrixMatrixProduct(const Matrix_<float>& a, const Matrix_<float>& b, Matrix_<float>& c,
                             const float alpha = 1.0 , const float beta = 0.0, const char* Ta = "N", const char* Tb = "N");

    void SymMatrixMatrixProduct(const CBLAS_SIDE side, const double alpha, const Matrix_<double>& a, const Matrix_<double>& b,
                                const double beta, Matrix_<double>& c);
    void SymMatrixMatrixProduct(const Matrix_<double>& a, const Matrix_<double>& b, Matrix_<double>& c,
                                const double alpha = 1.0 , const double beta = 0.0, const CBLAS_SIDE side = CblasLeft);
    void SymMatrixMatrixProduct(const CBLAS_SIDE side, const float alpha, const Matrix_<float>& a, const Matrix_<float>& b,
                                const float beta, Matrix_<float>& c);
    void SymMatrixMatrixProduct(const Matrix_<float>& a, const Matrix_<float>& b, Matrix_<float>& c,
                                const float alpha = 1.0 , const float beta = 0.0, const CBLAS_SIDE side = CblasLeft);
    
    template<typename... Args>
    void MatrixMatrixProduct(const CPURuntime& rt_, Args&&... args) {
        (MatrixMatrixProduct(args...));                
    }

    template<typename... Args>
    void SymMatrixMatrixProduct(const CPURuntime& rt_, Args&&... args) {
        (SymMatrixMatrixProduct(args...));                  
    }


    }
}
#endif
