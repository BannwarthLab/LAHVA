#ifndef LAHVA_ADD_LEVEL3_CPU_HPP
#define LAHVA_ADD_LEVEL3_CPU_HPP
#include "linalg.hpp"
#include "const.h"

namespace lahva
{
    namespace cpu
    {

        void MatrixMatrixProduct(const char* Ta, const char* Tb, const double alpha, const BlockDiagMatrix<double>& a, const Matrix_<double>& b,
                             const double beta, Matrix_<double>& c);
        void MatrixMatrixProduct(const BlockDiagMatrix<double>& a, const Matrix_<double>& b, Matrix_<double>& c,
                             const double alpha = 1.0 , const double beta = 0.0, const char* Ta = "N", const char* Tb = "N") {
                                MatrixMatrixProduct(Ta, Tb, alpha, a, b, beta, c);
                             };
        void MatrixMatrixProduct(const char* Ta, const char* Tb, const float alpha, const BlockDiagMatrix<float>& a, const Matrix_<float>& b,
                             const float beta, Matrix_<float>& c);
        void MatrixMatrixProduct(const BlockDiagMatrix<float>& a, const Matrix_<float>& b, Matrix_<float>& c,
                             const float alpha = 1.0 , const float beta = 0.0, const char* Ta = "N", const char* Tb = "N") {
                                MatrixMatrixProduct(Ta, Tb, alpha, a, b, beta, c);
                             };

        void MatrixMatrixProduct(const char* Ta, const char* Tb, const double alpha, const Matrix_<double>& a, const BlockDiagMatrix<double>& b,
                             const double beta, Matrix_<double>& c);
        void MatrixMatrixProduct(const Matrix_<double>& a, const BlockDiagMatrix<double>& b, Matrix_<double>& c,
                             const double alpha = 1.0 , const double beta = 0.0, const char* Ta = "N", const char* Tb = "N") {
                                MatrixMatrixProduct(Ta, Tb, alpha, a, b, beta, c);
                             };
         
         void MatrixMatrixProduct(const char* Ta, const char* Tb, const float alpha, const Matrix_<float>& a, const BlockDiagMatrix<float>& b,
                             const float beta, Matrix_<float>& c);
         void MatrixMatrixProduct(const Matrix_<float>& a, const BlockDiagMatrix<float>& b, Matrix_<float>& c,
                             const float alpha = 1.0 , const float beta = 0.0, const char* Ta = "N", const char* Tb = "N") {
                                MatrixMatrixProduct(Ta, Tb, alpha, a, b, beta, c);
                             };


    } // namespace cpu

}
#endif