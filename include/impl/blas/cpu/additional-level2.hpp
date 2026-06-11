#ifndef LAHVA_ADD_LEVEL2_CPU_HPP
#define LAHVA_ADD_LEVEL2_CPU_HPP
#include "linalg.hpp"
#include "const.h"

namespace lahva
{
    namespace cpu
    {

        
        template <typename T>
        void AddMatrices(const char *Ta, const char *Tb, const T alpha, const Matrix_<T> &a, const T beta,
                         const Matrix_<T> &b, Matrix_<T> &c);
        template <typename T>
        void AddMatrices(const Matrix_<T> &a, const Matrix_<T> &b, Matrix_<T> &c,
                         const T alpha = 1.0, const T beta = 1.0, const char *Ta = "N", const char *Tb = "N");

        template <typename... Args>
        void AddMatrices(const CPURuntime &rt_, Args &&...args)
        {
            (AddMatrices(args...));
        }

        void MatrixVectorProduct(const char *T, const double alpha, const BlockDiagMatrix<double>& a, const Vector<double>& x,
                                 const int incx, const double beta, Vector<double>& y, const int incy);
        void MatrixVectorProduct(const BlockDiagMatrix<double>& a, const Vector<double>& x, Vector<double>& y, 
                                 const char *T="N", const double alpha=1.0, const int incx=1, const double beta=0.0, const int incy=1) {
                                    MatrixVectorProduct(T, alpha, a, x, incx, beta, y, incy);
                                 };
        
        void MatrixVectorProduct(const char *T, const float alpha, const BlockDiagMatrix<float>& a, const Vector<float>& x,
                                 const int incx, const float beta, Vector<float>& y, const int incy);
        void MatrixVectorProduct(const BlockDiagMatrix<float>& a, const Vector<float>& x, Vector<float>& y, 
                                 const char *T="N", const float alpha=1.0, const int incx=1, const float beta=0.0, const int incy=1) {
                                    MatrixVectorProduct(T, alpha, a, x, incx, beta, y, incy);
                                 };

    } // namespace gpu

}
#endif