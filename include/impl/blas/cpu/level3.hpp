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

    void MatrixMatrixProduct(const char* Ta, const char* Tb, const complex_double alpha, const Matrix_<complex_double>& a, 
                             const Matrix_<complex_double>& b, const complex_double beta, Matrix_<complex_double>& c);
    void MatrixMatrixProduct(const Matrix_<complex_double>& a, const Matrix_<complex_double>& b, Matrix_<complex_double>& c,
                             const complex_double alpha = 1.0 , const complex_double beta = 0.0, const char* Ta = "N", const char* Tb = "N");
    void MatrixMatrixProduct(const char* Ta, const char* Tb, const complex_float alpha, const Matrix_<complex_float>& a, 
                             const Matrix_<complex_float>& b, const complex_float beta, Matrix_<complex_float>& c);
    void MatrixMatrixProduct(const Matrix_<complex_float>& a, const Matrix_<complex_float>& b, Matrix_<complex_float>& c,
                             const complex_float alpha = 1.0 , const complex_float beta = 0.0, const char* Ta = "N", const char* Tb = "N");


/// @brief Block-diagonal matrix multiply: C = alpha * op(A) * op(B) + beta * C
        ///
        /// Performs C = alpha * op(A) * op(B) + beta * C where A is block-diagonal,
        /// and op(X) is X, X^T, or X^H depending on transpose flags.
        ///
        /// @param Ta    Transpose for A: "N" (none), "T" (transpose), "C" (conjugate-transpose).
        /// @param Tb    Transpose for B: "N" (none), "T" (transpose), "C" (conjugate-transpose).
        /// @param alpha Scaling factor for op(A)*op(B).
        /// @param a     Block-diagonal input matrix.
        /// @param b     Dense input matrix.
        /// @param beta  Scaling factor for existing c.
        /// @param c     Output matrix (modified).
        template <typename T>
        void MatrixMatrixProduct(const char *Ta, const char *Tb, const T alpha, const BlockDiagMatrix_<T> &a, const Matrix_<T> &b,
                                 const T beta, Matrix_<T> &c);

        /// @brief Block-diagonal multiply with default parameters, convenience overload.
        ///
        /// Performs C = alpha * op(A) * op(B) + beta * C with defaults: Ta="N", Tb="N".
        ///
        /// @param a     Block-diagonal input matrix.
        /// @param b     Dense input matrix.
        /// @param c     Output matrix (modified).
        /// @param alpha Scaling factor for A*B (default: 1.0).
        /// @param beta  Scaling factor for existing c (default: 0.0).
        /// @param Ta    Transpose for A (default: "N").
        /// @param Tb    Transpose for B (default: "N").
        template <typename T, typename Scalar = T, typename Scalar2 = T>
        void MatrixMatrixProduct(const BlockDiagMatrix_<T> &a, const Matrix_<T> &b, Matrix_<T> &c,
                                const Scalar alpha = 1.0, const Scalar2 beta = 0.0, const char *Ta = "N", const char *Tb = "N")
        {
            MatrixMatrixProduct(Ta, Tb, (T)alpha, a, b, (T)beta, c);
        };

        /// @brief Block-diagonal matrix multiply: C = alpha * op(A) * op(B) + beta * C
        ///
        /// Performs C = alpha * op(A) * op(B) + beta * C where B is block-diagonal,
        /// and op(X) is X, X^T, or X^H depending on transpose flags.
        ///
        /// @param Ta    Transpose for A: "N" (none), "T" (transpose), "C" (conjugate-transpose).
        /// @param Tb    Transpose for B: "N" (none), "T" (transpose), "C" (conjugate-transpose).
        /// @param alpha Scaling factor for op(A)*op(B).
        /// @param a     Block-diagonal input matrix.
        /// @param b     Dense input matrix.
        /// @param beta  Scaling factor for existing c.
        /// @param c     Output matrix (modified).
        template <typename T>
        void MatrixMatrixProduct(const char *Ta, const char *Tb, const T alpha, const Matrix_<T> &a, const BlockDiagMatrix_<T> &b,
                                 const T beta, Matrix_<T> &c);

        /// @brief Block-diagonal multiply with default parameters, convenience overload.
        ///
        /// Performs C = alpha * op(A) * op(B) + beta * C with defaults: Ta="N", Tb="N".
        ///
        /// @param a     Block-diagonal input matrix.
        /// @param b     Dense input matrix.
        /// @param c     Output matrix (modified).
        /// @param alpha Scaling factor for A*B (default: 1.0).
        /// @param beta  Scaling factor for existing c (default: 0.0).
        /// @param Ta    Transpose for A (default: "N").
        /// @param Tb    Transpose for B (default: "N").
        template <typename T, typename Scalar = T, typename Scalar2 = T>
        void MatrixMatrixProduct(const Matrix_<T> &a, const BlockDiagMatrix_<T> &b, Matrix_<T> &c,
                                const Scalar alpha = 1.0, const Scalar2 beta = 0.0, const char *Ta = "N", const char *Tb = "N")
        {
            MatrixMatrixProduct(Ta, Tb, (T)alpha, a, b, (T)beta, c);
        };


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
