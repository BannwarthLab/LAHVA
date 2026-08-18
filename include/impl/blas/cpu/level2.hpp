#ifndef LAHVA_C_LEVEL2_HPP
#define LAHVA_C_LEVEL2_HPP
#include "const.h"
#include "linalg.hpp"

namespace lahva{
    namespace cpu{
    void OuterVectorProduct(const Vector<double>& x, const Vector<double>& y, Matrix<double>& A, size_t incx = 1, size_t incy = 1, const double alpha = 1.0);
    void OuterVectorProduct(const Vector<float>& x, const Vector<float>& y, Matrix<float>& A, size_t incx = 1, size_t incy = 1, const float alpha = 1.0);

    void MatrixVectorProduct(const char* T, const double alpha, const Matrix_<double>& a, const Vector_<double>& x, 
                             const size_t incx, const double beta, Vector_<double>& y, const size_t incy);
    void MatrixVectorProduct(const Matrix_<double>& a, const Vector_<double>& x, Vector_<double>& y,
                             const char* T = "N", const double alpha = 1.0, const double beta = 0.0, const size_t incx = 1, const size_t incy = 1);
    void MatrixVectorProduct(const char* T, const float alpha, const Matrix_<float>& a, const Vector_<float>& x, 
                             const size_t incx, const float beta, Vector_<float>& y, const size_t incy);
    void MatrixVectorProduct(const Matrix_<float>& a, const Vector_<float>& x, Vector_<float>& y,
                             const char* T = "N", const float alpha = 1.0, const float beta = 0.0, const size_t incx = 1, const size_t incy = 1);

    void MatrixVectorProduct(const char* T, const complex_double alpha, const Matrix_<complex_double>& a, const Vector_<complex_double>& x, 
                             const size_t incx, const complex_double beta, Vector_<complex_double>& y, const size_t incy);
    void MatrixVectorProduct(const Matrix_<complex_double>& a, const Vector_<complex_double>& x, Vector_<complex_double>& y,
                             const char* T = "N", const complex_double alpha = 1.0, const complex_double beta = 0.0, const size_t incx = 1, const size_t incy = 1);
    void MatrixVectorProduct(const char* T, const complex_float alpha, const Matrix_<complex_float>& a, const Vector_<complex_float>& x, 
                             const size_t incx, const complex_float beta, Vector_<complex_float>& y, const size_t incy);
    void MatrixVectorProduct(const Matrix_<complex_float>& a, const Vector_<complex_float>& x, Vector_<complex_float>& y,
                             const char* T = "N", const complex_float alpha = 1.0, const complex_float beta = 0.0, const size_t incx = 1, const size_t incy = 1);

        /// @brief Block-diagonal matrix-vector multiply with full parameters: y = alpha * op(A) * x + beta * y
        ///
        /// Performs block-wise matrix-vector multiplication where A is a block-diagonal matrix.
        /// Supports transpose operations on each diagonal block.
        ///
        /// @tparam T Numeric element type (double, float).
        /// @param Ta   Transpose option for A: "N" (no transpose, A), "T" (transpose, A^T), "C" (conjugate-transpose, A^H).
        /// @param alpha Scaling factor for op(A)*x.
        /// @param a     Block-diagonal input matrix (BlockDiagMatrix_<T>).
        /// @param x     Input vector (Vector_<T>).
        /// @param incx  Stride for x (in elements).
        /// @param beta  Scaling factor applied to existing contents of y.
        /// @param y     Output vector (Vector_<T>).
        /// @param incy  Stride for y (in elements).
        template <typename T>
        void MatrixVectorProduct(const char *Ta, const T alpha, const BlockDiagMatrix_<T> &a, const Vector_<T> &x,
                                 const size_t incx, const T beta, Vector_<T> &y, const size_t incy);

        /// @brief Block-diagonal matrix-vector multiply: y = alpha * A * x + beta * y
        ///
        /// Performs block-wise matrix-vector multiplication where A is a block-diagonal matrix.
        /// Each block of A multiplies the corresponding portion of x using optimized BLAS
        ///
        /// @tparam T Numeric element type (double, float).
        /// @param alpha Scaling factor for A*x.
        /// @param a     Block-diagonal input matrix (BlockDiagMatrix_<T>).
        /// @param x     Input vector (Vector_<T>).
        /// @param beta  Scaling factor applied to existing contents of y.
        /// @param y     Output vector (Vector_<T>).
        template <typename T>
        void MatrixVectorProduct(const T alpha, const BlockDiagMatrix_<T> &a, const Vector_<T> &x,
                                 const T beta, Vector_<T> &y)
        {
            MatrixVectorProduct("N", alpha, a, x, 1, beta, y, 1);
        };

        /// @brief Convenience wrapper matching standard Matrix signature: y := op(A)*x + beta*y
        ///
        /// This template overload matches the standard Matrix interface with support for
        /// transpose, strides, and scalar factors. Supports the common case y := A*x with defaults.
        ///
        /// @tparam T Numeric element type (double, float).
        /// @param a     Block-diagonal input matrix (BlockDiagMatrix_<T>).
        /// @param x     Input vector (Vector_<T>).
        /// @param y     Output vector (Vector_<T>).
        /// @param Ta    Transpose option for A: "N" (default, no transpose), "T" (transpose, A^T), "C" (conjugate-transpose, A^H). Default: "N".
        /// @param alpha Scaling factor for op(A)*x. Default: 1.0.
        /// @param beta  Scaling factor for existing y. Default: 0.0.
        /// @param incx  Stride for x (in elements). Default: 1.
        /// @param incy  Stride for y (in elements). Default: 1.
        template <typename T>
        inline void MatrixVectorProduct(const BlockDiagMatrix_<T> &a, const Vector_<T> &x, Vector_<T> &y,
                                        const char *Ta = "N", const T alpha = static_cast<T>(1.0), const T beta = static_cast<T>(0.0), const size_t incx = 1, const size_t incy = 1)
        {
            MatrixVectorProduct(Ta, alpha, a, x, incx, beta, y, incy);
        };

        /// @brief Simplified convenience wrapper: y := alpha*A*x + beta*y with defaults.
        ///
        /// This template overload uses default transpose="N", alpha=1 and beta=0 to
        /// support the common case y := A*x, with convenience scalar type conversion.
        ///
        /// @tparam T Numeric element type (double, float).
        /// @param a     Block-diagonal input matrix (BlockDiagMatrix_<T>).
        /// @param x     Input vector (Vector_<T>).
        /// @param y     Output vector (Vector_<T>).
        /// @param alpha Scaling factor for A*x. Default: 1.0.
        /// @param beta  Scaling factor for existing y. Default: 0.0.
        template <typename T, typename Scalar = T, typename Scalar2 = T>
        void MatrixVectorProduct(const BlockDiagMatrix_<T> &a, const Vector_<T> &x, Vector_<T> &y,
                                 const Scalar alpha = 1.0, const Scalar2 beta = 0.0)
        {
            MatrixVectorProduct("N", (T)alpha, a, x, 1, (T)beta, y, 1);
        };


    void SymMatrixVectorProduct(const double alpha, const Matrix_<double>& a, const Vector_<double>& x, 
                             const size_t incx, const double beta, Vector_<double>& y, const size_t incy);
    void SymMatrixVectorProduct(const Matrix_<double>& a, const Vector_<double>& x, Vector_<double>& y,
                             const double alpha = 1.0, const double beta = 0.0, const size_t incx = 1, const size_t incy = 1);
    void SymMatrixVectorProduct(const float alpha, const Matrix_<float>& a, const Vector_<float>& x, 
                             const size_t incx, const float beta, Vector_<float>& y, const size_t incy);
    void SymMatrixVectorProduct(const Matrix_<float>& a, const Vector_<float>& x, Vector_<float>& y,
                             const float alpha = 1.0, const float beta = 0.0, const size_t incx = 1, const size_t incy = 1);

    void SymMatrixVectorProduct(const double alpha, const LowTriMatrix_<double>& a, const Vector_<double>& x, 
                             const size_t incx, const double beta, Vector_<double>& y, const size_t incy);
    void SymMatrixVectorProduct(const LowTriMatrix_<double>& a, const Vector_<double>& x, Vector_<double>& y,
                             const double alpha = 1.0, const double beta = 0.0, const size_t incx = 1, const size_t incy = 1);
    void SymMatrixVectorProduct(const float alpha, const LowTriMatrix_<float>& a, const Vector_<float>& x, 
                             const size_t incx, const float beta, Vector_<float>& y, const size_t incy);
    void SymMatrixVectorProduct(const LowTriMatrix_<float>& a, const Vector_<float>& x, Vector_<float>& y,
                             const float alpha = 1.0, const float beta = 0.0, const size_t incx = 1, const size_t incy = 1);
    
    void LowTriMatrixVectorProduct(const char* T, const CBLAS_DIAG unit, const LowTriMatrix_<double>& a, Vector_<double>& x, const size_t incx);
    void LowTriMatrixVectorProduct(const LowTriMatrix_<double>& a, Vector_<double>& x, const char* T = "N", const CBLAS_DIAG unit = CblasNonUnit, const size_t incx = 1);
    void LowTriMatrixVectorProduct(const char* T, const CBLAS_DIAG unit, const LowTriMatrix_<float>& a, Vector_<float>& x, const size_t incx);
    void LowTriMatrixVectorProduct(const LowTriMatrix_<float>& a, Vector_<float>& x, const char* T = "N", const CBLAS_DIAG unit = CblasNonUnit, const size_t incx = 1);
    
    template<typename... Args>
    void OuterVectorProduct(const CPURuntime& rt_, Args&&... args) {
        (OuterVectorProduct(args...));                    
    }


    template<typename... Args>
    void LowTriMatrixVectorProduct(const CPURuntime& rt_, Args&&... args) {
        (LowTriMatrixVectorProduct(args...));                    
    }

    template<typename... Args>
    void MatrixVectorProduct(const CPURuntime& rt_, Args&&... args) {
        (MatrixVectorProduct(args...));                
    }

    template<typename... Args>
    void SymMatrixVectorProduct(const CPURuntime& rt_, Args&&... args) {
        (SymMatrixVectorProduct(args...));                  
    }
    
    }
} 

#endif