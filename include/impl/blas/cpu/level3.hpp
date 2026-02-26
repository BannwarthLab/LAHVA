#ifndef LAHVA_C_LEVEL_3_HPP
#define LAHVA_C_LEVEL_3_HPP
#include "linalg.hpp"
#include "const.h"
namespace lahva
{
    namespace cpu
    {
        /// @brief Multiply two matrices of type Matrix_<T> and store the result in `c`.
        ///
        /// This C++-style wrapper performs size and transpose checks on Matrix_<T>
        /// instances and forwards the call to the lower-level pointer-based
        /// MatrixMatrixProduct implementation declared in the C-style header.
        ///
        /// @tparam T Numeric element type (double, float, complex_double, complex_float).
        /// @param Ta    Transpose option for A ("N","T","C").
        /// @param Tb    Transpose option for B ("N","T","C").
        /// @param alpha Scaling factor for op(A)*op(B).
        /// @param a     Left-hand input matrix (Matrix_<T>).
        /// @param b     Right-hand input matrix (Matrix_<T>).
        /// @param beta  Scaling factor applied to existing contents of `c`.
        /// @param c     Output matrix which will receive the result.
        template <typename T>
        void MatrixMatrixProduct(const char *Ta, const char *Tb, const T alpha, const Matrix_<T> &a, const Matrix_<T> &b,
                                 const T beta, Matrix_<T> &c);

        /// @brief Convenience overload allowing scalar types differing from matrix element type.
        ///
        /// Casts provided scalar types to `T` and forwards to the primary
        /// MatrixMatrixProduct implementation. Useful when calling with double
        /// literals against float matrices or other mixed-precision convenience calls.
        ///
        /// @tparam T      Matrix element type.
        /// @tparam Scalar Scalar type for `alpha`.
        /// @tparam Scalar2 Scalar type for `beta`.
        template <typename T, typename Scalar, typename Scalar2>
        void MatrixMatrixProduct(const char *Ta, const char *Tb, const Scalar alpha, const Matrix_<T> &a, const Matrix_<T> &b,
                                 const Scalar2 beta, Matrix_<T> &c)
        {
            MatrixMatrixProduct(Ta, Tb, (T)alpha, a, b, (T)beta, c);
        };

        /// @brief Convenience overload that uses default transpose flags and scalar defaults.
        ///
        /// Calls MatrixMatrixProduct(Ta,Tb,...) with `Ta`/`Tb` defaulting to "N" and
        /// alpha/beta defaulting to 1/0 respectively. Useful for the common case C = A*B.
        ///
        /// @tparam T Matrix element type.
        /// @tparam Scalar Scalar type for `alpha` (defaults to 1.0 if omitted).
        /// @tparam Scalar2 Scalar type for `beta` (defaults to 0.0 if omitted).
        template <typename T, typename Scalar, typename Scalar2>
        void MatrixMatrixProduct(const Matrix_<T> &a, const Matrix_<T> &b, Matrix_<T> &c,
                                 const Scalar alpha = 1.0, const Scalar2 beta = 0.0, const char *Ta = "N", const char *Tb = "N")
        {
            MatrixMatrixProduct(Ta, Tb, (T)alpha, a, b, (T)beta, c);
        };

        /// @brief Symmetric matrix-matrix multiply (double precision).
        ///
        /// Performs C = alpha * A * B + beta * C when `side == CblasLeft`, or
        /// C = alpha * B * A + beta * C when `side == CblasRight`. A is assumed symmetric.
        ///
        /// @param side Side on which the symmetric matrix A appears (CblasLeft/CblasRight).
        /// @param alpha Scaling factor for the product involving A.
        /// @param a Symmetric matrix A.
        /// @param b Matrix B.
        /// @param beta Scaling factor for existing contents of C.
        /// @param c Output matrix C (destination / input-output).
        void SymMatrixMatrixProduct(const CBLAS_SIDE side, const double alpha, const Matrix_<double> &a, const Matrix_<double> &b,
                                    const double beta, Matrix_<double> &c);

        /// @brief Convenience overload for double-precision symmetric multiply using defaults.
        ///
        /// Defaults to left-side application (CblasLeft) and alpha=1.0, beta=0.0.
        inline void SymMatrixMatrixProduct(const Matrix_<double> &a, const Matrix_<double> &b, Matrix_<double> &c,
                                           const double alpha = 1.0, const double beta = 0.0, const CBLAS_SIDE side = CblasLeft)
        {
            SymMatrixMatrixProduct(side, alpha, a, b, beta, c);
        };
        /// @brief Symmetric matrix-matrix multiply (single precision).
        ///
        /// Float variant of the symmetric multiply.
        void SymMatrixMatrixProduct(const CBLAS_SIDE side, const float alpha, const Matrix_<float> &a, const Matrix_<float> &b,
                                    const float beta, Matrix_<float> &c);

        /// @brief Convenience overload for single-precision symmetric multiply using defaults.
        inline void SymMatrixMatrixProduct(const Matrix_<float> &a, const Matrix_<float> &b, Matrix_<float> &c,
                                           const float alpha = 1.0, const float beta = 0.0, const CBLAS_SIDE side = CblasLeft)
        {
            SymMatrixMatrixProduct(side, alpha, a, b, beta, c);
        };

        /// @brief Runtime-dispatching overload that allows passing a CPURuntime first.
        ///
        /// The runtime parameter is ignored for CPU implementations but this overload
        /// mirrors the signatures used for GPU/CUDA dispatching in the project API.
        template <typename... Args>
        void MatrixMatrixProduct(const CPURuntime &rt_, Args &&...args)
        {
            (MatrixMatrixProduct(args...));
        }

        /// @brief Runtime-dispatching overload for symmetric multiply (CPURuntime first).
        ///
        /// The runtime parameter is ignored for CPU implementations but keeps the API
        /// consistent with other backend dispatchers.
        template <typename... Args>
        void SymMatrixMatrixProduct(const CPURuntime &rt_, Args &&...args)
        {
            (SymMatrixMatrixProduct(args...));
        }

    }
}
#endif
