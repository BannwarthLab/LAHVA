#ifndef LAHVA_ADD_LEVEL2_CPU_HPP
#define LAHVA_ADD_LEVEL2_CPU_HPP
#include "linalg.hpp"
#include "const.h"

namespace lahva
{
    namespace cpu
    {

        /// @brief Add two matrices: C = alpha*op(A) + beta*op(B)
        /// @tparam T Numerical type of the matrices
        /// @param Ta char to specify if matrix A is transposed ("T") or not ("N")
        /// @param Tb char to specify if matrix B is transposed ("T") or not ("N")
        /// @param alpha Scalar multiplier for matrix A
        /// @param a Matrix A
        /// @param beta Scalar multiplier for matrix B
        /// @param b Matrix B
        /// @param c Matrix to store the result in
        template <typename T>
        void AddMatrices(const char *Ta, const char *Tb, const T alpha, const Matrix_<T> &a, const T beta,
                         const Matrix_<T> &b, Matrix_<T> &c);

        /// @brief Add two matrices: C = alpha*op(A) + beta*op(B)
        /// @tparam T Numerical type of the matrices
        /// @param a Matrix A
        /// @param b Matrix B
        /// @param c Matrix to store the result in
        /// @param alpha Scalar multiplier for matrix A
        /// @param beta Scalar multiplier for matrix B
        /// @param Ta char to specify if matrix A is transposed ("T") or not ("N"), default value is "N"
        /// @param Tb char to specify if matrix B is transposed ("T") or not ("N"), default value is "N"
        template <typename T>
        void AddMatrices(const Matrix_<T> &a, const Matrix_<T> &b, Matrix_<T> &c,
                         const T alpha = 1.0, const T beta = 1.0, const char *Ta = "N", const char *Tb = "N");

                /// @brief Wrapper function to add two matrices, with CPURuntime as first argument
        /// @tparam ...Args the additional input arguments forwarded to AddMatrices
        /// @param rt_ the CPURuntime instance, which is ignored
        /// @param ...args the additional input arguments forwarded to AddMatrices
        template <typename... Args>
        void AddMatrices(const CPURuntime &rt_, Args &&...args)
        {
            (AddMatrices(args...));
        }

    } // namespace gpu

}
#endif