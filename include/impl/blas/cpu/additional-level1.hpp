#ifndef LAHVA_ADD_LEVEL1_CPU_HPP
#define LAHVA_ADD_LEVEL1_CPU_HPP
#include <memory>
#include <vector>
#include <iterator>
#include <assert.h>
#include <iostream>
#include "impl/tensor/cpu/tensor.hpp"
#include "impl/tensor/cpu/vector.hpp"
#include "impl/tensor/cpu/matrix.hpp"
#include "impl/tensor/cpu/lowtrimatrix.hpp"
#include "impl/tensor/allocators.hpp"
#include "const.h"
#include "level1.h"

namespace lahva
{
    namespace cpu
    {
        /// @brief Compute the trace of a matrix
        /// @tparam T Numerical type of the matrix
        /// @param cpurt CPU Runtime instance
        /// @param matrix Matrix to compute the trace of
        /// @param use_diag Use the diagonal of the matrix
        /// @return The trace of the matrix
        template <typename T>
        double ComputeTrace(const CPURuntime &cpurt, const Matrix_<T> &matrix, bool use_diag = false);

        /// @brief Compute the trace of a matrix
        /// @param cpurt CPU Runtime instance
        /// @param diag Vector containing the diagonal of the matrix
        /// @return The trace of the matrix
        double ComputeTrace(const CPURuntime &cpurt, const Vector_<double> &diag);

        /// @brief Compute the Frobenius norm of a matrix
        /// @tparam T Numerical type of the matrix
        /// @param mat Matrix to compute the Frobenius norm of
        /// @return The Frobenius norm of the matrix
        template <typename T>
        T FrobeniusNorm(const Tensor<T> &mat);

        /// @brief Compute the Frobenius norm of the difference of two matrices
        /// @tparam T Numerical type of the matrix
        /// @param mat Matrix to compute the Frobenius norm of
        /// @param mat2 Matrix to compute the Frobenius norm of
        /// @return The Frobenius norm of the difference of the two matrices
        template <typename T>
        T FrobeniusNorm(const Tensor<T> &mat, const Tensor<T> &mat2);

        /// @brief Compute the Frobenius norm of the difference of two matrices
        /// @tparam T Numerical type of the matrix
        /// @param cpurt CPU Runtime instance
        /// @param mat Matrix to compute the Frobenius norm of
        /// @param mat2 Matrix to compute the Frobenius norm of
        /// @return The Frobenius norm of the difference of the two matrices
        template <typename T>
        T FrobeniusNorm(const CPURuntime &cpurt, const Tensor<T> &mat, const Tensor<T> &mat2)
        {
            return FrobeniusNorm(mat, mat2);
        };

        /// @brief Compute the Frobenius norm of a matrices
        /// @tparam T Numerical type of the matrix
        /// @param cpurt CPU Runtime instance
        /// @param mat Matrix to compute the Frobenius norm of
        /// @return The Frobenius norm of the matrix
        template <typename T>
        T FrobeniusNorm(const CPURuntime &cpurt, const Tensor<T> &mat);

       
        template <typename T>
        T FrobeniusInnerProduct(const Matrix_<T> &mat1, const Matrix_<T> &mat2);
        template <typename T>
        T FrobeniusInnerProduct(const CPURuntime &rt, const Matrix_<T> &mat1, const Matrix_<T> &mat2);

        /// @brief Get the diagonal of a matrix
        /// @tparam T Numerical type of the matrix
        /// @tparam U Allocator type of the matrix
        /// @param cpurt CPU Runtime instance
        /// @param mat Matrix to get the diagonal of
        /// @param vec Vector to store the diagonal in
        template <typename T, typename U>
        void GetDiagonal(const CPURuntime &cpurt, const Matrix<T, U> &mat, Vector<T, U> &vec)
        {
            vec = mat.get_diagonal();
        };

        /// @brief Set the diagonal of a matrix
        /// @tparam T Numerical type of the matrix
        /// @tparam U Allocator type of the matrix
        /// @param cpurt CPU Runtime instance
        /// @param vec Vector containing the diagonal
        /// @param m Matrix to set the diagonal of
        template <typename T, typename U>
        void SetDiagonal(const CPURuntime &cpurt, const Vector<T, U> &vec, Matrix<T, U> &m)
        {
            m.set_diagonal(vec);
        };

        /// define the function pointer type for the kernel, with 1 input variable
        template <typename T>
        using func_t1D = T (*)(T);

        /// @brief Apply a kernel to a matrix
        /// @tparam T Numerical type of the matrix
        /// @param funcPtr Function pointer to the kernel
        /// @param mat Matrix to apply the kernel to
        template <typename T>
        void ApplyKernel(func_t1D<T> funcPtr, Tensor<T> &mat)
        {
            #pragma omp parallel for collapse(1)
            for (size_t i = 0; i < mat.size(); i++)
            {
                mat.data()[i] = funcPtr(mat.data()[i]);
            }
        };

        /// define the function pointer type for the kernel, with 0 input variables
        template <typename T>
        using func_t0D = T (*)();
        
        /// @brief Apply a kernel to a matrix
        /// @tparam T Numerical type of the matrix
        /// @param funcPtr Function pointer to the kernel
        /// @param mat Matrix to apply the kernel to
        template <typename T>
        void ApplyKernel(func_t0D<T> funcPtr, Tensor<T> &mat)
        {
            #pragma omp parallel for collapse(1)
            for (size_t i = 0; i < mat.size(); i++)
            {
                mat.data()[i] = funcPtr();
            }
        };

        // Applay Kernel function with CPU runtime, which is not used
        template <typename... Args>
        void ApplyKernel(const CPURuntime &rt_, Args &&...args)
        {
            (ApplyKernel(args...));
        }
        template<typename T>
        void HadamardProduct(const Tensor<T> &X, Tensor<T> &Y);

        template<typename T>
        void HadamardProduct(const Tensor<T> &X, const Tensor<T> &Y, Tensor<T> &Z);

        template<typename T>
        void HadamardProduct(const size_t ndim, const T* X, const T* Y, T* Z, bool increment = false);

        template <typename... Args>
        void HadamardProduct(const CPURuntime &rt_, Args &&...args)
        {
            (HadamardProduct(args...));
        } 
        

    } // namespace gpu

}
#endif