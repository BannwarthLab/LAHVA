/// @file additional-level1.hpp
/// @brief C++-style additional Level-1 operations beyond standard BLAS.
///
// Additional Level-1 BLAS-like operations (vector * vector) - C++ template declarations
// These functions are C++-style wrappers providing additional linear algebra utilities
// beyond standard BLAS Level-1 operations. These wrappers operate on Tensor objects and 
// dispatch to optimized CPU implementations.
// Each overload is provided for double and float precision; both precisions are documented explicitly.

#ifndef LAHVA_ADD_LEVEL1_CPU_HPP
#define LAHVA_ADD_LEVEL1_CPU_HPP
#include "impl/tensor/cpu/tensor.hpp"
#include "impl/tensor/cpu/vector.hpp"
#include "impl/tensor/cpu/matrix.hpp"
#include "impl/tensor/cpu/lowtrimatrix.hpp"
#include "const.h"
#include "level1.h"

namespace lahva
{
    namespace cpu
    {
        /// @brief Compute the trace of a matrix
        ///
        /// Computes the trace of a matrix (sum of diagonal elements): trace(A) = sum(A_ii).
        /// If use_diag is true, computes the trace from a pre-extracted diagonal vector.
        ///
        /// @tparam T Numerical element type (double, float).
        /// @param cpurt CPU Runtime instance.
        /// @param matrix Input matrix.
        /// @param use_diag If true, compute trace from diagonal; if false, compute from full matrix. Default: false.
        /// @return The trace of the matrix as a double-precision scalar.
        template <typename T>
        double ComputeTrace(const CPURuntime &cpurt, const Matrix_<T> &matrix, bool use_diag = false);

        /// @brief Compute the trace of a matrix from its diagonal
        ///
        /// Computes the trace from a pre-extracted diagonal vector (sum of all elements).
        ///
        /// @param cpurt CPU Runtime instance.
        /// @param diag Input vector containing the diagonal elements.
        /// @return The trace (sum of diagonal elements) as a double-precision scalar.
        double ComputeTrace(const CPURuntime &cpurt, const Vector_<double> &diag);

        /// @brief Compute the Frobenius norm of a matrix
        ///
        /// Computes the Frobenius norm: ||A||_F = sqrt(sum(a_ij^2)).
        ///
        /// @tparam T Numerical element type (double, float).
        /// @param mat Input tensor/matrix.
        /// @return The Frobenius norm of the matrix.
        template <typename T>
        T FrobeniusNorm(const Tensor<T> &mat);

        /// @brief Compute the Frobenius norm of the difference of two matrices
        ///
        /// Computes the Frobenius norm of (mat - mat2): ||mat - mat2||_F = sqrt(sum((mat_ij - mat2_ij)^2)).
        ///
        /// @tparam T Numerical element type (double, float).
        /// @param mat First input tensor/matrix.
        /// @param mat2 Second input tensor/matrix.
        /// @return The Frobenius norm of the difference.
        template <typename T>
        T FrobeniusNorm(const Tensor<T> &mat, const Tensor<T> &mat2);

        /// @brief Runtime-dispatching overload for Frobenius norm difference (CPURuntime first)
        ///
        /// The runtime parameter is ignored for CPU implementations but this overload
        /// mirrors the signatures used for GPU/CUDA dispatching.
        ///
        /// @tparam T Numerical element type (double, float).
        /// @param cpurt CPU Runtime instance (ignored for CPU).
        /// @param mat First input tensor/matrix.
        /// @param mat2 Second input tensor/matrix.
        /// @return The Frobenius norm of the difference.
        template <typename T>
        T FrobeniusNorm(const CPURuntime &cpurt, const Tensor<T> &mat, const Tensor<T> &mat2)
        {
            return FrobeniusNorm(mat, mat2);
        };

        /// @brief Compute the Frobenius norm of a matrix with runtime dispatch
        ///
        /// Computes the Frobenius norm: ||mat||_F = sqrt(sum(mat_ij^2)).
        /// The runtime parameter is ignored for CPU implementations but this overload
        /// mirrors the signatures used for GPU/CUDA dispatching.
        ///
        /// @tparam T Numerical element type (double, float).
        /// @param cpurt CPU Runtime instance (ignored for CPU).
        /// @param mat Input tensor/matrix.
        /// @return The Frobenius norm of the matrix.
        template <typename T>
        T FrobeniusNorm(const CPURuntime &cpurt, const Tensor<T> &mat);

        /// @brief Compute the Frobenius inner product of two matrices
        ///
        /// Computes the Frobenius inner product (Frobenius dot product): mat1:mat2 = sum(mat1_ij * mat2_ij).
        ///
        /// @tparam T Numerical element type (double, float).
        /// @param mat1 First input matrix.
        /// @param mat2 Second input matrix.
        /// @return The Frobenius inner product as an element of type T.
        template <typename T>
        T FrobeniusInnerProduct(const Matrix_<T> &mat1, const Matrix_<T> &mat2);

        /// @brief Runtime-dispatching overload for Frobenius inner product (CPURuntime first)
        ///
        /// The runtime parameter is ignored for CPU implementations but this overload
        /// mirrors the signatures used for GPU/CUDA dispatching.
        ///
        /// @tparam T Numerical element type (double, float).
        /// @param rt CPU Runtime instance (ignored for CPU).
        /// @param mat1 First input matrix.
        /// @param mat2 Second input matrix.
        /// @return The Frobenius inner product as an element of type T.
        template <typename T>
        T FrobeniusInnerProduct(const CPURuntime &rt, const Matrix_<T> &mat1, const Matrix_<T> &mat2);

        /// @brief Extract the diagonal of a matrix
        ///
        /// Extracts the main diagonal of the input matrix into a vector.
        ///
        /// @tparam T Numerical element type (double, float).
        /// @tparam U Allocator type for the matrix and vector.
        /// @param cpurt CPU Runtime instance.
        /// @param mat Input matrix to extract diagonal from.
        /// @param vec Output vector to store the diagonal elements (overwritten).
        template <typename T, typename U>
        void GetDiagonal(const CPURuntime &cpurt, const Matrix<T, U> &mat, Vector<T, U> &vec)
        {
            vec = mat.get_diagonal();
        };

        /// @brief Set the diagonal of a matrix
        ///
        /// Sets the main diagonal of the matrix to the values in the input vector.
        ///
        /// @tparam T Numerical element type (double, float).
        /// @tparam U Allocator type for the matrix and vector.
        /// @param cpurt CPU Runtime instance.
        /// @param vec Input vector containing the new diagonal values.
        /// @param m Output matrix with diagonal set (overwritten).
        template <typename T, typename U>
        void SetDiagonal(const CPURuntime &cpurt, const Vector<T, U> &vec, Matrix<T, U> &m)
        {
            m.set_diagonal(vec);
        };

        /// Function pointer type for element-wise kernels with 1 input variable: T -> T.
        template <typename T>
        using func_t1D = T (*)(T);

        /// @brief Apply an element-wise kernel to all elements of a tensor
        ///
        /// Applies a user-provided function to each element of the tensor in parallel using OpenMP.
        /// Performs: mat[i] = funcPtr(mat[i]) for all elements, parallelized with #pragma omp parallel for.
        ///
        /// @tparam T Numerical element type (double, float).
        /// @param funcPtr Function pointer of type T (*)(T) to apply to each element.
        /// @param mat Input tensor (overwritten with transformed values).
        template <typename T>
        void ApplyKernel(func_t1D<T> funcPtr, Tensor<T> &mat)
        {
#pragma omp parallel for collapse(1)
            for (size_t i = 0; i < mat.size(); i++)
            {
                mat.data()[i] = funcPtr(mat.data()[i]);
            }
        };

        /// Function pointer type for element-wise kernels with 0 input variables: () -> T.
        template <typename T>
        using func_t0D = T (*)();

        /// @brief Apply a parameterless element-wise kernel to all elements of a tensor
        ///
        /// Applies a parameterless function to each element of the tensor in parallel using OpenMP.
        /// Performs: mat[i] = funcPtr() for all elements, parallelized with #pragma omp parallel for.
        /// Useful for initializing tensors or applying constant transformations.
        ///
        /// @tparam T Numerical element type (double, float).
        /// @param funcPtr Function pointer of type T (*)(void) to call for each element.
        /// @param mat Input tensor (overwritten with generated values).
        template <typename T>
        void ApplyKernel(func_t0D<T> funcPtr, Tensor<T> &mat)
        {
#pragma omp parallel for collapse(1)
            for (size_t i = 0; i < mat.size(); i++)
            {
                mat.data()[i] = funcPtr();
            }
        };

        /// @brief Compute the Hadamard product in-place: Y = X .* Y
        ///
        /// Performs element-wise multiplication of X and Y, storing the result back in Y.
        /// Operation: Y[i] = X[i] * Y[i] for all elements i.
        ///
        /// @tparam T Numerical element type (double, float).
        /// @param X First input tensor (read-only).
        /// @param Y Second input tensor, overwritten with the element-wise product.
        template <typename T>
        void HadamardProduct(const Tensor<T> &X, Tensor<T> &Y);

        /// @brief Compute the Hadamard product: Z = X .* Y
        ///
        /// Performs element-wise multiplication of X and Y, storing the result in Z.
        /// Operation: Z[i] = X[i] * Y[i] for all elements i.
        ///
        /// @tparam T Numerical element type (double, float).
        /// @param X First input tensor (read-only).
        /// @param Y Second input tensor (read-only).
        /// @param Z Output tensor to store the element-wise product (overwritten).
        template <typename T>
        void HadamardProduct(const Tensor<T> &X, const Tensor<T> &Y, Tensor<T> &Z);

        /// @brief Compute the Hadamard product with raw pointers: Z = X .* Y
        ///
        /// Performs element-wise multiplication of two arrays with optional accumulation.
        /// Operation: Z[i] = X[i] * Y[i] for all i (or Z[i] += X[i] * Y[i] if increment=true).
        ///
        /// @tparam T Numerical element type (double, float).
        /// @param ndim Number of elements in each array.
        /// @param X Pointer to first input array (read-only).
        /// @param Y Pointer to second input array (read-only).
        /// @param Z Pointer to output array (overwritten or accumulated into).
        /// @param increment If true, accumulate into Z (Z += X.*Y); if false, assign (Z = X.*Y). Default: false.
        template <typename T>
        void HadamardProduct(const size_t ndim, const T *X, const T *Y, T *Z, bool increment = false);

        /// @brief Runtime-dispatching overload that allows passing a CPURuntime first.
        ///
        /// The runtime parameter is ignored for CPU implementations but this overload
        /// mirrors the signatures used for GPU/CUDA dispatching.
        template <typename... Args>
        void ApplyKernel(const CPURuntime &rt_, Args &&...args)
        {
            (ApplyKernel(args...));
        }

        /// @brief Runtime-dispatching overload that allows passing a CPURuntime first.
        ///
        /// The runtime parameter is ignored for CPU implementations but this overload
        /// mirrors the signatures used for GPU/CUDA dispatching.
        template <typename... Args>
        void HadamardProduct(const CPURuntime &rt_, Args &&...args)
        {
            (HadamardProduct(args...));
        }

    } // namespace gpu

}
#endif