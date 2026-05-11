#include "impl/blas/cpu/additional-level1.hpp"
#ifdef W_OPENMP
#include <omp.h>
#endif
#include "../../utils/utils.hpp"

namespace lahva
{
    namespace cpu
    {
        /// @brief Compute the trace of a double-precision matrix
        ///
        /// Computes the trace of a matrix (sum of diagonal elements): trace(A) = sum(A_ii).
        /// If use_diag is true, computes the trace from a pre-extracted diagonal vector.
        ///
        /// @param cudart CPU Runtime instance.
        /// @param matrix Input matrix.
        /// @param use_diag If true, compute trace from diagonal; if false, compute from full matrix. Default: false.
        /// @return The trace of the matrix as a double-precision scalar.
        template <>
        double ComputeTrace<double>(const CPURuntime &cudart, const Matrix_<double> &matrix, bool use_diag)
        {
            double trace = 0;
#pragma omp parallel for shared(matrix) reduction(+ : trace)
            for (int i = 0; i < static_cast<int>(matrix.shape().first); i++)
                trace += (double)matrix(i, i);
            return trace;
        };

        /// @brief Compute the trace of a single-precision matrix
        ///
        /// Computes the trace of a matrix (sum of diagonal elements): trace(A) = sum(A_ii).
        /// If use_diag is true, computes the trace from a pre-extracted diagonal vector.
        ///
        /// @param cudart CPU Runtime instance.
        /// @param matrix Input matrix.
        /// @param use_diag If true, compute trace from diagonal; if false, compute from full matrix. Default: false.
        /// @return The trace of the matrix as a double-precision scalar.
        template <>
        double ComputeTrace<float>(const CPURuntime &cudart, const Matrix_<float> &matrix, bool use_diag)
        {
            double trace = 0;

#pragma omp parallel for shared(matrix) reduction(+ : trace)
            for (int i = 0; i < static_cast<int>(matrix.shape().first); i++)
                trace += (double)matrix(i, i);
            return trace;
        };

        /// @brief Compute the trace of a matrix from its diagonal vector
        ///
        /// Computes the trace from a pre-extracted diagonal vector (sum of all elements).
        ///
        /// @param cudart CPU Runtime instance.
        /// @param diag Input vector containing the diagonal elements.
        /// @return The trace (sum of diagonal elements) as a double-precision scalar.
        double ComputeTrace(const CPURuntime &cudart, const Vector_<double> &diag)
        {
            double trace = 0;

#pragma omp parallel for shared(diag) reduction(+ : trace)
            for (int i = 0; i < static_cast<int>(diag.size()); i++)
                trace += (double)diag[i];
            return trace;
        };

        /// @brief Compute the Frobenius norm of a tensor/matrix
        ///
        /// Computes the Frobenius norm: ||A||_F = sqrt(sum(a_ij^2)).
        ///
        /// @tparam T Numerical element type (double, float).
        /// @param mat Input tensor/matrix.
        /// @return The Frobenius norm of the matrix.
        template <typename T>
        T FrobeniusNorm(const Tensor<T> &mat)
        {
            T norm = 0;
#pragma omp parallel for reduction(+ : norm)
            for (int i = 0; i < static_cast<int>(mat.size()); i++)
            {
                norm = fma(mat.data()[i], mat.data()[i], norm);
            };
            norm = std::sqrt(norm);
            return norm;
        }

        /// @brief Compute the Frobenius norm of the difference of two tensors/matrices
        ///
        /// Computes the Frobenius norm of (mat - mat2): ||mat - mat2||_F = sqrt(sum((mat_ij - mat2_ij)^2)).
        ///
        /// @tparam T Numerical element type (double, float).
        /// @param mat First input tensor/matrix.
        /// @param mat2 Second input tensor/matrix.
        /// @return The Frobenius norm of the difference.
        template <typename T>
        T FrobeniusNorm(const Tensor<T> &mat, const Tensor<T> &mat2)
        {
            T norm = 0;
#pragma omp parallel for reduction(+ : norm)
            for (int i = 0; i < static_cast<int>(mat.size()); i++)
            {
                T diff = mat.data()[i] - mat2.data()[i];
                norm = fma(diff, diff, norm);
            };
            norm = std::sqrt(norm);
            return norm;
        }

        /// @brief Runtime-dispatching overload for symmetric multiply (CPURuntime first).
        ///
        /// The runtime parameter is ignored for CPU implementations but this overload
        /// mirrors the signatures used for GPU/CUDA dispatching.
        template <typename T>
        T FrobeniusNorm(const CPURuntime &rt, const Tensor<T> &mat)
        {
            return FrobeniusNorm(mat);
        };

        /// @brief Compute the Frobenius inner product of two matrices
        ///
        /// Computes the Frobenius inner product: <A, B> = sum(a_ij * b_ij) = trace(A^T * B).
        ///
        /// @tparam T Numerical element type (double, float).
        /// @param mat1 First input matrix.
        /// @param mat2 Second input matrix.
        /// @return The Frobenius inner product of the two matrices.
        template <typename T>
        T FrobeniusInnerProduct(const Matrix_<T> &mat1, const Matrix_<T> &mat2)
        {
            T norm = 0;
            check_equal_size(mat1, mat2);
#pragma omp parallel for shared(mat1, mat2) reduction(+ : norm)
            for (int i = 0; i < static_cast<int>(mat1.size()); i++)
            {
                norm += mat1.data()[i] * mat2.data()[i];
            }
            return norm;
        }
        /// @brief Runtime-dispatching overload for symmetric multiply (CPURuntime first).
        ///
        /// The runtime parameter is ignored for CPU implementations but this overload
        /// mirrors the signatures used for GPU/CUDA dispatching.
        template <typename T>
        T FrobeniusInnerProduct(const CPURuntime &rt, const Matrix_<T> &mat1, const Matrix_<T> &mat2)
        {
            return FrobeniusInnerProduct(mat1, mat2);
        };

        /// @brief Compute the Hadamard (element-wise) product of two tensors, storing result in second tensor
        ///
        /// Performs in-place element-wise multiplication: B = A ⊙ B (Hadamard product).
        ///
        /// @tparam T Numerical element type (double, float).
        /// @param A First input tensor (not modified).
        /// @param B Second input tensor (overwritten with the element-wise product).
        template <typename T>
        void HadamardProduct(const Tensor<T> &A, Tensor<T> &B)
        {
            check_equal_size(A, B);
#pragma omp parallel for shared(A, B)
            for (int i = 0; i < static_cast<int>(A.size()); i++)
            {
                B.data()[i] = A.data()[i] * B.data()[i];
            }
        }

        /// @brief Compute the Hadamard (element-wise) product of two tensors, storing result in third tensor
        ///
        /// Performs element-wise multiplication: C = A ⊙ B (Hadamard product).
        ///
        /// @tparam T Numerical element type (double, float).
        /// @param A First input tensor (not modified).
        /// @param B Second input tensor (not modified).
        /// @param C Output tensor to store the element-wise product (overwritten).
        template <typename T>
        void HadamardProduct(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
        {
            check_equal_size(A, B);
            check_equal_size(A, C);
#pragma omp parallel for shared(A, B, C)
            for (int i = 0; i < static_cast<int>(A.size()); i++)
            {
                C.data()[i] = A.data()[i] * B.data()[i];
            }
        }

        /// @brief Compute the Hadamard (element-wise) product of two pointer-based arrays
        ///
        /// Performs element-wise multiplication: C = A ⊙ B (Hadamard product) or C += A ⊙ B if increment is true.
        ///
        /// @tparam T Numerical element type (double, float).
        /// @param ndim Number of elements in arrays A, B, and C.
        /// @param A Pointer to first input array.
        /// @param B Pointer to second input array.
        /// @param C Pointer to output array (overwritten or incremented).
        /// @param increment If true, performs C += A ⊙ B; if false, performs C = A ⊙ B. Default: false.
        template <typename T>
        void HadamardProduct(const size_t ndim, const T *A, const T *B, T *C, bool increment)
        {
            if (increment)
            {
                #pragma omp parallel for shared(A, B, C)
                for (int i = 0; i < static_cast<int>(ndim); i++)
                {
                    C[i] += A[i] * B[i];
                }
                return;
            }
            else
            {
                #pragma omp parallel for shared(A, B, C)
                for (int i = 0; i < static_cast<int>(ndim); i++)
                {
                    C[i] = A[i] * B[i];
                }
            }
        }

        //// Explicit template instantiations
        template float FrobeniusNorm<float>(const Tensor<float> &mat);
        template double FrobeniusNorm<double>(const Tensor<double> &mat);
        template float FrobeniusNorm<float>(const Tensor<float> &mat, const Tensor<float> &mat2);
        template double FrobeniusNorm<double>(const Tensor<double> &mat, const Tensor<double> &mat2);
        template double FrobeniusNorm<double>(const CPURuntime &rt, const Tensor<double> &mat);
        template float FrobeniusNorm<float>(const CPURuntime &rt, const Tensor<float> &mat);
        //
        template float FrobeniusInnerProduct<float>(const Matrix_<float> &mat1, const Matrix_<float> &mat2);
        template double FrobeniusInnerProduct<double>(const Matrix_<double> &mat1, const Matrix_<double> &mat2);
        template double FrobeniusInnerProduct<double>(const CPURuntime &rt, const Matrix_<double> &mat1, const Matrix_<double> &mat2);
        template float FrobeniusInnerProduct<float>(const CPURuntime &rt, const Matrix_<float> &mat1, const Matrix_<float> &mat2);
        //
        template void HadamardProduct(const Tensor<double> &A, Tensor<double> &B);
        template void HadamardProduct(const Tensor<float> &A, Tensor<float> &B);
        template void HadamardProduct(const Tensor<double> &A, const Tensor<double> &B, Tensor<double> &C);
        template void HadamardProduct(const Tensor<float> &A, const Tensor<float> &B, Tensor<float> &C);
        template void HadamardProduct(const size_t ndim, const double *A, const double *B, double *C, bool increment);
        template void HadamardProduct(const size_t ndim, const float *A, const float *B, float *C, bool increment);
    } // namespace cpu

} // namespace lahva
