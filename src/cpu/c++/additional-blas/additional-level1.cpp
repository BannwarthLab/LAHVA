/// @file additional-level1.cpp
/// @brief CPU C++ implementations of extended/custom Level-1 BLAS operations.
///
/// Implements additional vector and trace operations beyond standard BLAS Level-1,
/// including trace computation with optional OpenMP parallelization. Provides type-specific
/// implementations for double, float, and complex types.

#include "impl/blas/cpu/additional-level1.hpp"
#ifdef W_OPENMP
#include <omp.h>
#endif
#include "../../utils/utils.hpp"
namespace lahva
{
    namespace cpu
    {
        /// @brief
        /// @param cudart
        /// @param matrix
        /// @param use_diag
        /// @return
        template <>
        double ComputeTrace<double>(const CPURuntime &cudart, const Matrix_<double> &matrix, bool use_diag)
        {
            double trace = 0;
#pragma omp parallel for shared(matrix) reduction(+ : trace)
            for (int i = 0; i < matrix.shape().first; i++)
                trace += (double)matrix(i, i);
            return trace;
        };

        template <>
        double ComputeTrace<float>(const CPURuntime &cudart, const Matrix_<float> &matrix, bool use_diag)
        {
            double trace = 0;

#pragma omp parallel for shared(matrix) reduction(+ : trace)
            for (int i = 0; i < matrix.shape().first; i++)
                trace += (double)matrix(i, i);
            return trace;
        };

        double ComputeTrace(const CPURuntime &cudart, const Vector_<double> &diag)
        {
            double trace = 0;

#pragma omp parallel for shared(diag) reduction(+ : trace)
            for (int i = 0; i < diag.size(); i++)
                trace += (double)diag[i];
            return trace;
        };

        template <typename T>
        T FrobeniusNorm(const Tensor<T> &mat)
        {
            T norm = 0;
#pragma omp parallel for reduction(+ : norm)
            for (int i = 0; i < mat.size(); i++)
            {
                norm = fma(mat.data()[i], mat.data()[i], norm);
            };
            norm = std::sqrt(norm);
            return norm;
        }

        template <typename T>
        T FrobeniusNorm(const Tensor<T> &mat, const Tensor<T> &mat2)
        {
            T norm = 0;
#pragma omp parallel for reduction(+ : norm)
            for (int i = 0; i < mat.size(); i++)
            {
                T diff = mat.data()[i] - mat2.data()[i];
                norm = fma(diff, diff, norm);
            };
            norm = std::sqrt(norm);
            return norm;
        }

        template <typename T>
        T FrobeniusNorm(const CPURuntime &rt, const Tensor<T> &mat)
        {
            return FrobeniusNorm(mat);
        };

        template <typename T>
        T FrobeniusInnerProduct(const Matrix_<T> &mat1, const Matrix_<T> &mat2)
        {
            T norm = 0;
            check_equal_size(mat1, mat2);
#pragma omp parallel for shared(mat1, mat2) reduction(+ : norm)
            for (int i = 0; i < (mat1.size()); i++)
            {
                norm += mat1.data()[i] * mat2.data()[i];
            }
            return norm;
        }

        template <typename T>
        T FrobeniusInnerProduct(const CPURuntime &rt, const Matrix_<T> &mat1, const Matrix_<T> &mat2)
        {
            return FrobeniusInnerProduct(mat1, mat2);
        };

        template <typename T>
        void HadamardProduct(const Tensor<T> &A, Tensor<T> &B)
        {
            check_equal_size(A, B);
#pragma omp parallel for shared(A, B)
            for (int i = 0; i < A.size(); i++)
            {
                B.data()[i] = A.data()[i] * B.data()[i];
            }
        }

        template <typename T>
        void HadamardProduct(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C)
        {
            check_equal_size(A, B);
            check_equal_size(A, C);
#pragma omp parallel for shared(A, B, C)
            for (int i = 0; i < A.size(); i++)
            {
                C.data()[i] = A.data()[i] * B.data()[i];
            }
        }

        template <typename T>
        void HadamardProduct(const size_t ndim, const T *A, const T *B, T *C, bool increment)
        {
            if (increment)
            {
                #pragma omp parallel for shared(A, B, C)
                for (int i = 0; i < ndim; i++)
                {
                    C[i] += A[i] * B[i];
                }
                return;
            }
            else
            {
                #pragma omp parallel for shared(A, B, C)
                for (int i = 0; i < ndim; i++)
                {
                    C[i] = A[i] * B[i];
                }
            }
        }

        template float FrobeniusNorm<float>(const Tensor<float> &mat);
        template double FrobeniusNorm<double>(const Tensor<double> &mat);
        template float FrobeniusNorm<float>(const Tensor<float> &mat, const Tensor<float> &mat2);
        template double FrobeniusNorm<double>(const Tensor<double> &mat, const Tensor<double> &mat2);
        template double FrobeniusNorm<double>(const CPURuntime &rt, const Tensor<double> &mat);
        template float FrobeniusNorm<float>(const CPURuntime &rt, const Tensor<float> &mat);
        template float FrobeniusInnerProduct<float>(const Matrix_<float> &mat1, const Matrix_<float> &mat2);
        template double FrobeniusInnerProduct<double>(const Matrix_<double> &mat1, const Matrix_<double> &mat2);
        template double FrobeniusInnerProduct<double>(const CPURuntime &rt, const Matrix_<double> &mat1, const Matrix_<double> &mat2);
        template float FrobeniusInnerProduct<float>(const CPURuntime &rt, const Matrix_<float> &mat1, const Matrix_<float> &mat2);
        template void HadamardProduct(const Tensor<double> &A, Tensor<double> &B);
        template void HadamardProduct(const Tensor<float> &A, Tensor<float> &B);
        template void HadamardProduct(const Tensor<double> &A, const Tensor<double> &B, Tensor<double> &C);
        template void HadamardProduct(const Tensor<float> &A, const Tensor<float> &B, Tensor<float> &C);
        template void HadamardProduct(const size_t ndim, const double *A, const double *B, double *C, bool increment);
        template void HadamardProduct(const size_t ndim, const float *A, const float *B, float *C, bool increment);
    } // namespace cpu

} // namespace lahva
