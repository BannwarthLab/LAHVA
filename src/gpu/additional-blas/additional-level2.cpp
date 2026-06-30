/// @file additional-level2.cpp
/// @brief GPU implementations of additional Level-2 matrix operations.
///
/// Implements GPU versions of additional matrix operations including scaled matrix addition
/// with optional transposition using NVIDIA's cuBLAS library.

#include "linalg.hpp"
#include "runtime.hpp"
#include "impl/blas/gpu/additional-level2.hpp"
#include "../gpu-utils/utils.hpp"
namespace lahva
{
    namespace gpu
    {
        /// @brief Computes scaled matrix addition C = alpha*A + beta*B for double-precision (DGEAM).
        ///
        /// Performs general matrix addition with optional transposition of input matrices.
        /// Supports C = alpha*A + beta*B, C = alpha*A^T + beta*B, and other transpose combinations.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param Ta Transposition flag for matrix A ('N' or 'T').
        /// @param Tb Transposition flag for matrix B ('N' or 'T').
        /// @param alpha Scalar factor for matrix A.
        /// @param a Input double-precision matrix.
        /// @param b Input double-precision matrix.
        /// @param beta Scalar factor for matrix B.
        /// @param c Output double-precision matrix, replaced with result.
        template<>
        void AddMatrices<double>(const CudaRuntime& cudart, const char* Ta, const char* Tb, const double alpha, const Matrix_<double>& a, const Matrix_<double>& b,
        const double beta, Matrix_<double>& c)
        {
            cublasOperation_t transa = get_trans(Ta);
            cublasOperation_t transb = get_trans(Tb);

            check_device_alloc( cudart, a);
            check_device_alloc( cudart, b);
            check_device_alloc( cudart, c);

            int m, n;
            std::tie(m, n) = check_same_shape_mm(a, b, c, transa, transb);

            size_t lda = get_leading(m, n, transa);
            size_t ldb = get_leading(m, n, transb);
            size_t ldc = get_leading(m, n);

            cudart.cublasSetStream_();
            get_cublas_error(cublasDgeam(cudart.handle, transa, transb, m, n, &alpha, a.gpu_data(), lda, &beta, b.gpu_data(), ldb, c.gpu_data(), ldc));
        };

        /// @brief Computes scaled matrix addition C = alpha*A + beta*B for single-precision (SGEAM).
        ///
        /// Performs general matrix addition with optional transposition of input matrices.
        /// Supports C = alpha*A + beta*B, C = alpha*A^T + beta*B, and other transpose combinations.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param Ta Transposition flag for matrix A ('N' or 'T').
        /// @param Tb Transposition flag for matrix B ('N' or 'T').
        /// @param alpha Scalar factor for matrix A.
        /// @param a Input single-precision matrix.
        /// @param b Input single-precision matrix.
        /// @param beta Scalar factor for matrix B.
        /// @param c Output single-precision matrix, replaced with result.
        template<>
        void AddMatrices<float>(const CudaRuntime& cudart, const char* Ta, const char* Tb, const float alpha, const Matrix_<float>& a, const Matrix_<float>& b,
        const float beta, Matrix_<float>& c)
        {
            cublasOperation_t transa = get_trans(Ta);
            cublasOperation_t transb = get_trans(Tb);

            check_device_alloc( cudart, a);
            check_device_alloc( cudart, b);
            check_device_alloc( cudart, c);

            int m, n;
            std::tie(m, n) = check_same_shape_mm(a, b, c, transa, transb);

            size_t lda = get_leading(m, n, transa);
            size_t ldb = get_leading(m, n, transb);
            size_t ldc = get_leading(m, n);

            cudart.cublasSetStream_();
            get_cublas_error(cublasSgeam(cudart.handle, transa, transb, m, n, &alpha, a.gpu_data(), lda, &beta, b.gpu_data(), ldb, c.gpu_data(), ldc));
        };

        /// @brief Computes scaled matrix addition with default parameters for double-precision (DGEAM).
        ///
        /// Performs general matrix addition with matrices passed in different argument order.
        /// Parameters have defaults: Ta defaults to "N", Tb defaults to "N".
        ///
        /// @param cudart CUDA runtime instance.
        /// @param a Input double-precision matrix.
        /// @param b Input double-precision matrix.
        /// @param c Output double-precision matrix, replaced with result.
        /// @param alpha Scalar factor for matrix A.
        /// @param beta Scalar factor for matrix B.
        /// @param Ta Transposition flag for matrix A ('N' or 'T').
        /// @param Tb Transposition flag for matrix B ('N' or 'T').
        template<>
        void AddMatrices<double>(const CudaRuntime& cudart, const Matrix_<double>& a, const Matrix_<double>& b, Matrix_<double>& c,
        const double alpha, const double beta, const char* Ta, const char* Tb)
        {
            cublasOperation_t transa = get_trans(Ta);
            cublasOperation_t transb = get_trans(Tb);

            check_device_alloc( cudart, a);
            check_device_alloc( cudart, b);
            check_device_alloc( cudart, c);

            int m, n;
            std::tie(m, n) = check_same_shape_mm(a, b, c, transa, transb);

            size_t lda = get_leading(m, n, transa);
            size_t ldb = get_leading(m, n, transb);
            size_t ldc = get_leading(m, n);

            cudart.cublasSetStream_();
            get_cublas_error(cublasDgeam(cudart.handle, transa, transb, m, n, &alpha, a.gpu_data(), lda, &beta, b.gpu_data(), ldb, c.gpu_data(), ldc));
        };

        /// @brief Computes scaled matrix addition with default parameters for single-precision (SGEAM).
        ///
        /// Performs general matrix addition with matrices passed in different argument order.
        /// Parameters have defaults: Ta defaults to "N", Tb defaults to "N".
        ///
        /// @param cudart CUDA runtime instance.
        /// @param a Input single-precision matrix.
        /// @param b Input single-precision matrix.
        /// @param c Output single-precision matrix, replaced with result.
        /// @param alpha Scalar factor for matrix A.
        /// @param beta Scalar factor for matrix B.
        /// @param Ta Transposition flag for matrix A ('N' or 'T').
        /// @param Tb Transposition flag for matrix B ('N' or 'T').
        template<>
        void AddMatrices<float>(const CudaRuntime& cudart, const Matrix_<float>& a, const Matrix_<float>& b, Matrix_<float>& c,
        const float alpha , const float beta, const char* Ta, const char* Tb)
        {
            cublasOperation_t transa = get_trans(Ta);
            cublasOperation_t transb = get_trans(Tb);

            check_device_alloc( cudart, a);
            check_device_alloc( cudart, b);
            check_device_alloc( cudart, c);

            int m, n;
            std::tie(m, n) = check_same_shape_mm(a, b, c, transa, transb);

            size_t lda = get_leading(m, n, transa);
            size_t ldb = get_leading(m, n, transb);
            size_t ldc = get_leading(m, n);

            cudart.cublasSetStream_();
            get_cublas_error(cublasSgeam(cudart.handle, transa, transb, m, n, &alpha, a.gpu_data(), lda, &beta, b.gpu_data(), ldb, c.gpu_data(), ldc));
        };
    }
}
