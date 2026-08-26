/// @file level3.cpp
/// @brief GPU-accelerated Level-3 BLAS operations implementation.
///
/// Implements GPU versions of Level-3 BLAS operations (matrix-matrix operations)
/// using NVIDIA's cuBLAS library. Supports matrix multiplication and symmetric matrix operations.

#include "impl/gpu/utils.hpp"
#include "impl/blas/gpu/level3.hpp"
#include "linalg.hpp"
#include "runtime.hpp"

namespace lahva
{
    namespace gpu
    {

        /// @brief Computes matrix-matrix product \f$\mathbf{C}=alpha*\mathbf{A}*\mathbf{B}+beta*\mathbf{C}\f$ for double-precision (DGEMM).
        ///
        /// Performs general matrix multiplication with optional transposition of input matrices.
        /// Computes C = alpha*A*B + beta*C, supporting A^T, B^T, or conjugate transpose operations.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param Ta Transposition flag for matrix A ('N' or 'T').
        /// @param Tb Transposition flag for matrix B ('N' or 'T').
        /// @param alpha Scalar factor for the matrix product.
        /// @param a Input double-precision matrix.
        /// @param b Input double-precision matrix.
        /// @param beta Scalar factor for matrix C.
        /// @param c Input/output double-precision matrix, replaced with result.
        void MatrixMatrixProduct(const CudaRuntime &cudart, const char *Ta, const char *Tb, const double alpha, const Matrix_<double> &a,
                                 const Matrix_<double> &b, const double beta, Matrix_<double> &c)
        {
            cublasOperation_t transa = get_trans(Ta);
            cublasOperation_t transb = get_trans(Tb);

            check_device_alloc(cudart, a);
            check_device_alloc(cudart, b);
            check_device_alloc(cudart, c);

            int m, n, k;
            std::tie(m, n, k) = check_size_mm(a, b, c, transa, transb);

            size_t lda = get_leading(m, k);
            size_t ldb = get_leading(k, n);
            size_t ldc = get_leading(m, n);

            cudart.cublasSetStream_();
            cublasStatus_t istat = cublasDgemm(cudart.handle, transa, transb, m, n, k, &alpha, a.gpu_data(), lda, b.gpu_data(), ldb, &beta, c.gpu_data(), ldc);
            get_cublas_error(istat);
        };

        /// @brief Computes matrix-matrix product with default parameters for double-precision (DGEMM).
        ///
        /// Performs general matrix multiplication with matrices passed in different argument order.
        /// Parameters have defaults: Ta defaults to "N", Tb defaults to "N", alpha defaults to 1.0, beta defaults to 0.0.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param a Input double-precision matrix.
        /// @param b Input double-precision matrix.
        /// @param c Input/output double-precision matrix, replaced with result.
        /// @param alpha Scalar factor for the matrix product.
        /// @param beta Scalar factor for matrix C.
        /// @param Ta Transposition flag for matrix A ('N' or 'T').
        /// @param Tb Transposition flag for matrix B ('N' or 'T').
        void MatrixMatrixProduct(const CudaRuntime &cudart, const Matrix_<double> &a, const Matrix_<double> &b, Matrix_<double> &c,
                                 const double alpha, const double beta, const char *Ta, const char *Tb)
        {
            cublasOperation_t transa = get_trans(Ta);
            cublasOperation_t transb = get_trans(Tb);

            check_device_alloc(cudart, a);
            check_device_alloc(cudart, b);
            check_device_alloc(cudart, c);

            int m, n, k;
            std::tie(m, n, k) = check_size_mm(a, b, c, transa, transb);

            size_t lda = get_leading(m, k);
            size_t ldb = get_leading(k, n);
            size_t ldc = get_leading(m, n);

            cudart.cublasSetStream_();
            cublasStatus_t istat = cublasDgemm(cudart.handle, transa, transb, m, n, k, &alpha, a.gpu_data(), lda, b.gpu_data(), ldb, &beta, c.gpu_data(), ldc);
            get_cublas_error(istat);
        };

        /// @brief Computes matrix-matrix product \f$\mathbf{C}=alpha*\mathbf{A}*\mathbf{B}+beta*\mathbf{C}\f$ for single-precision (SGEMM).
        ///
        /// Performs general matrix multiplication with optional transposition of input matrices.
        /// Computes C = alpha*A*B + beta*C, supporting A^T, B^T, or conjugate transpose operations.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param Ta Transposition flag for matrix A ('N' or 'T').
        /// @param Tb Transposition flag for matrix B ('N' or 'T').
        /// @param alpha Scalar factor for the matrix product.
        /// @param a Input single-precision matrix.
        /// @param b Input single-precision matrix.
        /// @param beta Scalar factor for matrix C.
        /// @param c Input/output single-precision matrix, replaced with result.
        void MatrixMatrixProduct(const CudaRuntime &cudart, const char *Ta, const char *Tb, const float alpha, const Matrix_<float> &a, const Matrix_<float> &b,
                                 const float beta, Matrix_<float> &c)
        {
            cublasOperation_t transa = get_trans(Ta);
            cublasOperation_t transb = get_trans(Tb);

            check_device_alloc(cudart, a);
            check_device_alloc(cudart, b);
            check_device_alloc(cudart, c);

            int m, n, k;
            std::tie(m, n, k) = check_size_mm(a, b, c, transa, transb);

            size_t lda = get_leading(m, k);
            size_t ldb = get_leading(k, n);
            size_t ldc = get_leading(m, n);

            cudart.cublasSetStream_();
            cublasStatus_t istat = cublasSgemm(cudart.handle, transa, transb, m, n, k, &alpha, a.gpu_data(), lda, b.gpu_data(), ldb, &beta, c.gpu_data(), ldc);
            get_cublas_error(istat);
        };

        /// @brief Computes matrix-matrix product with default parameters for single-precision (SGEMM).
        ///
        /// Performs general matrix multiplication with matrices passed in different argument order.
        /// Parameters have defaults: Ta defaults to "N", Tb defaults to "N", alpha defaults to 1.0, beta defaults to 0.0.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param a Input single-precision matrix.
        /// @param b Input single-precision matrix.
        /// @param c Input/output single-precision matrix, replaced with result.
        /// @param alpha Scalar factor for the matrix product.
        /// @param beta Scalar factor for matrix C.
        /// @param Ta Transposition flag for matrix A ('N' or 'T').
        /// @param Tb Transposition flag for matrix B ('N' or 'T').
        void MatrixMatrixProduct(const CudaRuntime &cudart, const Matrix_<float> &a, const Matrix_<float> &b, Matrix_<float> &c,
                                 const float alpha, const float beta, const char *Ta, const char *Tb)
        {
            cublasOperation_t transa = get_trans(Ta);
            cublasOperation_t transb = get_trans(Tb);

            check_device_alloc(cudart, a);
            check_device_alloc(cudart, b);
            check_device_alloc(cudart, c);

            int m, n, k;
            std::tie(m, n, k) = check_size_mm(a, b, c, transa, transb);

            size_t lda = get_leading(m, k);
            size_t ldb = get_leading(k, n);
            size_t ldc = get_leading(m, n);

            cudart.cublasSetStream_();
            cublasStatus_t istat = cublasSgemm(cudart.handle, transa, transb, m, n, k, &alpha, a.gpu_data(), lda, b.gpu_data(), ldb, &beta, c.gpu_data(), ldc);
            get_cublas_error(istat);
        };

        /// @brief Computes matrix-matrix product \f$\mathbf{C}=alpha*\mathbf{A}*\mathbf{B}+beta*\mathbf{C}\f$ for complex double-precision (ZGEMM).
        ///
        /// Performs general matrix multiplication with complex double-precision matrices and optional transposition.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param Ta Transposition flag for matrix A ('N' or 'T').
        /// @param Tb Transposition flag for matrix B ('N' or 'T').
        /// @param alpha Scalar factor for the matrix product.
        /// @param a Input complex double-precision matrix.
        /// @param b Input complex double-precision matrix.
        /// @param beta Scalar factor for matrix C.
        /// @param c Input/output complex double-precision matrix, replaced with result.
        void MatrixMatrixProduct(const CudaRuntime &cudart, const char *Ta, const char *Tb, const complex_double alpha, const Matrix_<complex_double> &a,
                                 const Matrix_<complex_double> &b, const complex_double beta, Matrix_<complex_double> &c)
        {
            cublasOperation_t transa = get_trans(Ta);
            cublasOperation_t transb = get_trans(Tb);

            check_device_alloc(cudart, a);
            check_device_alloc(cudart, b);
            check_device_alloc(cudart, c);

            int m, n, k;
            std::tie(m, n, k) = check_size_mm(a, b, c, transa, transb);

            size_t lda = get_leading(m, k);
            size_t ldb = get_leading(k, n);
            size_t ldc = get_leading(m, n);

            cudart.cublasSetStream_();
            get_cublas_error(cublasZgemm(cudart.handle, transa, transb, m, n, k, reinterpret_cast<const cuDoubleComplex *>(&alpha),
                                         reinterpret_cast<const cuDoubleComplex *>(a.gpu_data()), lda, reinterpret_cast<const cuDoubleComplex *>(b.gpu_data()), ldb,
                                         reinterpret_cast<const cuDoubleComplex *>(&beta), reinterpret_cast<cuDoubleComplex *>(c.gpu_data()), ldc));
        };

        /// @brief Computes matrix-matrix product with default parameters for complex double-precision (ZGEMM).
        ///
        /// Performs general matrix multiplication with complex double-precision matrices with default parameters.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param a Input complex double-precision matrix.
        /// @param b Input complex double-precision matrix.
        /// @param c Input/output complex double-precision matrix, replaced with result.
        /// @param alpha Scalar factor for the matrix product.
        /// @param beta Scalar factor for matrix C.
        /// @param Ta Transposition flag for matrix A ('N' or 'T').
        /// @param Tb Transposition flag for matrix B ('N' or 'T').
        void MatrixMatrixProduct(const CudaRuntime &cudart, const Matrix_<complex_double> &a, const Matrix_<complex_double> &b, Matrix_<complex_double> &c,
                                 const complex_double alpha, const complex_double beta, const char *Ta, const char *Tb)
        {
            cublasOperation_t transa = get_trans(Ta);
            cublasOperation_t transb = get_trans(Tb);

            check_device_alloc(cudart, a);
            check_device_alloc(cudart, b);
            check_device_alloc(cudart, c);

            int m, n, k;
            std::tie(m, n, k) = check_size_mm(a, b, c, transa, transb);

            size_t lda = get_leading(m, k);
            size_t ldb = get_leading(k, n);
            size_t ldc = get_leading(m, n);

            cudart.cublasSetStream_();
            get_cublas_error(cublasZgemm(cudart.handle, transa, transb, m, n, k, reinterpret_cast<const cuDoubleComplex *>(&alpha),
                                         reinterpret_cast<const cuDoubleComplex *>(a.gpu_data()), lda, reinterpret_cast<const cuDoubleComplex *>(b.gpu_data()), ldb,
                                         reinterpret_cast<const cuDoubleComplex *>(&beta), reinterpret_cast<cuDoubleComplex *>(c.gpu_data()), ldc));
        };

        /// @brief Computes matrix-matrix product \f$\mathbf{C}=alpha*\mathbf{A}*\mathbf{B}+beta*\mathbf{C}\f$ for complex single-precision (CGEMM).
        ///
        /// Performs general matrix multiplication with complex single-precision matrices and optional transposition.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param Ta Transposition flag for matrix A ('N' or 'T').
        /// @param Tb Transposition flag for matrix B ('N' or 'T').
        /// @param alpha Scalar factor for the matrix product.
        /// @param a Input complex single-precision matrix.
        /// @param b Input complex single-precision matrix.
        /// @param beta Scalar factor for matrix C.
        /// @param c Input/output complex single-precision matrix, replaced with result.
        void MatrixMatrixProduct(const CudaRuntime &cudart, const char *Ta, const char *Tb, const complex_float alpha, const Matrix_<complex_float> &a,
                                 const Matrix_<complex_float> &b, const complex_float beta, Matrix_<complex_float> &c)
        {
            cublasOperation_t transa = get_trans(Ta);
            cublasOperation_t transb = get_trans(Tb);

            check_device_alloc(cudart, a);
            check_device_alloc(cudart, b);
            check_device_alloc(cudart, c);

            int m, n, k;
            std::tie(m, n, k) = check_size_mm(a, b, c, transa, transb);

            size_t lda = get_leading(m, k);
            size_t ldb = get_leading(k, n);
            size_t ldc = get_leading(m, n);

            cudart.cublasSetStream_();
            get_cublas_error(cublasCgemm(cudart.handle, transa, transb, m, n, k, reinterpret_cast<const cuComplex *>(&alpha),
                                         reinterpret_cast<const cuComplex *>(a.gpu_data()), lda, reinterpret_cast<const cuComplex *>(b.gpu_data()), ldb,
                                         reinterpret_cast<const cuComplex *>(&beta), reinterpret_cast<cuComplex *>(c.gpu_data()), ldc));
        };

        /// @brief Computes matrix-matrix product with default parameters for complex single-precision (CGEMM).
        ///
        /// Performs general matrix multiplication with complex single-precision matrices with default parameters.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param a Input complex single-precision matrix.
        /// @param b Input complex single-precision matrix.
        /// @param c Input/output complex single-precision matrix, replaced with result.
        /// @param alpha Scalar factor for the matrix product.
        /// @param beta Scalar factor for matrix C.
        /// @param Ta Transposition flag for matrix A ('N' or 'T').
        /// @param Tb Transposition flag for matrix B ('N' or 'T').
        void MatrixMatrixProduct(const CudaRuntime &cudart, const Matrix_<complex_float> &a, const Matrix_<complex_float> &b, Matrix_<complex_float> &c,
                                 const complex_float alpha, const complex_float beta, const char *Ta, const char *Tb)
        {
            cublasOperation_t transa = get_trans(Ta);
            cublasOperation_t transb = get_trans(Tb);

            check_device_alloc(cudart, a);
            check_device_alloc(cudart, b);
            check_device_alloc(cudart, c);

            int m, n, k;
            std::tie(m, n, k) = check_size_mm(a, b, c, transa, transb);

            size_t lda = get_leading(m, k);
            size_t ldb = get_leading(k, n);
            size_t ldc = get_leading(m, n);

            cudart.cublasSetStream_();
            get_cublas_error(cublasCgemm(cudart.handle, transa, transb, m, n, k, reinterpret_cast<const cuComplex *>(&alpha),
                                         reinterpret_cast<const cuComplex *>(a.gpu_data()), lda, reinterpret_cast<const cuComplex *>(b.gpu_data()), ldb,
                                         reinterpret_cast<const cuComplex *>(&beta), reinterpret_cast<cuComplex *>(c.gpu_data()), ldc));
        };

        /// @brief Computes symmetric matrix-matrix product \f$\mathbf{C}=alpha*\mathbf{A}*\mathbf{B}+beta*\mathbf{C}\f$ for double-precision (DSYMM).
        ///
        /// Performs symmetric matrix multiplication where matrix A is symmetric.
        /// A is either on the left or right side depending on the side parameter.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param side Side mode: CUBLAS_SIDE_LEFT (A on left) or CUBLAS_SIDE_RIGHT (A on right).
        /// @param alpha Scalar factor for the matrix product.
        /// @param a Input double-precision symmetric matrix.
        /// @param b Input double-precision matrix.
        /// @param beta Scalar factor for matrix C.
        /// @param c Input/output double-precision matrix, replaced with result.
        void SymMatrixMatrixProduct(const CudaRuntime &cudart, const cublasSideMode_t side, const double alpha, const Matrix_<double> &a, const Matrix_<double> &b,
                                    const double beta, Matrix_<double> &c)
        {
            int m, n, k;
            std::tie(m, n, k) = check_size_mm(a, b, c);

            size_t lda = get_leading(m, k);
            size_t ldb = get_leading(k, n);
            size_t ldc = get_leading(m, n);

            check_device_alloc(cudart, a);
            check_device_alloc(cudart, b);
            check_device_alloc(cudart, c);

            cudart.cublasSetStream_();
            cublasStatus_t istat = cublasDsymm(cudart.handle, side, tri_gpu, m, n, &alpha, a.gpu_data(), lda, b.gpu_data(), ldb, &beta, c.gpu_data(), ldc);
            get_cublas_error(istat);
        };

        /// @brief Computes symmetric matrix-matrix product with default parameters for double-precision (DSYMM).
        ///
        /// Performs symmetric matrix multiplication with matrices passed in different argument order.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param a Input double-precision symmetric matrix.
        /// @param b Input double-precision matrix.
        /// @param c Input/output double-precision matrix, replaced with result.
        /// @param alpha Scalar factor for the matrix product.
        /// @param beta Scalar factor for matrix C.
        /// @param side Side mode: CUBLAS_SIDE_LEFT (A on left) or CUBLAS_SIDE_RIGHT (A on right).
        void SymMatrixMatrixProduct(const CudaRuntime &cudart, const Matrix_<double> &a, const Matrix_<double> &b, Matrix_<double> &c,
                                    const double alpha, const double beta, const cublasSideMode_t side)
        {
            int m, n, k;
            std::tie(m, n, k) = check_size_mm(a, b, c);

            size_t lda = get_leading(m, k);
            size_t ldb = get_leading(k, n);
            size_t ldc = get_leading(m, n);

            check_device_alloc(cudart, a);
            check_device_alloc(cudart, b);
            check_device_alloc(cudart, c);

            cudart.cublasSetStream_();
            cublasStatus_t istat = cublasDsymm(cudart.handle, side, tri_gpu, m, n, &alpha, a.gpu_data(), lda, b.gpu_data(), ldb, &beta, c.gpu_data(), ldc);
            get_cublas_error(istat);
        };

        /// @brief Computes symmetric matrix-matrix product \f$\mathbf{C}=alpha*\mathbf{A}*\mathbf{B}+beta*\mathbf{C}\f$ for single-precision (SSYMM).
        ///
        /// Performs symmetric matrix multiplication where matrix A is symmetric.
        /// A is either on the left or right side depending on the side parameter.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param side Side mode: CUBLAS_SIDE_LEFT (A on left) or CUBLAS_SIDE_RIGHT (A on right).
        /// @param alpha Scalar factor for the matrix product.
        /// @param a Input single-precision symmetric matrix.
        /// @param b Input single-precision matrix.
        /// @param beta Scalar factor for matrix C.
        /// @param c Input/output single-precision matrix, replaced with result.
        void SymMatrixMatrixProduct(const CudaRuntime &cudart, const cublasSideMode_t side, const float alpha, const Matrix_<float> &a, const Matrix_<float> &b,
                                    const float beta, Matrix_<float> &c)
        {
            int m, n, k;
            std::tie(m, n, k) = check_size_mm(a, b, c);

            size_t lda = get_leading(m, k);
            size_t ldb = get_leading(k, n);
            size_t ldc = get_leading(m, n);

            check_device_alloc(cudart, a);
            check_device_alloc(cudart, b);
            check_device_alloc(cudart, c);

            cudart.cublasSetStream_();
            cublasStatus_t istat = cublasSsymm(cudart.handle, side, tri_gpu, m, n, &alpha, a.gpu_data(), lda, b.gpu_data(), ldb, &beta, c.gpu_data(), ldc);
            get_cublas_error(istat);
        };

        /// @brief Computes symmetric matrix-matrix product with default parameters for single-precision (SSYMM).
        ///
        /// Performs symmetric matrix multiplication with matrices passed in different argument order.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param a Input single-precision symmetric matrix.
        /// @param b Input single-precision matrix.
        /// @param c Input/output single-precision matrix, replaced with result.
        /// @param alpha Scalar factor for the matrix product.
        /// @param beta Scalar factor for matrix C.
        /// @param side Side mode: CUBLAS_SIDE_LEFT (A on left) or CUBLAS_SIDE_RIGHT (A on right).
        void SymMatrixMatrixProduct(const CudaRuntime &cudart, const Matrix_<float> &a, const Matrix_<float> &b, Matrix_<float> &c,
                                    const float alpha, const float beta, const cublasSideMode_t side)
        {
            int m, n, k;
            std::tie(m, n, k) = check_size_mm(a, b, c);

            size_t lda = get_leading(m, k);
            size_t ldb = get_leading(k, n);
            size_t ldc = get_leading(m, n);

            check_device_alloc(cudart, a);
            check_device_alloc(cudart, b);
            check_device_alloc(cudart, c);

            cudart.cublasSetStream_();
            cublasStatus_t istat = cublasSsymm(cudart.handle, side, tri_gpu, m, n, &alpha, a.gpu_data(), lda, b.gpu_data(), ldb, &beta, c.gpu_data(), ldc);
            get_cublas_error(istat);
        };

        /// @brief Computes matrix-matrix product using TF32 precision for single-precision matrices (GEMM).
        ///
        /// Performs general matrix multiplication using TensorFloat-32 (TF32) precision for higher performance
        /// on Ampere and newer architectures, computing C = alpha*A*B + beta*C.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param Ta Transposition flag for matrix A ('N' or 'T').
        /// @param Tb Transposition flag for matrix B ('N' or 'T').
        /// @param alpha Scalar factor for the matrix product.
        /// @param a Input single-precision matrix.
        /// @param b Input single-precision matrix.
        /// @param beta Scalar factor for matrix C.
        /// @param c Input/output single-precision matrix, replaced with result.
        void MatrixMatrixProductTF32(const CudaRuntime &cudart, const char *Ta, const char *Tb, const float alpha, const Matrix_<float> &a, const Matrix_<float> &b,
                                     const float beta, Matrix_<float> &c)
        {
            cublasOperation_t transa = get_trans(Ta);
            cublasOperation_t transb = get_trans(Tb);

            check_device_alloc(cudart, a);
            check_device_alloc(cudart, b);
            check_device_alloc(cudart, c);

            cudaDataType_t data_type = CUDA_R_32F;
            cublasComputeType_t computeType = CUBLAS_COMPUTE_32F_FAST_TF32;

            int m, n, k;
            std::tie(m, n, k) = check_size_mm(a, b, c, transa, transb);

            size_t lda = get_leading(m, k);
            size_t ldb = get_leading(k, n);
            size_t ldc = get_leading(m, n);

            cudart.cublasSetStream_();
            cublasStatus_t istat = cublasGemmEx(cudart.handle, transa, transb, m, n, k, &alpha, a.gpu_data(), data_type, lda, b.gpu_data(),
                                                data_type, ldb, &beta, c.gpu_data(), data_type, ldc, computeType, CUBLAS_GEMM_DEFAULT);
            get_cublas_error(istat);
        };

        /// @brief General matrix-matrix multiplication using TF32 tensor cores.
        ///
        /// Computes C = alpha*op(A)*op(B) + beta*C using NVIDIA TensorFloat-32 (TF32) format
        /// for fast matrix multiplication on Ampere and newer architectures.
        /// TF32 uses reduced precision mantissa while maintaining single-precision range,
        /// providing 5-10x speedup over pure FP32 with minimal accuracy loss.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param a Input matrix A (single precision).
        /// @param b Input matrix B (single precision).
        /// @param c Input/output matrix C (single precision, updated with result).
        /// @param alpha Scaling factor for A*B product.
        /// @param beta Scaling factor for C.
        /// @param Ta Transposition flag for A ('N' or 'T').
        /// @param Tb Transposition flag for B ('N' or 'T').
        void MatrixMatrixProductTF32(const CudaRuntime &cudart, const Matrix_<float> &a, const Matrix_<float> &b, Matrix_<float> &c,
                                     const float alpha, const float beta, const char *Ta, const char *Tb)
        {
            cublasOperation_t transa = get_trans(Ta);
            cublasOperation_t transb = get_trans(Tb);

            check_device_alloc(cudart, a);
            check_device_alloc(cudart, b);
            check_device_alloc(cudart, c);

            int m, n, k;
            std::tie(m, n, k) = check_size_mm(a, b, c, transa, transb);
            cudaDataType_t data_type = CUDA_R_32F;
            cublasComputeType_t computeType = CUBLAS_COMPUTE_32F_FAST_TF32;
            size_t lda = get_leading(m, k);
            size_t ldb = get_leading(k, n);
            size_t ldc = get_leading(m, n);

            cudart.cublasSetStream_();
            cublasStatus_t istat = cublasGemmEx(cudart.handle, transa, transb, m, n, k, &alpha, a.gpu_data(), data_type, lda, b.gpu_data(),
                                                data_type, ldb, &beta, c.gpu_data(), data_type, ldc, computeType, CUBLAS_GEMM_DEFAULT);
            get_cublas_error(istat);
        };

        /// @brief General matrix-matrix multiplication with FP16 inputs and FP32 output.
        ///
        /// Computes C = alpha*op(A)*op(B) + beta*C with half-precision (FP16) inputs
        /// and single-precision (FP32) output using FP32 accumulation.
        /// Provides mixed-precision computation for memory efficiency and speed.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param Ta Transposition flag for A ('N' or 'T').
        /// @param Tb Transposition flag for B ('N' or 'T').
        /// @param alpha Scaling factor for A*B product (single precision).
        /// @param a Input matrix A (half precision).
        /// @param b Input matrix B (half precision).
        /// @param beta Scaling factor for C (single precision).
        /// @param c Input/output matrix C (single precision, updated with result).
        void MatrixMatrixProductFP16(const CudaRuntime &cudart, const char *Ta, const char *Tb, const float alpha, const Matrix_<__half> &a, const Matrix_<__half> &b,
                                     const float beta, Matrix_<float> &c)
        {
            cublasOperation_t transa = get_trans(Ta);
            cublasOperation_t transb = get_trans(Tb);

            check_device_alloc(cudart, a);
            check_device_alloc(cudart, b);
            check_device_alloc(cudart, c);

            int m, n, k;
            std::tie(m, n, k) = check_size_mm(a, b, c, transa, transb);

            cudaDataType_t sp_type = CUDA_R_32F;
            cudaDataType_t half_type = CUDA_R_16F;
            cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;

            size_t lda = get_leading(m, k);
            size_t ldb = get_leading(k, n);
            size_t ldc = get_leading(m, n);

            cudart.cublasSetStream_();
            cublasStatus_t istat = cublasGemmEx(cudart.handle, transa, transb, m, n, k, &alpha, a.gpu_data(), half_type, lda, b.gpu_data(),
                                                half_type, ldb, &beta, c.gpu_data(), sp_type, ldc, computeType, CUBLAS_GEMM_DEFAULT);
            get_cublas_error(istat);
        };

        void MatrixMatrixProductFP16(const CudaRuntime &cudart, const Matrix_<__half> &a, const Matrix_<__half> &b, Matrix_<float> &c,
                                     const float alpha, const float beta, const char *Ta, const char *Tb)
        {
            MatrixMatrixProductFP16(cudart, Ta, Tb, alpha, a, b, beta, c);
        };

        // Double precision implementation
        void MatrixMatrixProduct(CudaRuntime &cudart, const char *Ta, const char *Tb,
                                 const double alpha, const BlockMatrix_<double> &a,
                                 const Matrix_<double> &b, const double beta, Matrix_<double> &c)
        {
            MatrixMatrixProduct_sparse_dense(cudart, Ta, Tb, alpha, a, b, beta, c);
        }

        // Float precision implementation
        void MatrixMatrixProduct(CudaRuntime &cudart, const char *Ta, const char *Tb,
                                 const float alpha, const BlockMatrix_<float> &a,
                                 const Matrix_<float> &b, const float beta, Matrix_<float> &c)
        {
            MatrixMatrixProduct_sparse_dense(cudart, Ta, Tb, alpha, a, b, beta, c);
        }


        template <typename T>
        void MatrixMatrixProduct_sparse_dense(CudaRuntime &cudart, const char *Ta, const char *Tb,
                                 const T alpha, const BlockMatrix_<T> &a,
                                 const Matrix_<T> &b, const T beta, Matrix_<T> &c)
        {

            check_device_alloc(cudart, b);
            check_device_alloc(cudart, c);

            cusparseOperation_t OpA = get_cusparse_trans(Ta);
            cusparseOperation_t OpB = get_cusparse_trans(Tb);

            int64_t M, N, m, k;
            std::tie(M, N, m, k) = check_size_sparse_dense(a, b, OpA, OpB);

            if ( a.num_blocks() == 0) throw std::invalid_argument("MatrixMatrixProduct: sparse matrix a has no non-zero blocks");

            SparseFormat format_to_use = a.get_sparse_format();
            if (OpA == CUSPARSE_OPERATION_TRANSPOSE)
            {
                format_to_use = SparseFormat::CSR;
            }

            cudaDataType_t precision = get_cuda_datatype<T>();
            SparseMatrix<T> sparse(cudart, a, format_to_use); // TODO Fix that sparse matrix is not instantiated every time
            sparse.allocate_gpu_memory();
            sparse.transfer_to_device(cudart);
            sparse.create_descriptor(cudart, precision);
            cusparseSpMatDescr_t mat_a = sparse.get_descriptor();

            // Create dense descriptors for b and c
            cusparseHandle_t handle = cudart.getcuSparseHandle();
            cusparseDnMatDescr_t mat_b = nullptr;
            get_cusparse_error(cusparseCreateDnMat(&mat_b, b.shape().first, b.shape().second, b.shape().first,
                                                   b.gpu_data(), precision, CUSPARSE_ORDER_COL));
            cusparseDnMatDescr_t mat_c = nullptr;
            get_cusparse_error(cusparseCreateDnMat(&mat_c, M, N, M,
                                                   c.gpu_data(), precision, CUSPARSE_ORDER_COL));

            // Workspace
            size_t workspace_size = 0;
            get_cusparse_error(cusparseSpMM_bufferSize(handle, OpA, OpB,
                                                       &alpha, mat_a, mat_b, &beta, mat_c,
                                                       precision, CUSPARSE_SPMM_ALG_DEFAULT, &workspace_size));
            void *workspace = nullptr;
            if (workspace_size > 0)
            {
                get_cuda_error(cudaMalloc(&workspace, workspace_size));
            }

            // Perform the matrix-matrix multiplication
            get_cusparse_error(cusparseSpMM(handle, OpA, OpB,
                                            &alpha, mat_a, mat_b, &beta, mat_c,
                                            precision, CUSPARSE_SPMM_ALG_DEFAULT, workspace));

            cudart.synchronize();

            if (workspace) get_cuda_error(cudaFree(workspace));
            get_cusparse_error(cusparseDestroyDnMat(mat_b));
            get_cusparse_error(cusparseDestroyDnMat(mat_c));
        }

        // Double precision implementation
        void MatrixMatrixProduct(CudaRuntime &cudart, const char *Ta, const char *Tb,
                                 const double alpha, const Matrix_<double> &a,
                                 const BlockMatrix_<double> &b, const double beta, Matrix_<double> &c)
        {
            MatrixMatrixProduct_dense_sparse(cudart, Ta, Tb, alpha, a, b, beta, c);
        }

        // Float precision implementation
        void MatrixMatrixProduct(CudaRuntime &cudart, const char *Ta, const char *Tb,
                                 const float alpha, const Matrix_<float> &a,
                                 const BlockMatrix_<float> &b, const float beta, Matrix_<float> &c)
        {
            MatrixMatrixProduct_dense_sparse(cudart, Ta, Tb, alpha, a, b, beta, c);
        }

        template <typename T>
        void MatrixMatrixProduct_dense_sparse(CudaRuntime &cudart, const char *Ta, const char *Tb,
                                 const T alpha, const Matrix_<T> &a,
                                 const BlockMatrix_<T> &b, const T beta, Matrix_<T> &c)
        {
            check_device_alloc(cudart, a);
            check_device_alloc(cudart, c);

            if ( b.num_blocks() == 0) throw std::invalid_argument("MatrixMatrixProduct: sparse matrix b has no non-zero blocks");

            cusparseOperation_t userOpA = get_cusparse_trans(Ta);
            cusparseOperation_t userOpB = get_cusparse_trans(Tb);

            int64_t M, N, m, k;
            std::tie(M, N, m, k) = check_size_dense_sparse(a, b, userOpA, userOpB);

            // We flip the operations because cusparseSpMM expects the sparse matrix to be the first argument, and the dense matrix to be the second argument.
            cusparseOperation_t cusparse_opA = flip_cusparse_trans(userOpB);
            cusparseOperation_t cusparse_opB = flip_cusparse_trans(userOpA);

            // BSR doesn't support transposed operations, so fall back to CSR if needed
            SparseFormat format_to_use = b.get_sparse_format();
            if (cusparse_opA == CUSPARSE_OPERATION_TRANSPOSE && format_to_use == SparseFormat::BSR)
            {
                format_to_use = SparseFormat::CSR;
            }

            SparseMatrix<T> sparse(cudart, b, format_to_use);
            sparse.allocate_gpu_memory();
            sparse.transfer_to_device(cudart);

            int64_t rows_A = static_cast<int64_t>(a.shape().first);
            int64_t cols_A = static_cast<int64_t>(a.shape().second);

            cudaDataType_t precision = get_cuda_datatype<T>();

            sparse.create_descriptor(cudart, precision);
            cusparseSpMatDescr_t mat_b = sparse.get_descriptor();

            T *a_gpu = a.gpu_data();
            T *c_gpu = c.gpu_data();

            cusparseHandle_t handle = cudart.getcuSparseHandle();
            cusparseDnMatDescr_t mat_a = nullptr;
            get_cusparse_error(cusparseCreateDnMat(&mat_a, rows_A, cols_A, rows_A,
                                                   (void *)a_gpu, precision, CUSPARSE_ORDER_COL));
            cusparseDnMatDescr_t mat_c = nullptr;
            get_cusparse_error(cusparseCreateDnMat(&mat_c, N, M, M,
                                                   (void *)c_gpu, precision, CUSPARSE_ORDER_ROW));

            size_t workspace_size = 0;
            get_cusparse_error(cusparseSpMM_bufferSize(handle, cusparse_opA, cusparse_opB,
                                                       &alpha, mat_b, mat_a, &beta, mat_c,
                                                       precision, CUSPARSE_SPMM_ALG_DEFAULT, &workspace_size));

            void *workspace = nullptr;
            if (workspace_size > 0)
            {
                get_cuda_error(cudaMalloc(&workspace, workspace_size));
            }

            get_cusparse_error(cusparseSpMM(handle, cusparse_opA, cusparse_opB,
                                            &alpha, mat_b, mat_a, &beta, mat_c,
                                            precision, CUSPARSE_SPMM_ALG_DEFAULT, workspace));

            cudart.synchronize();

            if (workspace) get_cuda_error(cudaFree(workspace));
            // mat_b is managed by sparse matrix and will be destroyed in its destructor
            get_cusparse_error(cusparseDestroyDnMat(mat_a));
            get_cusparse_error(cusparseDestroyDnMat(mat_c));
        }

    }
} // namespace lahva
