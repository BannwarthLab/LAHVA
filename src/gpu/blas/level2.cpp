/// @file level2.cpp
/// @brief GPU-accelerated Level-2 BLAS operations implementation.
///
/// Implements GPU versions of Level-2 BLAS operations (matrix-vector operations)
/// using NVIDIA's cuBLAS library. Supports matrix-vector multiplication, outer products,
/// and related operations on GPU vectors and matrices.

#include "impl/gpu/utils.hpp"
#include "impl/blas/gpu/level2.hpp"
#include "linalg.hpp"
#include "runtime.hpp"

namespace lahva
{
    namespace gpu
    {
        /// @brief Simple interface to DGER \f$\mathbf{A}=alpha*\vec{x}*\vec{y}**T + A\f$
        /// @param x Vector x
        /// @param incx stride of Vector x
        /// @param y vector y
        /// @param incy stride of Vector y
        /// @param alpha value by which \f$\vec{x}*\vec{y}**T\f$ is scaled
        /// @param A matrix A of size ndimX by ndimY
        void OuterVectorProduct(const CudaRuntime &cudart, const Vector<double>& x, const Vector<double>& y, Matrix<double>& A, size_t incx, size_t incy, const double alpha) {
            if (A.shape().first != x.size() || A.shape().second != y.size()) {
                throw std::invalid_argument("OuterVectorProduct: Output matrix A has incorrect shape.");
            }

            size_t inx = incx;
            size_t iny = incy;

            check_device_alloc(cudart, A);
            check_device_alloc(cudart, x);
            check_device_alloc(cudart, y);

            size_t lda = get_leading(x.size(), y.size());
            cudart.cublasSetStream_();
            cublasStatus_t istat = cublasDger(cudart.handle, x.size(), y.size(), &alpha, x.gpu_data(), inx, y.gpu_data(), iny, A.gpu_data(), lda);
            get_cublas_error(istat);
        }

        /// @brief Simple interface to SGER \f$\mathbf{A}=alpha*\vec{x}*\vec{y}**T + A\f$
        /// @param x Vector x
        /// @param incx stride of Vector x
        /// @param y vector y
        /// @param incy stride of Vector y
        /// @param alpha value by which \f$\vec{x}*\vec{y}**T\f$ is scaled
        /// @param A matrix A of size ndimX by ndimY
        void OuterVectorProduct(const CudaRuntime &cudart, const Vector<float>& x, const Vector<float>& y, Matrix<float>& A, size_t incx, size_t incy, const float alpha) {
            if (A.shape().first != x.size() || A.shape().second != y.size()) {
                throw std::invalid_argument("OuterVectorProduct: Output matrix A has incorrect shape.");
            }

            size_t inx = incx;
            size_t iny = incy;

            check_device_alloc(cudart, A);
            check_device_alloc(cudart, x);
            check_device_alloc(cudart, y);

            size_t lda = get_leading(x.size(), y.size());
            cudart.cublasSetStream_();
            cublasStatus_t istat = cublasSger(cudart.handle, x.size(), y.size(), &alpha, x.gpu_data(), inx, y.gpu_data(), iny, A.gpu_data(), lda);
            get_cublas_error(istat);
        }
       
        /// @brief Computes matrix-vector product \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$ for double-precision (DGEMV).
        ///
        /// Performs matrix-vector multiplication supporting optional transposition of matrix A.
        /// Computes \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$, \f$\vec{y}=alpha*\mathbf{A}^T*\vec{x}+beta*\vec{y}\f$, or \f$\vec{y}=alpha*conj(\mathbf{A}^T)*\vec{x}+beta*\vec{y}\f$.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param T Transposition character: 'N' (no transpose), 'T' (transpose), 'C' (conjugate transpose).
        /// @param alpha Scalar factor for the matrix-vector product.
        /// @param a Input double-precision matrix in column-major ordering.
        /// @param x Input double-precision vector.
        /// @param incx Memory stride for vector x.
        /// @param beta Scalar factor for vector y.
        /// @param y Input/output double-precision vector, replaced with result.
        /// @param incy Memory stride for vector y.
        void MatrixVectorProduct(const CudaRuntime &cudart, const char *T, const double alpha, const Matrix_<double> &a,
                                 const Vector_<double> &x, const size_t incx, const double beta, Vector_<double> &y, const size_t incy)
        {
            size_t inx = incx;
            size_t iny = incy;

            check_device_alloc(cudart, a);
            check_device_alloc(cudart, x);
            check_device_alloc(cudart, y);

            cublasOperation_t trans = get_trans(T);
            int nrow, ncol;
            std::tie(nrow, ncol) = check_size_mv(a, x, y, trans);
            size_t lda = get_leading(nrow, ncol);
            cudart.cublasSetStream_();
            cublasStatus_t istat = cublasDgemv(cudart.handle, trans, nrow, ncol, &alpha, a.gpu_data(), lda, x.gpu_data(), inx, &beta, y.gpu_data(), iny);
            get_cublas_error(istat);
        };

        /// @brief Computes matrix-vector product with default parameters for double-precision (DGEMV).
        ///
        /// Performs matrix-vector multiplication \f$\vec{y}=\mathbf{A}*\vec{x}\f$ with optional transposition.
        /// Parameters have defaults: T defaults to "N", alpha defaults to 1.0, beta defaults to 0.0, strides default to 1.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param a Input double-precision matrix in column-major ordering.
        /// @param x Input double-precision vector.
        /// @param y Input/output double-precision vector, replaced with result.
        /// @param T Transposition character: 'N' (no transpose), 'T' (transpose), 'C' (conjugate transpose).
        /// @param alpha Scalar factor for the matrix-vector product.
        /// @param beta Scalar factor for vector y.
        /// @param incx Memory stride for vector x.
        /// @param incy Memory stride for vector y.
        void MatrixVectorProduct(const CudaRuntime &cudart, const Matrix_<double> &a, const Vector_<double> &x, Vector_<double> &y, const char *T,
                                 const double alpha, const double beta, const size_t incx, const size_t incy)
        {
            size_t inx = incx;
            size_t iny = incy;

            check_device_alloc(cudart, a);
            check_device_alloc(cudart, x);
            check_device_alloc(cudart, y);

            cublasOperation_t trans = get_trans(T);
            int nrow, ncol;
            std::tie(nrow, ncol) = check_size_mv(a, x, y, trans);
            size_t lda = get_leading(nrow, ncol);
            cudart.cublasSetStream_();
            cublasStatus_t istat = cublasDgemv(cudart.handle, trans, nrow, ncol, &alpha, a.gpu_data(), lda, x.gpu_data(), inx, &beta, y.gpu_data(), iny);
            get_cublas_error(istat);
        };

        /// @brief Computes matrix-vector product \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$ for single-precision (SGEMV).
        ///
        /// Performs matrix-vector multiplication supporting optional transposition of matrix A.
        /// Computes \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$, \f$\vec{y}=alpha*\mathbf{A}^T*\vec{x}+beta*\vec{y}\f$, or \f$\vec{y}=alpha*conj(\mathbf{A}^T)*\vec{x}+beta*\vec{y}\f$.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param T Transposition character: 'N' (no transpose), 'T' (transpose), 'C' (conjugate transpose).
        /// @param alpha Scalar factor for the matrix-vector product.
        /// @param a Input single-precision matrix in column-major ordering.
        /// @param x Input single-precision vector.
        /// @param incx Memory stride for vector x.
        /// @param beta Scalar factor for vector y.
        /// @param y Input/output single-precision vector, replaced with result.
        /// @param incy Memory stride for vector y.
        void MatrixVectorProduct(const CudaRuntime &cudart, const char *T, const float alpha, const Matrix_<float> &a,
                                 const Vector_<float> &x, const size_t incx, const float beta, Vector_<float> &y, const size_t incy)
        {
            size_t inx = incx;
            size_t iny = incy;

            check_device_alloc(cudart, a);
            check_device_alloc(cudart, x);
            check_device_alloc(cudart, y);

            cublasOperation_t trans = get_trans(T);
            int nrow, ncol;
            std::tie(nrow, ncol) = check_size_mv(a, x, y, trans);
            size_t lda = get_leading(nrow, ncol);
            cudart.cublasSetStream_();
            cublasStatus_t istat = cublasSgemv(cudart.handle, trans, nrow, ncol, &alpha, a.gpu_data(), lda, x.gpu_data(), inx, &beta, y.gpu_data(), iny);
            get_cublas_error(istat);
        };

        /// @brief Computes matrix-vector product with default parameters for single-precision (SGEMV).
        ///
        /// Performs matrix-vector multiplication \f$\vec{y}=\mathbf{A}*\vec{x}\f$ with optional transposition.
        /// Parameters have defaults: T defaults to "N", alpha defaults to 1.0, beta defaults to 0.0, strides default to 1.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param a Input single-precision matrix in column-major ordering.
        /// @param x Input single-precision vector.
        /// @param y Input/output single-precision vector, replaced with result.
        /// @param T Transposition character: 'N' (no transpose), 'T' (transpose), 'C' (conjugate transpose).
        /// @param alpha Scalar factor for the matrix-vector product.
        /// @param beta Scalar factor for vector y.
        /// @param incx Memory stride for vector x.
        /// @param incy Memory stride for vector y.
        void MatrixVectorProduct(const CudaRuntime &cudart, const Matrix_<float> &a, const Vector_<float> &x, Vector_<float> &y, const char *T,
                                 const float alpha, const float beta, const size_t incx, const size_t incy)
        {
            size_t inx = incx;
            size_t iny = incy;

            check_device_alloc(cudart, a);
            check_device_alloc(cudart, x);
            check_device_alloc(cudart, y);

            cublasOperation_t trans = get_trans(T);
            int nrow, ncol;
            std::tie(nrow, ncol) = check_size_mv(a, x, y, trans);
            size_t lda = get_leading(nrow, ncol);
            cudart.cublasSetStream_();
            cublasStatus_t istat = cublasSgemv(cudart.handle, trans, nrow, ncol, &alpha, a.gpu_data(), lda, x.gpu_data(), inx, &beta, y.gpu_data(), iny);
            get_cublas_error(istat);
        };

        /*! @brief Simple interface to DGEMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
       or \f$\vec{y}=alpha*\mathbf{A}^\intercal*\vec{x}+beta*\vec{y}\f$ or
       \f$\vec{y}=alpha*conj(\mathbf{A}^\intercal)*\vec{x}+beta*\vec{y}\f$ for specified stride
           @param[in] T character representing the equation execute "N", "T", or "C"
           @param[in] alpha double value by which A*x is scaled
           @param[in] a pointer to the A matrix in column-cudart.handle ordering
           @param[in] x pointer to the Vector_ x
           @param[in] incx stride of Vector_ x
           @param[in] beta double value by which y is scaled
           @param[in,out] y pointer to the y Vector_ values
           @param[in] incy stride of Vector_ y
       */
        void MatrixVectorProduct(const CudaRuntime &cudart, const char *T, const complex_double alpha, const Matrix_<complex_double> &a,
                                 const Vector_<complex_double> &x, const size_t incx, const complex_double beta, Vector_<complex_double> &y, const size_t incy)
        {
            size_t inx = incx;
            size_t iny = incy;

            check_device_alloc(cudart, a);
            check_device_alloc(cudart, x);
            check_device_alloc(cudart, y);

            cublasOperation_t trans = get_trans(T);
            int nrow, ncol;
            std::tie(nrow, ncol) = check_size_mv(a, x, y, trans);
            size_t lda = get_leading(nrow, ncol);
            cudart.cublasSetStream_();
            get_cublas_error(cublasZgemv(cudart.handle, trans, nrow, ncol, reinterpret_cast<const cuDoubleComplex*>(&alpha), 
                                         reinterpret_cast<const cuDoubleComplex*>(a.gpu_data()), lda, reinterpret_cast<const cuDoubleComplex*>(x.gpu_data()), 
                                         inx, reinterpret_cast<const cuDoubleComplex*>(&beta), reinterpret_cast<cuDoubleComplex*>(y.gpu_data()), iny));
        };

        /*! @brief Simple interface to DGEMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
        or \f$\vec{y}=alpha*\mathbf{A}^\intercal*\vec{x}+beta*\vec{y}\f$ or
        \f$\vec{y}=alpha*conj(\mathbf{A}^\intercal)*\vec{x}+beta*\vec{y}\f$ for specified stride
        @param[in] a pointer to the A matrix in column-cudart.handle ordering
        @param[in] x pointer to the Vector_ x
        @param[in,out] y pointer to the y Vector_ values
        @param[in] T (optional, default value "N") character representing the equation execute "N", "T", or "C"
        @param[in] alpha (optional, default 1.0) double value by which A*x is scaled
        @param[in] beta (optional, default 0.0) double value by which y is scaled
        @param[in] incx (optional, default 1) stride of Vector_ x
        @param[in] incy (optional, default 1) stride of Vector_ y
        */
        void MatrixVectorProduct(const CudaRuntime &cudart, const Matrix_<complex_double> &a, const Vector_<complex_double> &x, Vector_<complex_double> &y, const char *T,
                                 const complex_double alpha, const complex_double beta, const size_t incx, const size_t incy)
        {
            size_t inx = incx;
            size_t iny = incy;

            check_device_alloc(cudart, a);
            check_device_alloc(cudart, x);
            check_device_alloc(cudart, y);

            cublasOperation_t trans = get_trans(T);
            int nrow, ncol;
            std::tie(nrow, ncol) = check_size_mv(a, x, y, trans);
            size_t lda = get_leading(nrow, ncol);
            cudart.cublasSetStream_();
            get_cublas_error(cublasZgemv(cudart.handle, trans, nrow, ncol, reinterpret_cast<const cuDoubleComplex*>(&alpha), 
                                         reinterpret_cast<const cuDoubleComplex*>(a.gpu_data()), lda, reinterpret_cast<const cuDoubleComplex*>(x.gpu_data()), 
                                         inx, reinterpret_cast<const cuDoubleComplex*>(&beta), reinterpret_cast<cuDoubleComplex*>(y.gpu_data()), iny));
        };

        /*! @brief Simple interface to DGEMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
        or \f$\vec{y}=alpha*\mathbf{A}^\intercal*\vec{x}+beta*\vec{y}\f$ or
        \f$\vec{y}=alpha*conj(\mathbf{A}^\intercal)*\vec{x}+beta*\vec{y}\f$  for specified stride
        @param[in] T character representing the equation execute "N", "T", or "C"
        @param[in] alpha double value by which A*x is scaled
        @param[in] a pointer to the A matrix in column-cudart.handle ordering
        @param[in] x pointer to the Vector_ x
        @param[in] incx stride of Vector_ x
        @param[in] beta double value by which y is scaled
        @param[in,out] y pointer to the y Vector_ values
        @param[in] incy stride of Vector_ y
        */
        void MatrixVectorProduct(const CudaRuntime &cudart, const char *T, const complex_float alpha, const Matrix_<complex_float> &a,
                                 const Vector_<complex_float> &x, const size_t incx, const complex_float beta, Vector_<complex_float> &y, const size_t incy)
        {
            size_t inx = incx;
            size_t iny = incy;

            check_device_alloc(cudart, a);
            check_device_alloc(cudart, x);
            check_device_alloc(cudart, y);

            cublasOperation_t trans = get_trans(T);
            int nrow, ncol;
            std::tie(nrow, ncol) = check_size_mv(a, x, y, trans);
            size_t lda = get_leading(nrow, ncol);
            cudart.cublasSetStream_();
            get_cublas_error(cublasCgemv(cudart.handle, trans, nrow, ncol, reinterpret_cast<const cuComplex*>(&alpha), 
                                         reinterpret_cast<const cuComplex*>(a.gpu_data()), lda, reinterpret_cast<const cuComplex*>(x.gpu_data()), 
                                         inx, reinterpret_cast<const cuComplex*>(&beta), reinterpret_cast<cuComplex*>(y.gpu_data()), iny));
        };

        /*! @brief Simple interface to SGEMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
        or \f$\vec{y}=alpha*\mathbf{A}^\intercal*\vec{x}+beta*\vec{y}\f$ or
        \f$\vec{y}=alpha*conj(\mathbf{A}^\intercal)*\vec{x}+beta*\vec{y}\f$ for specified stride
        @param[in] a pointer to the A matrix in column-cudart.handle ordering
        @param[in] x pointer to the Vector_ x
        @param[in,out] y pointer to the y Vector_ values
        @param[in] T (optional, default value "N") character representing the equation execute "N", "T", or "C"
        @param[in] alpha (optional, default 1.0) double value by which A*x is scaled
        @param[in] beta (optional, default 0.0) double value by which y is scaled
        @param[in] incx (optional, default 1) stride of Vector_ x
        @param[in] incy (optional, default 1) stride of Vector_ y
        */
        void MatrixVectorProduct(const CudaRuntime &cudart, const Matrix_<complex_float> &a, const Vector_<complex_float> &x, Vector_<complex_float> &y, const char *T,
                                 const complex_float alpha, const complex_float beta, const size_t incx, const size_t incy)
        {
            size_t inx = incx;
            size_t iny = incy;

            check_device_alloc(cudart, a);
            check_device_alloc(cudart, x);
            check_device_alloc(cudart, y);

            cublasOperation_t trans = get_trans(T);
            int nrow, ncol;
            std::tie(nrow, ncol) = check_size_mv(a, x, y, trans);
            size_t lda = get_leading(nrow, ncol);
            cudart.cublasSetStream_();
            get_cublas_error(cublasCgemv(cudart.handle, trans, nrow, ncol, reinterpret_cast<const cuComplex*>(&alpha), 
                                         reinterpret_cast<const cuComplex*>(a.gpu_data()), lda, reinterpret_cast<const cuComplex*>(x.gpu_data()), 
                                         inx, reinterpret_cast<const cuComplex*>(&beta), reinterpret_cast<cuComplex*>(y.gpu_data()), iny));
        };

        /// @brief Computes batched matrix-vector product for block-diagonal matrices \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$ (double precision).
        ///
        /// Performs batched matrix-vector multiplication on block-diagonal matrices supporting optional transposition.
        /// Computes \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$, \f$\vec{y}=alpha*\mathbf{A}^T*\vec{x}+beta*\vec{y}\f$, or \f$\vec{y}=alpha*conj(\mathbf{A}^T)*\vec{x}+beta*\vec{y}\f$.
        /// The operation is performed using CUDA's strided batch GEMV on GPU-resident blocks.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param ta Transposition character: 'N' (no transpose), 'T' (transpose), 'C' (conjugate transpose).
        /// @param alpha Scalar factor for the matrix-vector product.
        /// @param a Input block-diagonal matrix with double-precision blocks.
        /// @param x Input double-precision vector matching the column structure of a.
        /// @param beta Scalar factor for vector y.
        /// @param y Input/output double-precision vector, replaced with result.
        void MatrixVectorProduct(const CudaRuntime& cudart, const char* ta, const double alpha, const BlockDiagMatrix<double>& a,
                                const Vector<double>& x, const double beta, Vector<double>& y)
        {
            cublasOperation_t transa = get_trans(ta);
            int num_blocks = a.num_blocks();

            // Ensure block-diagonal matrix A is on GPU
            const auto& gpu_a_data = a.ensure_on_gpu(cudart);

            size_t max_m = gpu_a_data.max_m;
            size_t max_k = gpu_a_data.max_k;
            long long padded_stride_a = gpu_a_data.padded_stride;
            double *d_a = gpu_a_data.d_data;

            size_t lda = max_m;
            size_t incx = 1;
            size_t incy = 1;

            // Pack X vector blocks and copy to GPU
            Vector<double> x_packed(max_k * num_blocks, 0.0);
            a.pack_vector_to_gpu(cudart, x, x_packed, a.get_row_offsets(), a.get_block_cols());
            double *d_x = x_packed.gpu_data();

            // Pack Y vector blocks and copy to GPU
            Vector<double> y_packed(max_m * num_blocks, 0.0);
            a.pack_vector_to_gpu(cudart, y, y_packed, a.get_row_offsets(), a.get_block_rows());
            double *d_y = y_packed.gpu_data();

            // Perform batched GEMV: Y = alpha * op(A) * X + beta * Y
            cudart.cublasSetStream_();
            cublasStatus_t istat = cublasDgemvStridedBatched(
                cudart.handle,
                transa,
                max_m, max_k,
                &alpha,
                d_a, lda, padded_stride_a,
                d_x, incx, max_k,
                &beta,
                d_y, incy, max_m,
                num_blocks
            );
            get_cublas_error(istat);

            // Copy result back to CPU and unpack
            cudart.synchronize();
            y_packed.copy2host(cudart);
            a.unpack_vector_from_gpu(cudart, y_packed, y, a.get_row_offsets(), a.get_block_rows());
        }

        /// @brief Computes batched matrix-vector product for block-diagonal matrices \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$ (single precision).
        ///
        /// Performs batched matrix-vector multiplication on block-diagonal matrices supporting optional transposition.
        /// Computes \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$, \f$\vec{y}=alpha*\mathbf{A}^T*\vec{x}+beta*\vec{y}\f$, or \f$\vec{y}=alpha*conj(\mathbf{A}^T)*\vec{x}+beta*\vec{y}\f$.
        /// The operation is performed using CUDA's strided batch GEMV on GPU-resident blocks.
        ///
        /// @param cudart CUDA runtime instance.
        /// @param ta Transposition character: 'N' (no transpose), 'T' (transpose), 'C' (conjugate transpose).
        /// @param alpha Scalar factor for the matrix-vector product.
        /// @param a Input block-diagonal matrix with single-precision blocks.
        /// @param x Input single-precision vector matching the column structure of a.
        /// @param beta Scalar factor for vector y.
        /// @param y Input/output single-precision vector, replaced with result.
        void MatrixVectorProduct(const CudaRuntime& cudart, const char* ta, const float alpha, const BlockDiagMatrix<float>& a,
                                const Vector<float>& x, const float beta, Vector<float>& y)
        {
            cublasOperation_t transa = get_trans(ta);
            int num_blocks = a.num_blocks();

            // Ensure block-diagonal matrix A is on GPU
            const auto& gpu_a_data = a.ensure_on_gpu(cudart);
            size_t max_m = gpu_a_data.max_m;
            size_t max_k = gpu_a_data.max_k;
            long long padded_stride_a = gpu_a_data.padded_stride;
            float *d_a = reinterpret_cast<float*>(gpu_a_data.d_data);

            size_t lda = max_m;
            size_t incx = 1;
            size_t incy = 1;

            // Pack X vector blocks and copy to GPU
            Vector<float> x_packed(max_k * num_blocks, 0.0f);
            a.pack_vector_to_gpu(cudart, x, x_packed, a.get_row_offsets(), a.get_block_cols());
            float *d_x = x_packed.gpu_data();

            // Pack Y vector blocks and copy to GPU
            Vector<float> y_packed(max_m * num_blocks, 0.0f);
            a.pack_vector_to_gpu(cudart, y, y_packed, a.get_row_offsets(), a.get_block_rows());
            float *d_y = y_packed.gpu_data();

            // Perform batched GEMV: Y = alpha * op(A) * X + beta * Y
            cudart.cublasSetStream_();
            cublasStatus_t istat = cublasSgemvStridedBatched(
                cudart.handle,
                transa,
                max_m, max_k,
                &alpha,
                d_a, lda, padded_stride_a,
                d_x, incx, max_k,
                &beta,
                d_y, incy, max_m,
                num_blocks
            );
            get_cublas_error(istat);

            // Copy result back to CPU and unpack
            cudart.synchronize();
            y_packed.copy2host(cudart);
            a.unpack_vector_from_gpu(cudart, y_packed, y, a.get_row_offsets(), a.get_block_rows());
        }

        /*! @brief Simple interface to DSYMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
        for specified stride
            @param[in] alpha double value by which A*x is scaled
            @param[in] a pointer to the A matrix in column-cudart.handle ordering
            @param[in] x pointer to the Vector_ x
            @param[in] incx stride of Vector_ x
            @param[in] beta double value by which y is scaled
            @param[in,out] y pointer to the y Vector_ values
            @param[in] incy stride of Vector_ y
        */
        void SymMatrixVectorProduct(const CudaRuntime &cudart, const double alpha, const Matrix_<double> &a,
                                    const Vector_<double> &x, const size_t incx, const double beta, Vector_<double> &y, const size_t incy)
        {

            int nrow, ncol;
            std::tie(nrow, ncol) = check_size_mv(a, x, y);
            size_t inx = incx;
            size_t iny = incy;
            check_device_alloc(cudart, a);
            check_device_alloc(cudart, x);
            check_device_alloc(cudart, y);

            size_t lda = get_leading(nrow, ncol);
            cudart.cublasSetStream_();
            cublasStatus_t istat = cublasDsymv(cudart.handle, tri_gpu, ncol, &alpha, a.gpu_data(), lda, x.gpu_data(), inx, &beta, y.gpu_data(), iny);
            get_cublas_error(istat);
        };

        /*! @brief Simple interface to DSYMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
        for specified stride
            @param[in] a pointer to the A matrix in column-cudart.handle ordering
            @param[in] x pointer to the Vector_ x
            @param[in] incx stride of Vector_ x
            @param[in,out] y pointer to the y Vector_ values
            @param[in] alpha (optional, default 1.0) double value by which A*x is scaled
            @param[in] beta (optional, default 0.0) double value by which y is scaled
            @param[in] incy (optional, default 1) stride of Vector_ y
        */
        void SymMatrixVectorProduct(const CudaRuntime &cudart, const Matrix_<double> &a, const Vector_<double> &x, Vector_<double> &y,
                                    const double alpha, const double beta, const size_t incx, const size_t incy)
        {
            int nrow, ncol;
            std::tie(nrow, ncol) = check_size_mv(a, x, y);
            size_t inx = incx;
            size_t iny = incy;
            check_device_alloc(cudart, a);
            check_device_alloc(cudart, x);
            check_device_alloc(cudart, y);

            size_t lda = get_leading(nrow, ncol);
            cudart.cublasSetStream_();
            cublasStatus_t istat = cublasDsymv(cudart.handle, tri_gpu, ncol, &alpha, a.gpu_data(), lda, x.gpu_data(), inx, &beta, y.gpu_data(), iny);
            get_cublas_error(istat);
        };

        /*! @brief Simple interface to DSYMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
        for specified stride
            @param[in] alpha float value by which A*x is scaled
            @param[in] a pointer to the A matrix in column-cudart.handle ordering
            @param[in] x pointer to the Vector_ x
            @param[in] incx stride of Vector_ x
            @param[in] beta float value by which y is scaled
            @param[in,out] y pointer to the y Vector_ values
            @param[in] incy stride of Vector_ y
        */
        void SymMatrixVectorProduct(const CudaRuntime &cudart, const float alpha, const Matrix_<float> &a,
                                    const Vector_<float> &x, const size_t incx, const float beta, Vector_<float> &y, const size_t incy)
        {
            int nrow, ncol;
            std::tie(nrow, ncol) = check_size_mv(a, x, y);
            size_t inx = incx;
            size_t iny = incy;
            check_device_alloc(cudart, a);
            check_device_alloc(cudart, x);
            check_device_alloc(cudart, y);

            size_t lda = get_leading(nrow, ncol);
            cudart.cublasSetStream_();
            cublasStatus_t istat = cublasSsymv(cudart.handle, tri_gpu, ncol, &alpha, a.gpu_data(), lda, x.gpu_data(), inx, &beta, y.gpu_data(), iny);
            get_cublas_error(istat);
        };

        /*! @brief Simple interface to SSYMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
        for specified stride
            @param[in] a pointer to the A matrix in column-cudart.handle ordering
            @param[in] x pointer to the Vector_ x
            @param[in] incx stride of Vector_ x
            @param[in,out] y pointer to the y Vector_ values
            @param[in] alpha (optional, default 1.0) float value by which A*x is scaled
            @param[in] beta (optional, default 0.0) float value by which y is scaled
            @param[in] incy (optional, default 1) stride of Vector_ y
        */
        void SymMatrixVectorProduct(const CudaRuntime &cudart, const Matrix_<float> &a, const Vector_<float> &x, Vector_<float> &y,
                                    const float alpha, const float beta, const size_t incx, const size_t incy)
        {
            int nrow, ncol;
            std::tie(nrow, ncol) = check_size_mv(a, x, y);
            size_t inx = incx;
            size_t iny = incy;
            check_device_alloc(cudart, a);
            check_device_alloc(cudart, x);
            check_device_alloc(cudart, y);

            size_t lda = get_leading(nrow, ncol);
            cudart.cublasSetStream_();
            cublasStatus_t istat = cublasSsymv(cudart.handle, tri_gpu, ncol, &alpha, a.gpu_data(), lda, x.gpu_data(), inx, &beta, y.gpu_data(), iny);
            get_cublas_error(istat);
        };

        /*! @brief Simple interface to DSYMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
        for specified stride
            @param[in] alpha double value by which A*x is scaled
            @param[in] a pointer to the A matrix in column-cudart.handle ordering
            @param[in] x pointer to the Vector_ x
            @param[in] incx stride of Vector_ x
            @param[in] beta double value by which y is scaled
            @param[in,out] y pointer to the y Vector_ values
            @param[in] incy stride of Vector_ y
        */
        void SymMatrixVectorProduct(const CudaRuntime &cudart, const double alpha, const LowTriMatrix_<double> &a,
                                    const Vector_<double> &x, const size_t incx, const double beta, Vector_<double> &y, const size_t incy)
        {
            int nrow, ncol;
            std::tie(nrow, ncol) = check_size_mv(a, x, y);
            size_t inx = incx;
            size_t iny = incy;
            check_device_alloc(cudart, a);
            check_device_alloc(cudart, x);
            check_device_alloc(cudart, y);
            cudart.cublasSetStream_();
            cublasStatus_t istat = cublasDspmv(cudart.handle, tri_gpu, ncol, &alpha, a.gpu_data(), x.gpu_data(), inx, &beta, y.gpu_data(), iny);
            get_cublas_error(istat);
        };

        /*! @brief Simple interface to DSYMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
        for specified stride
            @param[in] a pointer to the A matrix in column-cudart.handle ordering
            @param[in] x pointer to the Vector_ x
            @param[in] incx stride of Vector_ x
            @param[in,out] y pointer to the y Vector_ values
            @param[in] alpha (optional, default 1.0) double value by which A*x is scaled
            @param[in] beta (optional, default 0.0) double value by which y is scaled
            @param[in] incy (optional, default 1) stride of Vector_ y
        */
        void SymMatrixVectorProduct(const CudaRuntime &cudart, const LowTriMatrix_<double> &a, const Vector_<double> &x, Vector_<double> &y,
                                    const double alpha, const double beta, const size_t incx, const size_t incy)
        {
            int nrow, ncol;
            std::tie(nrow, ncol) = check_size_mv(a, x, y);
            size_t inx = incx;
            size_t iny = incy;
            check_device_alloc(cudart, a);
            check_device_alloc(cudart, x);
            check_device_alloc(cudart, y);

            cudart.cublasSetStream_();
            cublasStatus_t istat = cublasDspmv(cudart.handle, tri_gpu, ncol, &alpha, a.gpu_data(), x.gpu_data(), inx, &beta, y.gpu_data(), iny);
            get_cublas_error(istat);
        };

        /*! @brief Simple interface to DSYMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
        for specified stride
            @param[in] alpha float value by which A*x is scaled
            @param[in] a pointer to the A matrix in column-cudart.handle ordering
            @param[in] x pointer to the Vector_ x
            @param[in] incx stride of Vector_ x
            @param[in] beta float value by which y is scaled
            @param[in,out] y pointer to the y Vector_ values
            @param[in] incy stride of Vector_ y
        */
        void SymMatrixVectorProduct(const CudaRuntime &cudart, const float alpha, const LowTriMatrix_<float> &a,
                                    const Vector_<float> &x, const size_t incx, const float beta, Vector_<float> &y, const size_t incy)
        {
            int nrow, ncol;
            std::tie(nrow, ncol) = check_size_mv(a, x, y);
            size_t inx = incx;
            size_t iny = incy;
            check_device_alloc(cudart, a);
            check_device_alloc(cudart, x);
            check_device_alloc(cudart, y);
            cudart.cublasSetStream_();
            cublasStatus_t istat = cublasSspmv(cudart.handle, tri_gpu, ncol, &alpha, a.gpu_data(), x.gpu_data(), inx, &beta, y.gpu_data(), iny);
            get_cublas_error(istat);
        };

        /*! @brief Simple interface to SSYMV performing \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
        for specified stride
            @param[in] a pointer to the A matrix in column-cudart.handle ordering
            @param[in] x pointer to the Vector_ x
            @param[in] incx stride of Vector_ x
            @param[in,out] y pointer to the y Vector_ values
            @param[in] alpha (optional, default 1.0) float value by which A*x is scaled
            @param[in] beta (optional, default 0.0) float value by which y is scaled
            @param[in] incy (optional, default 1) stride of Vector_ y
        */
        void SymMatrixVectorProduct(const CudaRuntime &cudart, const LowTriMatrix_<float> &a, const Vector_<float> &x, Vector_<float> &y,
                                    const float alpha, const float beta, const size_t incx, const size_t incy)
        {
            int nrow, ncol;
            std::tie(nrow, ncol) = check_size_mv(a, x, y);
            size_t inx = incx;
            size_t iny = incy;
            check_device_alloc(cudart, a);
            check_device_alloc(cudart, x);
            check_device_alloc(cudart, y);

            cudart.cublasSetStream_();
            cublasStatus_t istat = cublasSspmv(cudart.handle, tri_gpu, ncol, &alpha, a.gpu_data(), x.gpu_data(), inx, &beta, y.gpu_data(), iny);
            get_cublas_error(istat);
        };

    } // namespace gpu
} // namespace lahva