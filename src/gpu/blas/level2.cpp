/// @file level2.cpp
/// @brief GPU-accelerated Level-2 BLAS operations implementation.
///
/// Implements GPU versions of Level-2 BLAS operations (matrix-vector operations)
/// using NVIDIA's cuBLAS library. Supports matrix-vector multiplication, outer products,
/// and related operations on GPU vectors and matrices.

#include "impl/gpu/utils.hpp"
#include "impl/blas/gpu/level2.hpp"
#include "impl/gpu/unpack-vector.hpp"
#include "linalg.hpp"
#include "runtime.hpp"
#include <cuda_runtime.h>

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

        template <typename T>
        void MatrixVectorProduct_blockdiag(CudaRuntime& cudart, const char* ta, const T alpha, const BlockDiagMatrix<T>& a,
                                const Vector<T>& x, const T beta, Vector<T>& y)
        {
            cublasOperation_t transa = get_trans(ta);
            int num_blocks = a.num_blocks();

            const auto& gpu_a_data = a.ensure_on_gpu(cudart);
            size_t max_m = gpu_a_data.max_m;
            size_t max_k = gpu_a_data.max_k;
            long long padded_stride_a = gpu_a_data.padded_stride;
            T *d_a = reinterpret_cast<T*>(gpu_a_data.d_data);

            size_t lda = max_m;
            size_t incx = 1;
            size_t incy = 1;

            y.copy2device(cudart);

            Vector<T> x_packed(max_k * num_blocks, (T)0);
            a.pack_vector_to_gpu(cudart, x, x_packed, a.get_row_offsets(), a.get_block_cols());
            T *d_x = x_packed.gpu_data();

            Vector<T> y_packed(max_m * num_blocks, (T)0);
            a.pack_vector_to_gpu(cudart, y, y_packed, a.get_row_offsets(), a.get_block_rows());
            T *d_y = y_packed.gpu_data();

            cudart.cublasSetStream_();
            cublasStatus_t istat;
            if (std::is_same<T, double>::value) {
                istat = cublasDgemvStridedBatched(
                    cudart.handle, transa, max_m, max_k, (const double*)&alpha,
                    (const double*)d_a, lda, padded_stride_a,
                    (const double*)d_x, incx, max_k,
                    (const double*)&beta, (double*)d_y, incy, max_m, num_blocks);
            } else {
                istat = cublasSgemvStridedBatched(
                    cudart.handle, transa, max_m, max_k, (const float*)&alpha,
                    (const float*)d_a, lda, padded_stride_a,
                    (const float*)d_x, incx, max_k,
                    (const float*)&beta, (float*)d_y, incy, max_m, num_blocks);
            }
            get_cublas_error(istat);

            unpack_vector(cudart, y_packed.gpu_data(), y.gpu_data(), a.get_d_row_offsets(cudart), a.get_d_block_row_sizes(cudart), max_m, num_blocks);
        }

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
        void MatrixVectorProduct(CudaRuntime& cudart, const char* ta, const double alpha, const BlockDiagMatrix<double>& a,
                                const Vector<double>& x, const double beta, Vector<double>& y)
        {
            MatrixVectorProduct_blockdiag(cudart, ta, alpha, a, x, beta, y);
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
        void MatrixVectorProduct(CudaRuntime& cudart, const char* ta, const float alpha, const BlockDiagMatrix<float>& a,
                                const Vector<float>& x, const float beta, Vector<float>& y)
        {
            MatrixVectorProduct_blockdiag(cudart, ta, alpha, a, x, beta, y);
        }

        /// @brief Sparse matrix-vector product with BlockMatrix via cuSPARSE.
        ///
        /// Performs sparse matrix-vector multiplication \f$\vec{y}=alpha*\mathbf{A}*\vec{x}+beta*\vec{y}\f$
        /// or \f$\vec{y}=alpha*\mathbf{A}^T*\vec{x}+beta*\vec{y}\f$ using cuSPARSE operations.
        /// The BlockMatrix is converted to sparse format (BSR for uniform blocks, CSR for non-uniform blocks).
        ///
        /// @tparam T Numeric element type (float or double).
        /// @param cudart CUDA runtime instance.
        /// @param ta Transposition character: 'N' (no transpose), 'T' (transpose).
        /// @param alpha Scalar factor for the matrix-vector product.
        /// @param a Input block-structured sparse matrix.
        /// @param x Input vector to multiply (size must match matrix columns or rows if transposed).
        /// @param beta Scalar factor for vector y.
        /// @param y Input/output vector, replaced with result (size must match matrix rows or columns if transposed).
        template <typename T>
        void MatrixVectorProduct_sparse(CudaRuntime &cudart, const char* ta, const T alpha, const BlockMatrix_<T>& a,
                                const Vector_<T>& x, const T beta, Vector_<T>& y)
        {
            cusparseOperation_t op = get_cusparse_trans(ta);
            check_size_mv(a, x, y, op);

            check_device_alloc(cudart, x);
            check_device_alloc(cudart, y);

            cudaDataType_t precision = get_cuda_datatype<T>();

            // Convert BlockMatrix to SparseMatrix and copy to GPU
            SparseMatrix<T> sparse(cudart, a, a.get_sparse_format());
            sparse.allocate_gpu_memory();
            sparse.copy2device(cudart);
            sparse.create_descriptor(cudart, precision);

            cusparseSpMatDescr_t mat_a = sparse.get_descriptor();

            // Create dense vector descriptors
            cusparseDnVecDescr_t vec_x_descr, vec_y_descr;
            get_cusparse_error(cusparseCreateDnVec(&vec_x_descr, x.size(), x.gpu_data(), precision));
            get_cusparse_error(cusparseCreateDnVec(&vec_y_descr, y.size(), y.gpu_data(), precision));

            // Allocate work buffer
            cusparseHandle_t sparseHandle = cudart.getcuSparseHandle();
            size_t bufferSize = 0;
            get_cusparse_error(cusparseSpMV_bufferSize(sparseHandle, op, &alpha, mat_a, vec_x_descr,
                                                        &beta, vec_y_descr, precision, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));

            void *buffer = nullptr;
            if (bufferSize > 0) {
                get_cuda_error(cudaMalloc(&buffer, bufferSize));
            }

            // Perform multiplication
            get_cusparse_error(cusparseSpMV(sparseHandle, op, &alpha, mat_a, vec_x_descr,
                                           &beta, vec_y_descr, precision, CUSPARSE_SPMV_ALG_DEFAULT, buffer));

            // Cleanup
            if (buffer) {
                get_cuda_error(cudaFree(buffer));
            }
            get_cusparse_error(cusparseDestroyDnVec(vec_x_descr));
            get_cusparse_error(cusparseDestroyDnVec(vec_y_descr));
        }

        // Explicit template instantiations
        template void MatrixVectorProduct_sparse<float>(CudaRuntime&, const char*, float, const BlockMatrix_<float>&, const Vector_<float>&, float, Vector_<float>&);
        template void MatrixVectorProduct_sparse<double>(CudaRuntime&, const char*, double, const BlockMatrix_<double>&, const Vector_<double>&, double, Vector_<double>&);
        template void MatrixVectorProduct_blockdiag<float>(CudaRuntime&, const char*, float, const BlockDiagMatrix<float>&, const Vector<float>&, float, Vector<float>&);
        template void MatrixVectorProduct_blockdiag<double>(CudaRuntime&, const char*, double, const BlockDiagMatrix<double>&, const Vector<double>&, double, Vector<double>&);

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