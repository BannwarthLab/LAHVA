#include "linalg.hpp"
#include "runtime.hpp"
#include "impl/blas/gpu/level3.hpp"
#include "../gpu-utils/utils.hpp"

namespace lahva
{
    namespace gpu
    {

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

        void MatrixMatrixProduct(const CudaRuntime &cudart, const char *Ta, const char *Tb,
                                 const double alpha, const BlockMatrix_<double> &a,
                                 const Matrix_<double> &b, const double beta, Matrix_<double> &c)
        {

            check_device_alloc(cudart, b);
            check_device_alloc(cudart, c);

            cusparseOperation_t opA = get_cusparse_trans(Ta);
            cusparseOperation_t opB = get_cusparse_trans(Tb);

            size_t num_blocks = a.num_blocks();
            if (num_blocks == 0)
            {
                return;
            }

            SparseFormat format = a.get_sparse_format();
            SparseMatrix<double> sparse(cudart, a, format);
            if (!sparse.is_initialized())
            {
                return;
            }
            const auto &sparse_data = sparse.get_sparse_data();

            // Get matrix dimensions for later use
            Shape a_shape = a.shape();
            int64_t m = static_cast<int64_t>(a_shape.first);
            int64_t k = static_cast<int64_t>(a_shape.second);

            int64_t blockDim_m = 1, blockDim_n = 1;
            if (format == SparseFormat::BSR && num_blocks > 0)
            {
                Shape first_block_shape = a.get_block_shape(0);
                blockDim_m = static_cast<int64_t>(first_block_shape.first);
                blockDim_n = static_cast<int64_t>(first_block_shape.second);
            }

            sparse.allocate_gpu_memory();
            sparse.transfer_to_device(cudart);

            int64_t n = (opB == CUSPARSE_OPERATION_NON_TRANSPOSE)
                            ? static_cast<int64_t>(b.shape().second)
                            : static_cast<int64_t>(b.shape().first);

            int64_t b_rows_required = (opA == CUSPARSE_OPERATION_NON_TRANSPOSE) ? k : m;
            int64_t b_rows_stored = (opB == CUSPARSE_OPERATION_NON_TRANSPOSE) ? b_rows_required : n;
            int64_t b_cols_stored = (opB == CUSPARSE_OPERATION_NON_TRANSPOSE) ? n : b_rows_required;

            int64_t c_m = (opA == CUSPARSE_OPERATION_NON_TRANSPOSE) ? m : k;
            int64_t c_n = n;

            cusparseHandle_t handle = nullptr;
            get_cusparse_error(cusparseCreate(&handle));

            cusparseSpMatDescr_t mat_descr = nullptr;
            if (sparse_data.format == SparseFormat::BSR)
            {
                int64_t num_block_rows = (m + blockDim_m - 1) / blockDim_m;
                int64_t num_block_cols = (k + blockDim_n - 1) / blockDim_n;
                int64_t nnz_blocks = static_cast<int64_t>(num_blocks);

                get_cusparse_error(cusparseCreateBsr(&mat_descr,
                                                     num_block_rows,
                                                     num_block_cols,
                                                     nnz_blocks,
                                                     blockDim_m,
                                                     blockDim_n,
                                                     sparse_data.d_row_offsets,
                                                     sparse_data.d_col_indices,
                                                     sparse_data.d_values,
                                                     CUSPARSE_INDEX_32I,
                                                     CUSPARSE_INDEX_32I,
                                                     CUSPARSE_INDEX_BASE_ZERO,
                                                     CUDA_R_64F,
                                                     CUSPARSE_ORDER_ROW));
            }
            else
            {
                get_cusparse_error(cusparseCreateCsr(&mat_descr,
                                                     m, k, sparse.nnz(),
                                                     sparse_data.d_row_offsets,
                                                     sparse_data.d_col_indices,
                                                     sparse_data.d_values,
                                                     CUSPARSE_INDEX_32I,
                                                     CUSPARSE_INDEX_32I,
                                                     CUSPARSE_INDEX_BASE_ZERO,
                                                     CUDA_R_64F));
            }

            cusparseDnMatDescr_t descr_b = nullptr;
            cusparseDnMatDescr_t descr_c = nullptr;

            get_cusparse_error(cusparseCreateDnMat(&descr_b, b_rows_stored, b_cols_stored, b_rows_stored, (void *)b.gpu_data(),
                                                   CUDA_R_64F, CUSPARSE_ORDER_COL));
            get_cusparse_error(cusparseCreateDnMat(&descr_c, c_m, c_n, c_m, (void *)c.gpu_data(),
                                                   CUDA_R_64F, CUSPARSE_ORDER_COL));

            // Allocate workspace if needed
            size_t workspace_size = 0;
            get_cusparse_error(cusparseSpMM_bufferSize(handle, opA, opB,
                                                       &alpha, mat_descr, descr_b,
                                                       &beta, descr_c,
                                                       CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, &workspace_size));

            void *workspace = nullptr;
            if (workspace_size > 0)
            {
                get_cuda_error(cudaMalloc(&workspace, workspace_size));
            }

            get_cusparse_error(cusparseSpMM(handle, opA, opB,
                                            &alpha, mat_descr, descr_b,
                                            &beta, descr_c,
                                            CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, workspace));

            cudart.synchronize();

            get_cuda_error(cudaMemcpy(c.data(), c.gpu_data(), c.shape().first * c.shape().second * sizeof(double), cudaMemcpyDeviceToHost));

            if (workspace)
            {
                get_cuda_error(cudaFree(workspace));
            }

            get_cusparse_error(cusparseDestroySpMat(mat_descr));
            get_cusparse_error(cusparseDestroyDnMat(descr_b));
            get_cusparse_error(cusparseDestroyDnMat(descr_c));
            get_cusparse_error(cusparseDestroy(handle));

            sparse.release_gpu_memory();
        }

        void MatrixMatrixProduct(const CudaRuntime &cudart, const char *Ta, const char *Tb,
                                 const float alpha, const BlockMatrix_<float> &a,
                                 const Matrix_<float> &b, const float beta, Matrix_<float> &c)
        {

            check_device_alloc(cudart, b);
            check_device_alloc(cudart, c);

            cusparseOperation_t opA = get_cusparse_trans(Ta);
            cusparseOperation_t opB = get_cusparse_trans(Tb);

            size_t num_blocks = a.num_blocks();
            if (num_blocks == 0)
            {
                return;
            }

            SparseFormat format = a.get_sparse_format();
            SparseMatrix<float> sparse(cudart, a, format);
            if (!sparse.is_initialized())
            {
                return;
            }
            const auto &sparse_data = sparse.get_sparse_data();

            Shape a_shape = a.shape();
            int64_t m = static_cast<int64_t>(a_shape.first);
            int64_t k = static_cast<int64_t>(a_shape.second);

            int64_t blockDim_m = 1, blockDim_n = 1;
            if (format == SparseFormat::BSR && num_blocks > 0)
            {
                Shape first_block_shape = a.get_block_shape(0);
                blockDim_m = static_cast<int64_t>(first_block_shape.first);
                blockDim_n = static_cast<int64_t>(first_block_shape.second);
            }

            sparse.allocate_gpu_memory();
            sparse.transfer_to_device(cudart);

            int64_t n = (opB == CUSPARSE_OPERATION_NON_TRANSPOSE)
                            ? static_cast<int64_t>(b.shape().second)
                            : static_cast<int64_t>(b.shape().first);

            int64_t b_rows_required = (opA == CUSPARSE_OPERATION_NON_TRANSPOSE) ? k : m;
            int64_t b_rows_stored = (opB == CUSPARSE_OPERATION_NON_TRANSPOSE) ? b_rows_required : n;
            int64_t b_cols_stored = (opB == CUSPARSE_OPERATION_NON_TRANSPOSE) ? n : b_rows_required;

            int64_t c_m = (opA == CUSPARSE_OPERATION_NON_TRANSPOSE) ? m : k;
            int64_t c_n = n;

            cusparseHandle_t handle = nullptr;
            cusparseStatus_t status = cusparseCreate(&handle);
            if (status != CUSPARSE_STATUS_SUCCESS)
            {
                throw std::runtime_error("Failed to create cusparse handle");
            }

            cusparseSpMatDescr_t mat_descr = nullptr;
            if (sparse_data.format == SparseFormat::BSR)
            {
                int64_t num_block_rows = (m + blockDim_m - 1) / blockDim_m;
                int64_t num_block_cols = (k + blockDim_n - 1) / blockDim_n;
                int64_t nnz_blocks = static_cast<int64_t>(num_blocks);

                get_cusparse_error(cusparseCreateBsr(&mat_descr,
                                                     num_block_rows,
                                                     num_block_cols,
                                                     nnz_blocks,
                                                     blockDim_m,
                                                     blockDim_n,
                                                     sparse_data.d_row_offsets,
                                                     sparse_data.d_col_indices,
                                                     sparse_data.d_values,
                                                     CUSPARSE_INDEX_32I,
                                                     CUSPARSE_INDEX_32I,
                                                     CUSPARSE_INDEX_BASE_ZERO,
                                                     CUDA_R_32F,
                                                     CUSPARSE_ORDER_ROW));
            }
            else
            {
                get_cusparse_error(cusparseCreateCsr(&mat_descr,
                                                     m, k, sparse.nnz(),
                                                     sparse_data.d_row_offsets,
                                                     sparse_data.d_col_indices,
                                                     sparse_data.d_values,
                                                     CUSPARSE_INDEX_32I,
                                                     CUSPARSE_INDEX_32I,
                                                     CUSPARSE_INDEX_BASE_ZERO,
                                                     CUDA_R_32F));
            }

            cusparseDnMatDescr_t descr_b = nullptr;
            cusparseDnMatDescr_t descr_c = nullptr;

            get_cusparse_error(cusparseCreateDnMat(&descr_b, b_rows_stored, b_cols_stored, b_rows_stored, (void *)b.gpu_data(),
                                                   CUDA_R_32F, CUSPARSE_ORDER_COL));
            get_cusparse_error(cusparseCreateDnMat(&descr_c, c_m, c_n, c_m, (void *)c.gpu_data(),
                                                   CUDA_R_32F, CUSPARSE_ORDER_COL));

            // Allocate workspace if needed
            size_t workspace_size = 0;
            get_cusparse_error(cusparseSpMM_bufferSize(handle, opA, opB,
                                                       &alpha, mat_descr, descr_b,
                                                       &beta, descr_c,
                                                       CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &workspace_size));

            void *workspace = nullptr;
            if (workspace_size > 0)
            {
                get_cuda_error(cudaMalloc(&workspace, workspace_size));
            }

            get_cusparse_error(cusparseSpMM(handle, opA, opB,
                                            &alpha, mat_descr, descr_b,
                                            &beta, descr_c,
                                            CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, workspace));

            cudart.synchronize();

            get_cuda_error(cudaMemcpy(c.data(), c.gpu_data(), c.shape().first * c.shape().second * sizeof(float), cudaMemcpyDeviceToHost));

            // Cleanup
            if (workspace)
            {
                get_cuda_error(cudaFree(workspace));
            }

            get_cusparse_error(cusparseDestroySpMat(mat_descr));
            get_cusparse_error(cusparseDestroyDnMat(descr_b));
            get_cusparse_error(cusparseDestroyDnMat(descr_c));
            get_cusparse_error(cusparseDestroy(handle));

            sparse.release_gpu_memory();
        }

    }
} // namespace lahva
