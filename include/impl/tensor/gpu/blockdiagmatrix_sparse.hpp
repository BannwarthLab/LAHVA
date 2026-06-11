#pragma once

#include <vector>
#include <cstring>
#include <cusparse.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "impl/tensor/gpu/gputensor.hpp"
#include "impl/tensor/gpu/matrix.hpp"
#include "impl/tensor/gpu/sparse_format.hpp"
#include "runtime.hpp"

namespace lahva
{
    namespace gpu
    {
        
        //! @brief GPU-based sparse representation of block diagonal matrix in CSR or BSR format
        //! 
        //! This class stores a block diagonal matrix directly on the GPU in either Compressed Sparse Row (CSR)
        //! or BSR format, enabling efficient sparse matrix operations without CPU-GPU transfers during computation.
        //!
        template <typename T>
        class BlockDiagMatrixSparse
        {
        protected:
            int64_t m_;              //!< Number of rows
            int64_t n_;              //!< Number of columns
            int64_t nnz_;            //!< Number of non-zero elements
            
            SparseFormat format_;     //!< Selected sparse format (CSR, or BSR)
            
            // CSR format members
            double *d_values_;        //!< GPU device pointer to non-zero values (CSR)
            int *d_row_offsets_;      //!< GPU device pointer to row offsets (CSR format)
            int *d_col_indices_;      //!< GPU device pointer to column indices (CSR format)
            
            // BSR (Block Sparse Row) format members
            double *d_bsr_values_;            //!< GPU device pointer to BSR block values
            int *d_bsr_row_offsets_;          //!< GPU device pointer to BSR block row offsets
            int *d_bsr_col_indices_;          //!< GPU device pointer to BSR block column indices
            int bsr_num_block_rows_;          //!< Number of block rows in BSR
            std::vector<int> bsr_block_rows_; //!< Row dimensions of each block
            std::vector<int> bsr_block_cols_; //!< Column dimensions of each block
            
            cusparseSpMatDescr_t mat_descr_;  //!< cuSPARSE sparse matrix descriptor (CSR)
            bool initialized_;     //!< Whether GPU data has been allocated
            
            // Metadata for reconstruction/debugging
            int num_blocks_;
            std::vector<int> block_row_sizes_;
            std::vector<int> block_col_sizes_;
            std::vector<int> row_offsets_;
            std::vector<int> col_offsets_;
        public:
            BlockDiagMatrixSparse(SparseFormat format = SparseFormat::CSR) 
                : m_(0), n_(0), nnz_(0), format_(format), 
                  d_values_(nullptr), d_row_offsets_(nullptr), d_col_indices_(nullptr),
                  d_bsr_values_(nullptr), d_bsr_row_offsets_(nullptr), d_bsr_col_indices_(nullptr),
                  bsr_num_block_rows_(0),
                  mat_descr_(nullptr), initialized_(false), num_blocks_(0)
            {
            }
            
            //! @brief Construct from a GPU BlockDiagMatrix
            BlockDiagMatrixSparse(const gpu::BlockDiagMatrix<T>& bd_matrix,
                                SparseFormat format = SparseFormat::CSR)
                : m_(0), n_(0), nnz_(0), format_(format),
                  d_values_(nullptr), d_row_offsets_(nullptr), d_col_indices_(nullptr),
                  d_bsr_values_(nullptr), d_bsr_row_offsets_(nullptr), d_bsr_col_indices_(nullptr),
                  bsr_num_block_rows_(0),
                  mat_descr_(nullptr), initialized_(false), num_blocks_(0)
            {
                convert_from_gpu_block_diag(bd_matrix);
            }
            
            virtual ~BlockDiagMatrixSparse()
            {
                free_gpu_memory();
            }
            
           
            void convert_from_gpu_block_diag(const gpu::BlockDiagMatrix<T>& bd_matrix)
            {
                // Free existing GPU memory
                free_gpu_memory();
                
                num_blocks_ = bd_matrix.num_blocks();
                
                if (num_blocks_ == 0) {
                    return;
                }
                
                // Calculate total size and nnz
                int64_t total_rows = 0;
                int64_t total_nnz = 0;
                
                block_row_sizes_.clear();
                block_col_sizes_.clear();
                
                for (size_t i = 0; i < bd_matrix.num_blocks(); ++i)
                {
                    const gpu::Matrix<T>& block = bd_matrix.get_block(i);
                    int64_t block_rows = block.shape().first;
                    int64_t block_cols = block.shape().second;
                    
                    block_row_sizes_.push_back(block_rows);
                    block_col_sizes_.push_back(block_cols);
                    
                    total_rows += block_rows;
                    total_nnz += block_rows * block_cols;
                }
                
                m_ = total_rows;
                n_ = total_rows;
                nnz_ = total_nnz;
                
                row_offsets_ = bd_matrix.get_row_offsets();
                col_offsets_ = bd_matrix.get_col_offsets();
                
                if (format_ == SparseFormat::CSR)
                {
                    convert_to_csr_gpu(bd_matrix, total_rows, total_nnz);
                }
                else if (format_ == SparseFormat::BSR)
                {
                    convert_to_bsr_gpu(bd_matrix);
                }
                
                initialized_ = true;
            }
            
            //! @brief Helper function to convert GPU BlockDiagMatrix to CSR format
            void convert_to_csr_gpu(const gpu::BlockDiagMatrix<T>& bd_matrix, 
                                   int64_t total_rows, int64_t total_nnz)
            {
                // Allocate host arrays
                std::vector<double> h_values(total_nnz);
                std::vector<int> h_row_offsets(total_rows + 1);
                std::vector<int> h_col_indices(total_nnz);
                
                // Fill CSR structure
                // Note: Matrices are stored in column-major order
                // Block data indexing: element(i,j) is at data[j * block_rows + i]
                int row_idx = 0;
                int nnz_idx = 0;
                h_row_offsets[0] = 0;
                
                for (size_t i = 0; i < bd_matrix.num_blocks(); ++i)
                {
                    const gpu::Matrix<T>& block = bd_matrix.get_block(i);
                    int block_rows = block.shape().first;
                    int block_cols = block.shape().second;
                    
                    // Copy block data from GPU to host temporarily
                    std::vector<T> h_block_data(block_rows * block_cols);
                    get_cuda_error(cudaMemcpy(h_block_data.data(), block.data(), 
                                         block_rows * block_cols * sizeof(T), 
                                         cudaMemcpyDeviceToHost));
                    
                    // Iterate through block in row-major order to build CSR
                    for (int ii = 0; ii < block_rows; ++ii)
                    {
                        for (int jj = 0; jj < block_cols; ++jj)
                        {
                            // Column-major indexing: element (ii, jj) at index jj * block_rows + ii
                            h_values[nnz_idx] = static_cast<double>(h_block_data[jj * block_rows + ii]);
                            int global_col = bd_matrix.get_col_offsets()[i] + jj;
                            h_col_indices[nnz_idx] = global_col;
                            nnz_idx++;
                        }
                        h_row_offsets[row_idx + 1] = nnz_idx;
                        row_idx++;
                    }
                }
                
                // Allocate GPU memory
                get_cuda_error(cudaMalloc(&d_values_, total_nnz * sizeof(double)));
                get_cuda_error(cudaMalloc(&d_row_offsets_, (total_rows + 1) * sizeof(int)));
                get_cuda_error(cudaMalloc(&d_col_indices_, total_nnz * sizeof(int)));
                
                // Copy to GPU
                get_cuda_error(cudaMemcpy(d_values_, h_values.data(), total_nnz * sizeof(double), 
                                     cudaMemcpyHostToDevice));
                get_cuda_error(cudaMemcpy(d_row_offsets_, h_row_offsets.data(), (total_rows + 1) * sizeof(int),
                                     cudaMemcpyHostToDevice));
                get_cuda_error(cudaMemcpy(d_col_indices_, h_col_indices.data(), total_nnz * sizeof(int),
                                     cudaMemcpyHostToDevice));
            }
            
            
            //! @brief Helper function to convert GPU BlockDiagMatrix to BSR format
            void convert_to_bsr_gpu(const gpu::BlockDiagMatrix<T>& bd_matrix)
            {
                int num_blocks_bsr = bd_matrix.num_blocks();
                bsr_num_block_rows_ = num_blocks_bsr;
                bsr_block_rows_.clear();
                bsr_block_cols_.clear();
                
                // Calculate total size for block values (assumed uniform block size)
                int total_block_elements = 0;
                
                for (size_t i = 0; i < num_blocks_bsr; ++i)
                {
                    const gpu::Matrix<T>& block = bd_matrix.get_block(i);
                    int block_rows = block.shape().first;
                    int block_cols = block.shape().second;
                    
                    bsr_block_rows_.push_back(block_rows);
                    bsr_block_cols_.push_back(block_cols);
                    total_block_elements += block_rows * block_cols;
                }
                
                // Allocate host arrays for BSR format
                std::vector<double> h_bsr_values(total_block_elements);
                std::vector<int> h_bsr_row_offsets(num_blocks_bsr + 1);
                std::vector<int> h_bsr_col_indices(num_blocks_bsr);
                
                // Fill BSR structure
                h_bsr_row_offsets[0] = 0;
                int bsr_value_offset = 0;
                
                for (int i = 0; i < num_blocks_bsr; ++i)
                {
                    const gpu::Matrix<T>& block = bd_matrix.get_block(i);
                    int block_rows = block.shape().first;
                    int block_cols = block.shape().second;
                    int block_size = block_rows * block_cols;
                    
                    // Copy block from GPU to host
                    std::vector<T> h_block(block_size);
                    get_cuda_error(cudaMemcpy(h_block.data(), block.data(), block_size * sizeof(T), 
                                         cudaMemcpyDeviceToHost));
                    
                    // Transpose block from column-major to row-major for BSR format
                    // Column-major: element(i,j) at index j*block_rows + i
                    // Row-major: element(i,j) at index i*block_cols + j
                    for (int ii = 0; ii < block_rows; ++ii)
                    {
                        for (int jj = 0; jj < block_cols; ++jj)
                        {
                            h_bsr_values[bsr_value_offset + ii * block_cols + jj] = 
                                static_cast<double>(h_block[jj * block_rows + ii]);
                        }
                    }
                    
                    h_bsr_row_offsets[i + 1] = h_bsr_row_offsets[i] + 1; // One block per block row
                    h_bsr_col_indices[i] = i; // Diagonal block at position i
                    bsr_value_offset += block_size;
                }
                
                // Allocate GPU memory for BSR
                get_cuda_error(cudaMalloc(&d_bsr_values_, total_block_elements * sizeof(double)));
                get_cuda_error(cudaMalloc(&d_bsr_row_offsets_, (num_blocks_bsr + 1) * sizeof(int)));
                get_cuda_error(cudaMalloc(&d_bsr_col_indices_, num_blocks_bsr * sizeof(int)));
                
                // Copy BSR to GPU
                get_cuda_error(cudaMemcpy(d_bsr_values_, h_bsr_values.data(), total_block_elements * sizeof(double), 
                                     cudaMemcpyHostToDevice));
                get_cuda_error(cudaMemcpy(d_bsr_row_offsets_, h_bsr_row_offsets.data(), (num_blocks_bsr + 1) * sizeof(int),
                                     cudaMemcpyHostToDevice));
                get_cuda_error(cudaMemcpy(d_bsr_col_indices_, h_bsr_col_indices.data(), num_blocks_bsr * sizeof(int),
                                     cudaMemcpyHostToDevice));
            }
            
            //! @param[in] handle cuSPARSE handle
            //! @param[in] alpha Scalar multiplier for A * B
            //! @param[in] B Dense matrix (k x n) in column-major order on GPU
            //! @param[in] beta Scalar multiplier for C
            //! @param[in,out] C Result matrix (m x n) in column-major order on GPU
            void multiply(cusparseHandle_t handle, double alpha, const double *d_b, int64_t b_rows, int64_t b_cols,
                         double beta, double *d_c, int64_t c_rows, int64_t c_cols)
            {
                if (!initialized_ || m_ == 0)
                {
                    return;
                }
                
                if (format_ == SparseFormat::CSR)
                {
                    multiply_csr(handle, alpha, d_b, b_rows, b_cols, beta, d_c, c_rows, c_cols);
                }
                else if (format_ == SparseFormat::BSR)
                {
                    multiply_bsr(handle, alpha, d_b, b_rows, b_cols, beta, d_c, c_rows, c_cols);
                }
            }
            
            //! @brief CSR format matrix multiplication
            void multiply_csr(cusparseHandle_t handle, double alpha, const double *d_b, int64_t b_rows, int64_t b_cols,
                             double beta, double *d_c, int64_t c_rows, int64_t c_cols)
            {
                // Validate dimensions
                if (m_ == 0 || n_ == 0 || nnz_ == 0) {
                    throw std::runtime_error("BlockDiagMatrixSparse not properly initialized: m_=" + std::to_string(m_) + 
                                            ", n_=" + std::to_string(n_) + ", nnz_=" + std::to_string(nnz_));
                }
                if (m_ != c_rows) {
                    throw std::runtime_error("Dimension mismatch: sparse matrix rows (" + std::to_string(m_) + 
                                            ") != result matrix rows (" + std::to_string(c_rows) + ")");
                }
                if (n_ != b_rows) {
                    throw std::runtime_error("Dimension mismatch: sparse matrix cols (" + std::to_string(n_) + 
                                            ") != dense matrix rows (" + std::to_string(b_rows) + ")");
                }
                
                // Create sparse descriptor if not already created
                if (mat_descr_ == nullptr)
                {
                    cusparseCreateCsr(&mat_descr_, m_, n_, nnz_,
                                     d_row_offsets_, d_col_indices_, d_values_,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
                }
                
                // Create dense matrix descriptors for B and C
                cusparseDnMatDescr_t descr_b, descr_c;
                cusparseCreateDnMat(&descr_b, b_rows, b_cols, b_rows, (void*)d_b,
                                   CUDA_R_64F, CUSPARSE_ORDER_COL);
                cusparseCreateDnMat(&descr_c, c_rows, c_cols, c_rows, (void*)d_c,
                                   CUDA_R_64F, CUSPARSE_ORDER_COL);
                
                // Allocate workspace
                size_t workspace_size = 0;
                cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                       CUSPARSE_OPERATION_NON_TRANSPOSE,
                                       &alpha, mat_descr_, descr_b,
                                       &beta, descr_c,
                                       CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, &workspace_size);
                
                void *workspace = nullptr;
                if (workspace_size > 0)
                {
                    get_cuda_error(cudaMalloc(&workspace, workspace_size));
                }
                
                // Perform sparse matrix-dense matrix multiplication
                cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, mat_descr_, descr_b,
                            &beta, descr_c,
                            CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, workspace);
                
                // Cleanup
                if (workspace)
                {
                    get_cuda_error(cudaFree(workspace));
                }
                cusparseDestroyDnMat(descr_b);
                cusparseDestroyDnMat(descr_c);
            }
            
            
            //! @brief BSR format matrix multiplication
            void multiply_bsr(cusparseHandle_t handle, double alpha, const double *d_b, int64_t b_rows, int64_t b_cols,
                             double beta, double *d_c, int64_t c_rows, int64_t c_cols)
            {
                // Validate dimensions
                if (m_ == 0 || n_ == 0 || nnz_ == 0) {
                    throw std::runtime_error("BlockDiagMatrixSparse not properly initialized: m_=" + std::to_string(m_) + 
                                            ", n_=" + std::to_string(n_) + ", nnz_=" + std::to_string(nnz_));
                }
                if (m_ != c_rows) {
                    throw std::runtime_error("Dimension mismatch: sparse matrix rows (" + std::to_string(m_) + 
                                            ") != result matrix rows (" + std::to_string(c_rows) + ")");
                }
                if (n_ != b_rows) {
                    throw std::runtime_error("Dimension mismatch: sparse matrix cols (" + std::to_string(n_) + 
                                            ") != dense matrix rows (" + std::to_string(b_rows) + ")");
                }
                
                // Get block size from stored block dimensions (assuming uniform blocks)
                int blockDim = bsr_block_rows_.empty() ? 0 : bsr_block_rows_[0];
                int num_block_rows = bsr_num_block_rows_;
                int nnz_blocks = num_block_rows;  // Block diagonal matrix has num_blocks non-zero blocks
                
                // Create BSR sparse descriptor
                // NOTE: For BSR format, brows and bcols are the number of BLOCK ROWS/COLS, not total rows/cols!
                // The actual matrix dimensions are (brows * rowBlockSize) x (bcols * colBlockSize)
                // BSR format requires CUSPARSE_ORDER_ROW
                cusparseSpMatDescr_t mat_descr_bsr = nullptr;
                cusparseCreateBsr(&mat_descr_bsr,
                                 (int64_t)num_block_rows,  // brows (number of block rows, not total rows!)
                                 (int64_t)num_block_rows,  // bcols (number of block cols, not total cols!)
                                 (int64_t)nnz_blocks,  // bnnz
                                 (int64_t)blockDim,  // rowBlockSize
                                 (int64_t)blockDim,  // colBlockSize
                                 d_bsr_row_offsets_,  // bsrRowOffsets
                                 d_bsr_col_indices_,  // bsrColInd
                                 d_bsr_values_,  // bsrValues
                                 CUSPARSE_INDEX_32I,  // bsrRowOffsetsType
                                 CUSPARSE_INDEX_32I,  // bsrColIndType
                                 CUSPARSE_INDEX_BASE_ZERO,  // idxBase
                                 CUDA_R_64F,  // valueType
                                 CUSPARSE_ORDER_ROW);  // order - BSR requires row-major!
                
                // Create dense matrix descriptors for B and C
                cusparseDnMatDescr_t descr_b, descr_c;
                cusparseCreateDnMat(&descr_b, b_rows, b_cols, b_rows, (void*)d_b,
                                   CUDA_R_64F, CUSPARSE_ORDER_COL);
                cusparseCreateDnMat(&descr_c, c_rows, c_cols, c_rows, (void*)d_c,
                                   CUDA_R_64F, CUSPARSE_ORDER_COL);
                
                // Allocate workspace
                size_t workspace_size = 0;
                cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                       CUSPARSE_OPERATION_NON_TRANSPOSE,
                                       &alpha, mat_descr_bsr, descr_b,
                                       &beta, descr_c,
                                       CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, &workspace_size);
                
                void *workspace = nullptr;
                if (workspace_size > 0)
                {
                    get_cuda_error(cudaMalloc(&workspace, workspace_size));
                }
                
                // Perform sparse matrix-dense matrix multiplication using BSR format
                cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, mat_descr_bsr, descr_b,
                            &beta, descr_c,
                            CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, workspace);
                
                // Cleanup
                if (workspace)
                {
                    get_cuda_error(cudaFree(workspace));
                }
                cusparseDestroyDnMat(descr_b);
                cusparseDestroyDnMat(descr_c);
                cusparseDestroySpMat(mat_descr_bsr);
            }
            
           
            //! @param[in] handle cuSPARSE handle
            //! @param[in] alpha Scalar multiplier for A * x
            //! @param[in] d_b Dense vector (n,) on GPU
            //! @param[in] beta Scalar multiplier for y
            //! @param[in,out] d_c Result vector (m,) on GPU
            void multiply_vec(cusparseHandle_t handle, double alpha, const double *d_b, int64_t b_size,
                         double beta, double *d_c, int64_t c_size)
            {
                if (!initialized_ || m_ == 0)
                {
                    return;
                }
                
                if (format_ == SparseFormat::CSR)
                {
                    multiply_vec_csr(handle, alpha, d_b, b_size, beta, d_c, c_size);
                }
                else if (format_ == SparseFormat::BSR)
                {
                    multiply_vec_bsr(handle, alpha, d_b, b_size, beta, d_c, c_size);
                }
            }
            
            //! @brief CSR format matrix-vector multiplication
            void multiply_vec_csr(cusparseHandle_t handle, double alpha, const double *d_b, int64_t b_size,
                                 double beta, double *d_c, int64_t c_size)
            {
                // Validate dimensions
                if (m_ == 0 || n_ == 0 || nnz_ == 0) {
                    throw std::runtime_error("BlockDiagMatrixSparse not properly initialized: m_=" + std::to_string(m_) + 
                                            ", n_=" + std::to_string(n_) + ", nnz_=" + std::to_string(nnz_));
                }
                if (m_ != c_size) {
                    throw std::runtime_error("Dimension mismatch: sparse matrix rows (" + std::to_string(m_) + 
                                            ") != result vector size (" + std::to_string(c_size) + ")");
                }
                if (n_ != b_size) {
                    throw std::runtime_error("Dimension mismatch: sparse matrix cols (" + std::to_string(n_) + 
                                            ") != dense vector size (" + std::to_string(b_size) + ")");
                }
                
                // Create sparse descriptor if not already created
                if (mat_descr_ == nullptr)
                {
                    cusparseCreateCsr(&mat_descr_, m_, n_, nnz_,
                                     d_row_offsets_, d_col_indices_, d_values_,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
                }
                
                // Create dense vector descriptors for x and y
                cusparseDnVecDescr_t descr_b, descr_c;
                cusparseCreateDnVec(&descr_b, b_size, (void*)d_b, CUDA_R_64F);
                cusparseCreateDnVec(&descr_c, c_size, (void*)d_c, CUDA_R_64F);
                
                // Allocate workspace
                size_t workspace_size = 0;
                cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                       &alpha, mat_descr_, descr_b,
                                       &beta, descr_c,
                                       CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &workspace_size);
                
                void *workspace = nullptr;
                if (workspace_size > 0)
                {
                    get_cuda_error(cudaMalloc(&workspace, workspace_size));
                }
                
                // Perform sparse matrix-vector multiplication: y = alpha * A * x + beta * y
                cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, mat_descr_, descr_b,
                            &beta, descr_c,
                            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, workspace);
                
                // Cleanup
                if (workspace)
                {
                    get_cuda_error(cudaFree(workspace));
                }
                cusparseDestroyDnVec(descr_b);
                cusparseDestroyDnVec(descr_c);
            }
            
            
            //! @brief BSR format matrix-vector multiplication
            void multiply_vec_bsr(cusparseHandle_t handle, double alpha, const double *d_b, int64_t b_size,
                                 double beta, double *d_c, int64_t c_size)
            {
                // Validate dimensions
                if (m_ == 0 || n_ == 0 || nnz_ == 0) {
                    throw std::runtime_error("BlockDiagMatrixSparse not properly initialized: m_=" + std::to_string(m_) + 
                                            ", n_=" + std::to_string(n_) + ", nnz_=" + std::to_string(nnz_));
                }
                if (m_ != c_size) {
                    throw std::runtime_error("Dimension mismatch: sparse matrix rows (" + std::to_string(m_) + 
                                            ") != result vector size (" + std::to_string(c_size) + ")");
                }
                if (n_ != b_size) {
                    throw std::runtime_error("Dimension mismatch: sparse matrix cols (" + std::to_string(n_) + 
                                            ") != dense vector size (" + std::to_string(b_size) + ")");
                }
                
                // Get block size from stored block dimensions (assuming uniform blocks)
                int blockDim = bsr_block_rows_.empty() ? 0 : bsr_block_rows_[0];
                int num_block_rows = bsr_num_block_rows_;
                int nnz_blocks = num_block_rows;  // Block diagonal matrix has num_blocks non-zero blocks
                
                // Create matrix descriptor for legacy BSRMV API
                cusparseMatDescr_t mat_descr = nullptr;
                cusparseCreateMatDescr(&mat_descr);
                cusparseSetMatType(mat_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
                cusparseSetMatIndexBase(mat_descr, CUSPARSE_INDEX_BASE_ZERO);
                
                // Perform sparse matrix-vector multiplication using cusparseDbsrmv (legacy API)
                // y = alpha * A * x + beta * y
                cusparseDbsrmv(handle,
                              CUSPARSE_DIRECTION_ROW,
                              CUSPARSE_OPERATION_NON_TRANSPOSE,
                              num_block_rows,  // mb (number of block rows)
                              num_block_rows,  // nb (number of block cols)
                              nnz_blocks,      // nnzb (number of non-zero blocks)
                              &alpha,
                              mat_descr,
                              d_bsr_values_,
                              d_bsr_row_offsets_,
                              d_bsr_col_indices_,
                              blockDim,
                              d_b,
                              &beta,
                              d_c);
                
                // Cleanup
                cusparseDestroyMatDescr(mat_descr);
            }
            int64_t rows() const { return m_; }
            int64_t cols() const { return n_; }
            int64_t nnz() const { return nnz_; }
            
            //! @brief Get number of blocks
            int num_blocks() const { return num_blocks_; }
            
            //! @brief Check if data is initialized on GPU
            bool is_initialized() const { return initialized_; }
            
            //! @brief Get the sparse format being used
            SparseFormat get_format() const { return format_; }
            
        private:
            //! @brief Free all GPU memory
            void free_gpu_memory()
            {
                // Free CSR format memory
                if (d_values_)
                {
                    get_cuda_error(cudaFree(d_values_));
                    d_values_ = nullptr;
                }
                if (d_row_offsets_)
                {
                    get_cuda_error(cudaFree(d_row_offsets_));
                    d_row_offsets_ = nullptr;
                }
                if (d_col_indices_)
                {
                    get_cuda_error(cudaFree(d_col_indices_));
                    d_col_indices_ = nullptr;
                }
            
                // Free BSR format memory
                if (d_bsr_values_)
                {
                    get_cuda_error(cudaFree(d_bsr_values_));
                    d_bsr_values_ = nullptr;
                }
                if (d_bsr_row_offsets_)
                {
                    get_cuda_error(cudaFree(d_bsr_row_offsets_));
                    d_bsr_row_offsets_ = nullptr;
                }
                if (d_bsr_col_indices_)
                {
                    get_cuda_error(cudaFree(d_bsr_col_indices_));
                    d_bsr_col_indices_ = nullptr;
                }
                bsr_num_block_rows_ = 0;
                bsr_block_rows_.clear();
                bsr_block_cols_.clear();
                
                // Free sparse matrix descriptor
                if (initialized_)
                {
                    if (mat_descr_ != nullptr)
                    {
                        cusparseDestroySpMat(mat_descr_);
                        mat_descr_ = nullptr;
                    }
                    initialized_ = false;
                }
                m_ = 0;
                n_ = 0;
                nnz_ = 0;
            }
        };
        
    } // namespace gpu
} // namespace lahva
