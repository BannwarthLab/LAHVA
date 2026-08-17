#pragma once

#include <vector>
#include <cstring>
#include <cusparse.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "impl/tensor/gpu/gputensor.hpp"
#include "impl/tensor/gpu/matrix.hpp"
#include "impl/tensor/gpu/block-matrix-classes/block-matrix.hpp"
#include "runtime.hpp"

namespace lahva
{
    namespace gpu
    {

        /// @brief Helper struct for GPU sparse block-diagonal matrix data
        template<typename T>
        struct GPUSparseBlockDiagData {
            // Host memory
            std::vector<T> h_values;           ///< Host copy of sparse values
            std::vector<int> h_row_offsets;    ///< Host copy of row offsets
            std::vector<int> h_col_indices;    ///< Host copy of column indices

            // Device memory
            T *d_values = nullptr;             ///< Device pointer to sparse values (CSR/BSR)
            int *d_row_offsets = nullptr;      ///< Device pointer to row offsets
            int *d_col_indices = nullptr;      ///< Device pointer to column indices
            SparseFormat format = SparseFormat::CSR;  ///< Sparse format (CSR or BSR)
        };

        //! @brief GPU-based sparse representation of block diagonal matrix in CSR or BSR format
        //!
        //! Stores a block diagonal matrix directly on the GPU in either Compressed Sparse Row (CSR)
        //! or Block Sparse Row (BSR) format, enabling efficient sparse matrix operations without
        //! CPU-GPU transfers during computation. Inherits from Tensor_ to provide consistent
        //! GPU tensor semantics and memory management.
        template <typename T>
        class SparseMatrix : public virtual Tensor_<T>
        {
        public:
            //! @brief Default constructor
            SparseMatrix(SparseFormat format = SparseFormat::CSR)
                : m_(0), n_(0), nnz_(0), format_(format),
                  bsr_num_block_rows_(0),
                  mat_descr_(nullptr), initialized_(false), num_blocks_(0)
            {
            }

            //! @brief Construct and convert from a GPU BlockMatrix
            SparseMatrix(const CudaRuntime& cudart,
                         const gpu::BlockMatrix_<T>& block_matrix,
                         SparseFormat format = SparseFormat::CSR)
                : m_(0), n_(0), nnz_(0), format_(format),
                  bsr_num_block_rows_(0),
                  mat_descr_(nullptr), initialized_(false), num_blocks_(0)
            {
                convert_from_block_matrix(cudart, block_matrix);
            }

            virtual ~SparseMatrix()
            {
                free_gpu_memory();
            }

            //! @brief Convert from a GPU BlockMatrix to sparse format
            void convert_from_block_matrix(const CudaRuntime& cudart,
                                         const gpu::BlockMatrix_<T>& block_matrix)
            {
                // Free existing GPU memory
                free_gpu_memory();

                num_blocks_ = block_matrix.num_blocks();

                if (num_blocks_ == 0) {
                    return;
                }

                // Calculate total size, nnz, and get matrix dimensions from BlockMatrix
                int64_t total_nnz = 0;

                block_row_sizes_.clear();
                block_col_sizes_.clear();
                block_row_positions_.clear();
                block_col_positions_.clear();

                for (size_t i = 0; i < block_matrix.num_blocks(); ++i)
                {
                    Shape block_shape = block_matrix.get_block_shape(i);
                    int64_t block_rows = block_shape.first;
                    int64_t block_cols = block_shape.second;

                    block_row_sizes_.push_back(block_rows);
                    block_col_sizes_.push_back(block_cols);

                    total_nnz += block_rows * block_cols;
                }

                // Get actual matrix dimensions from BlockMatrix
                Shape matrix_shape = block_matrix.shape();
                m_ = matrix_shape.first;
                n_ = matrix_shape.second;
                nnz_ = total_nnz;

                row_offsets_ = block_matrix.get_row_offsets();
                col_offsets_ = block_matrix.get_col_offsets();

                // Get element-space positions for each block
                for (size_t i = 0; i < block_matrix.num_blocks(); ++i)
                {
                    block_row_positions_.push_back(block_matrix.get_block_row(i));
                    block_col_positions_.push_back(block_matrix.get_block_col(i));
                }

                if (format_ == SparseFormat::CSR)
                {
                    convert_to_csr_gpu(cudart, block_matrix);
                }
                else if (format_ == SparseFormat::BSR)
                {
                    convert_to_bsr_gpu(cudart, block_matrix);
                }

                initialized_ = true;
            }

            int64_t rows() const { return m_; }
            int64_t cols() const { return n_; }
            int64_t nnz() const { return nnz_; }
            int num_blocks() const { return num_blocks_; }
            bool is_initialized() const { return initialized_; }
            SparseFormat get_format() const { return format_; }

            //! @brief Get reference to sparse matrix data (host + device pointers)
            const GPUSparseBlockDiagData<T>& get_sparse_data() const { return sparse_data_; }

            //! @brief Get cusparse sparse matrix descriptor
            cusparseSpMatDescr_t get_descriptor() const { return mat_descr_; }

            //! @brief Allocate GPU memory for sparse matrix data
            void allocate_gpu_memory() {
                if (sparse_data_.d_values == nullptr && sparse_data_.h_values.size() > 0) {
                    get_cuda_error(cudaMalloc(&sparse_data_.d_values, sparse_data_.h_values.size() * sizeof(T)));
                }
                if (sparse_data_.d_row_offsets == nullptr && sparse_data_.h_row_offsets.size() > 0) {
                    get_cuda_error(cudaMalloc(&sparse_data_.d_row_offsets, sparse_data_.h_row_offsets.size() * sizeof(int)));
                }
                if (sparse_data_.d_col_indices == nullptr && sparse_data_.h_col_indices.size() > 0) {
                    get_cuda_error(cudaMalloc(&sparse_data_.d_col_indices, sparse_data_.h_col_indices.size() * sizeof(int)));
                }
            }

            //! @brief Copy sparse data to GPU device
            void transfer_to_device(const CudaRuntime& cudart) { copy2device(cudart); }

            //! @brief Copy sparse data from GPU device to host
            void transfer_to_host(const CudaRuntime& cudart) { copy2host(cudart); }

            //! @brief Free all GPU memory
            void release_gpu_memory() { free_gpu_memory(); }

            //! @brief Reconstruct dense matrix from sparse data (CSR or BSR format)
            Matrix<T> to_dense() const {
                // Initialize with zeros (for structural zeros in sparse layout)
                Matrix<T> dense(Shape{static_cast<unsigned int>(m_), static_cast<unsigned int>(n_)}, (T)0);

                if (format_ == SparseFormat::CSR) {
                    // Reconstruct from CSR format
                    for (int row = 0; row < static_cast<int>(m_); ++row) {
                        int col_start = sparse_data_.h_row_offsets[row];
                        int col_end = sparse_data_.h_row_offsets[row + 1];

                        for (int idx = col_start; idx < col_end; ++idx) {
                            int col = sparse_data_.h_col_indices[idx];
                            dense(row, col) = sparse_data_.h_values[idx];
                        }
                    }
                } else if (format_ == SparseFormat::BSR) {
                    // Reconstruct from BSR format with actual block positions
                    int value_offset = 0;

                    for (int block_idx = 0; block_idx < num_blocks_; ++block_idx) {
                        int block_rows = block_row_sizes_[block_idx];
                        int block_cols = block_col_sizes_[block_idx];
                        int element_row = block_row_positions_[block_idx];
                        int element_col = block_col_positions_[block_idx];

                        // Copy block data in row-major order
                        for (int i = 0; i < block_rows; ++i) {
                            for (int j = 0; j < block_cols; ++j) {
                                dense(element_row + i, element_col + j) =
                                    sparse_data_.h_values[value_offset + i * block_cols + j];
                            }
                        }

                        value_offset += block_rows * block_cols;
                    }
                }

                return dense;
            }

        private:

            // Sparse matrix dimensions
            int64_t m_;              //!< Number of rows
            int64_t n_;              //!< Number of columns
            int64_t nnz_;            //!< Number of non-zero elements
            SparseFormat format_;     //!< Selected sparse format (CSR or BSR)

            // Unified sparse matrix data structure (host + device)
            GPUSparseBlockDiagData<T> sparse_data_;

            // BSR-specific metadata
            int bsr_num_block_rows_;
            std::vector<int> bsr_block_rows_;
            std::vector<int> bsr_block_cols_;

            // cuSPARSE descriptor and state
            cusparseSpMatDescr_t mat_descr_;
            bool initialized_;

            // Block metadata for reconstruction
            int num_blocks_;
            std::vector<int> block_row_sizes_;
            std::vector<int> block_col_sizes_;
            std::vector<int> row_offsets_;
            std::vector<int> col_offsets_;
            std::vector<int> block_row_positions_;  // Element-space row positions for each block
            std::vector<int> block_col_positions_;  // Element-space column positions for each block

            //! @brief Allocate GPU memory and copy host data to device
            void allocate_and_copy_to_device(const CudaRuntime& cudart,
                                             GPUSparseBlockDiagData<T>& sparse_data)
            {
                get_cuda_error(cudaMalloc(&sparse_data.d_values, sparse_data.h_values.size() * sizeof(T)));
                get_cuda_error(cudaMalloc(&sparse_data.d_row_offsets, sparse_data.h_row_offsets.size() * sizeof(int)));
                get_cuda_error(cudaMalloc(&sparse_data.d_col_indices, sparse_data.h_col_indices.size() * sizeof(int)));
                copy2device(cudart);
            }

            //! @brief Free all GPU memory
            void free_gpu_memory()
            {
                if (sparse_data_.d_values)
                {
                    get_cuda_error(cudaFree(sparse_data_.d_values));
                    sparse_data_.d_values = nullptr;
                }
                if (sparse_data_.d_row_offsets)
                {
                    get_cuda_error(cudaFree(sparse_data_.d_row_offsets));
                    sparse_data_.d_row_offsets = nullptr;
                }
                if (sparse_data_.d_col_indices)
                {
                    get_cuda_error(cudaFree(sparse_data_.d_col_indices));
                    sparse_data_.d_col_indices = nullptr;
                }

                if (mat_descr_ != nullptr)
                {
                    cusparseDestroySpMat(mat_descr_);
                    mat_descr_ = nullptr;
                }

                bsr_num_block_rows_ = 0;
                bsr_block_rows_.clear();
                bsr_block_cols_.clear();
                block_row_positions_.clear();
                block_col_positions_.clear();
                initialized_ = false;
                m_ = 0;
                n_ = 0;
                nnz_ = 0;
            }

            //! @brief Convert GPU BlockMatrix to CSR format
            void convert_to_csr_gpu(const CudaRuntime &cudart,
                                   const gpu::BlockMatrix_<T>& block_matrix)
            {
                int total_rows = static_cast<int>(m_);
                int total_nnz = static_cast<int>(nnz_);

                // Prepare sparse data in member variable
                sparse_data_.format = format_;
                sparse_data_.h_values.resize(total_nnz);
                sparse_data_.h_row_offsets.resize(total_rows + 1);
                sparse_data_.h_col_indices.resize(total_nnz);

                // Build CSR structure row by row
                int current_nnz_idx = 0;
                sparse_data_.h_row_offsets[0] = 0;

                for (int row = 0; row < total_rows; ++row)
                {
                    // For each block, check if it contains this row
                    for (size_t block_idx = 0; block_idx < block_matrix.num_blocks(); ++block_idx)
                    {
                        Shape block_shape = block_matrix.get_block_shape(block_idx);
                        int block_rows = block_shape.first;
                        int block_cols = block_shape.second;
                        int block_row_start = block_row_positions_[block_idx];
                        int block_col_start = block_col_positions_[block_idx];

                        int block_row_end = block_row_start + block_rows;

                        // Check if this row is in this block
                        if (row >= block_row_start && row < block_row_end)
                        {
                            // Get raw block data pointer from abstract interface
                            const void* block_data_void = block_matrix.get_block_data(block_idx);
                            const T* block_data = static_cast<const T*>(block_data_void);

                            // Row within this block
                            int local_row = row - block_row_start;

                            // Add all elements from this row of the block to CSR
                            for (int jj = 0; jj < block_cols; ++jj)
                            {
                                // Column-major indexing: element (local_row, jj) at index jj * block_rows + local_row
                                sparse_data_.h_values[current_nnz_idx] = static_cast<double>(block_data[jj * block_rows + local_row]);
                                sparse_data_.h_col_indices[current_nnz_idx] = block_col_start + jj;
                                current_nnz_idx++;
                            }
                        }
                    }

                    sparse_data_.h_row_offsets[row + 1] = current_nnz_idx;
                }

                // Allocate and copy sparse matrix to the GPU
                allocate_and_copy_to_device(cudart, sparse_data_);

                // Synchronize to ensure all transfers complete
                cudart.synchronize();
            }
            
            
            //! @brief Helper function to convert GPU BlockMatrix to BSR format
            void convert_to_bsr_gpu(const CudaRuntime& cudart, const gpu::BlockMatrix_<T>& block_matrix)
            {
                size_t num_blocks_bsr = block_matrix.num_blocks();
                bsr_num_block_rows_ = num_blocks_bsr;
                bsr_block_rows_.clear();
                bsr_block_cols_.clear();

                // Calculate total size for block values
                int total_block_elements = 0;

                for (size_t i = 0; i < num_blocks_bsr; ++i)
                {
                    Shape block_shape = block_matrix.get_block_shape(i);
                    int block_rows = block_shape.first;
                    int block_cols = block_shape.second;

                    bsr_block_rows_.push_back(block_rows);
                    bsr_block_cols_.push_back(block_cols);
                    total_block_elements += block_rows * block_cols;
                }

                // Prepare sparse data in member variable
                sparse_data_.format = format_;
                sparse_data_.h_values.resize(total_block_elements);
                sparse_data_.h_row_offsets.resize(num_blocks_bsr + 1);
                sparse_data_.h_col_indices.resize(num_blocks_bsr);

                // Fill BSR structure
                sparse_data_.h_row_offsets[0] = 0;
                int bsr_value_offset = 0;

                for (int i = 0; i < num_blocks_bsr; ++i)
                {
                    Shape block_shape = block_matrix.get_block_shape(i);
                    int block_rows = block_shape.first;
                    int block_cols = block_shape.second;
                    int block_size = block_rows * block_cols;

                    // Get raw block data pointer from abstract interface
                    const void* block_data_void = block_matrix.get_block_data(i);
                    const T* block_data = static_cast<const T*>(block_data_void);

                    // Transpose block from column-major to row-major for BSR format
                    // Column-major: element(i,j) at index j*block_rows + i
                    // Row-major: element(i,j) at index i*block_cols + j
                    for (int ii = 0; ii < block_rows; ++ii)
                    {
                        for (int jj = 0; jj < block_cols; ++jj)
                        {
                            sparse_data_.h_values[bsr_value_offset + ii * block_cols + jj] =
                                static_cast<double>(block_data[jj * block_rows + ii]);
                        }
                    }

                    sparse_data_.h_row_offsets[i + 1] = sparse_data_.h_row_offsets[i] + 1;
                    sparse_data_.h_col_indices[i] = i;
                    bsr_value_offset += block_size;
                }

                // Allocate and copy sparse matrix to the GPU
                allocate_and_copy_to_device(cudart, sparse_data_);

                // Synchronize to ensure all transfers complete
                cudart.synchronize();
            }

            /// @brief Implement GPU tensor interface: copy data to GPU device
            /// @param cudart CUDA runtime instance
            void copy2device(const CudaRuntime &cudart) const override
            {
                SparseMatrix<T>* self = const_cast<SparseMatrix<T>*>(this);

                // Copy all data to device using async or sync based on runtime configuration
                if (cudart.asyncCopy())
                {
                    get_cuda_error(cudaMemcpyAsync(self->sparse_data_.d_values, self->sparse_data_.h_values.data(),
                                                   self->sparse_data_.h_values.size() * sizeof(T),
                                                   cudaMemcpyHostToDevice, cudart.getStream()));
                    get_cuda_error(cudaMemcpyAsync(self->sparse_data_.d_row_offsets, self->sparse_data_.h_row_offsets.data(),
                                                   self->sparse_data_.h_row_offsets.size() * sizeof(int),
                                                   cudaMemcpyHostToDevice, cudart.getStream()));
                    get_cuda_error(cudaMemcpyAsync(self->sparse_data_.d_col_indices, self->sparse_data_.h_col_indices.data(),
                                                   self->sparse_data_.h_col_indices.size() * sizeof(int),
                                                   cudaMemcpyHostToDevice, cudart.getStream()));
                }
                else
                {
                    get_cuda_error(cudaMemcpy(self->sparse_data_.d_values, self->sparse_data_.h_values.data(),
                                              self->sparse_data_.h_values.size() * sizeof(T),
                                              cudaMemcpyHostToDevice));
                    get_cuda_error(cudaMemcpy(self->sparse_data_.d_row_offsets, self->sparse_data_.h_row_offsets.data(),
                                              self->sparse_data_.h_row_offsets.size() * sizeof(int),
                                              cudaMemcpyHostToDevice));
                    get_cuda_error(cudaMemcpy(self->sparse_data_.d_col_indices, self->sparse_data_.h_col_indices.data(),
                                              self->sparse_data_.h_col_indices.size() * sizeof(int),
                                              cudaMemcpyHostToDevice));
                }
            }

            /// @brief Implement GPU tensor interface: copy data from GPU to host
            /// @param cudart CUDA runtime instance
            void copy2host(const CudaRuntime &cudart) override
            {
                // Copy all data from device back to host using async or sync based on runtime configuration
                if (cudart.asyncCopy())
                {
                    get_cuda_error(cudaMemcpyAsync(sparse_data_.h_values.data(), sparse_data_.d_values,
                                                   sparse_data_.h_values.size() * sizeof(T),
                                                   cudaMemcpyDeviceToHost, cudart.getStream()));
                    get_cuda_error(cudaMemcpyAsync(sparse_data_.h_row_offsets.data(), sparse_data_.d_row_offsets,
                                                   sparse_data_.h_row_offsets.size() * sizeof(int),
                                                   cudaMemcpyDeviceToHost, cudart.getStream()));
                    get_cuda_error(cudaMemcpyAsync(sparse_data_.h_col_indices.data(), sparse_data_.d_col_indices,
                                                   sparse_data_.h_col_indices.size() * sizeof(int),
                                                   cudaMemcpyDeviceToHost, cudart.getStream()));
                }
                else
                {
                    get_cuda_error(cudaMemcpy(sparse_data_.h_values.data(), sparse_data_.d_values,
                                              sparse_data_.h_values.size() * sizeof(T),
                                              cudaMemcpyDeviceToHost));
                    get_cuda_error(cudaMemcpy(sparse_data_.h_row_offsets.data(), sparse_data_.d_row_offsets,
                                              sparse_data_.h_row_offsets.size() * sizeof(int),
                                              cudaMemcpyDeviceToHost));
                    get_cuda_error(cudaMemcpy(sparse_data_.h_col_indices.data(), sparse_data_.d_col_indices,
                                              sparse_data_.h_col_indices.size() * sizeof(int),
                                              cudaMemcpyDeviceToHost));
                }
            }

            /// @brief Implement GPU tensor interface: check if data is allocated on device
            /// @return true if sparse data is on GPU device
            bool alloc_on_device() const override
            {
                return initialized_ && (sparse_data_.d_values != nullptr);
            }

            /// @brief Implement GPU tensor interface: get pointer to GPU data (const)
            /// @return pointer to GPU device data
            T *gpu_data() const override
            {
                return sparse_data_.d_values;
            }

            /// @brief Implement GPU tensor interface: get pointer to GPU data (non-const)
            /// @return pointer to GPU device data
            T *gpu_data() override
            {
                return sparse_data_.d_values;
            }

            /// @brief Implement Tensor interface: get total number of elements (sparse: nnz)
            /// @return number of non-zero elements
            size_t size() override { return nnz_; }

            /// @brief Implement Tensor interface: get total number of elements (const)
            /// @return number of non-zero elements
            size_t size() const override { return nnz_; }

            /// @brief Implement Tensor interface: get pointer to host data
            /// @return pointer to sparse values array
            T *data() override { return sparse_data_.h_values.data(); }

            /// @brief Implement Tensor interface: get const pointer to host data
            /// @return const pointer to sparse values array
            T *data() const override { return const_cast<T*>(sparse_data_.h_values.data()); }
        };

    } // namespace gpu
} // namespace lahva
